/**
 * Copyright 2018 Wei Dai <wdai3141@gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <include/cufhe.h>
#include <include/bootstrap_gpu.cuh>
#include <include/ntt_gpu/ntt.cuh>
#include <include/details/error_gpu.cuh>
#include <include/details/math.h>

#include <iostream>
using namespace std;

// Debug mode
// #undef DEBUG_ROTATE_A0 // Plaintext Blind Rotate with rotating only b
// #undef DEBUG_ROTATE    // Plaintext Blind Rotate without NTT/INTT

namespace cufhe
{

  using BootstrappingKeyNTT = TGSWSampleArray_T<FFP>;
  BootstrappingKeyNTT *bk_ntt = nullptr;
  MemoryDeleter bk_ntt_deleter = nullptr;
  KeySwitchingKey *ksk_dev = nullptr;
  MemoryDeleter ksk_dev_deleter = nullptr;
  CuNTTHandler<LARGE_N, NEGATIVE_CYCLIC_CONVOLUTION> *ntt_handler = nullptr;
  __device__ FFP sh[LARGE_N * ((SMALL_K + 1) * SMALL_L + 4 * (SMALL_K + 1))];

  template <uint32_t length, ConvKind conv_kind = NEGATIVE_CYCLIC_CONVOLUTION>
  __global__ void __BootstrappingKeyToNTT__(BootstrappingKeyNTT bk_ntt, BootstrappingKey bk,
                                            CuNTTHandler<length, NEGATIVE_CYCLIC_CONVOLUTION> ntt,
                                            uint32_t i, uint32_t j, uint32_t k,
                                            uint32_t stage)
  {
    TGSWSample tgsw;
    bk.ExtractTGSWSample(&tgsw, i);
    TLWESample tlwe;
    tgsw.ExtractTLWESample(&tlwe, j);
    Torus *poly_in = tlwe.ExtractPoly(k);
    TGSWSample_T<FFP> tgsw_ntt;
    bk_ntt.ExtractTGSWSample(&tgsw_ntt, i);
    TLWESample_T<FFP> tlwe_ntt;
    tgsw_ntt.ExtractTLWESample(&tlwe_ntt, j);
    FFP *poly_out = tlwe_ntt.ExtractPoly(k);
    if (stage == 0)
      ntt.template NTT_<Torus, N_BUTTERFLY>(poly_out, poly_in, stage, 0);
    else
      ntt.template NTT_<FFP, N_BUTTERFLY>(poly_out, poly_out, stage, 0);
    __threadfence();
    __syncthreads();
  }

  void BootstrappingKeyToNTT(const BootstrappingKey *bk)
  {
    BootstrappingKey *d_bk;
    d_bk = new BootstrappingKey(bk->n(), bk->k(), bk->l(), bk->w(), bk->t());
    std::pair<void *, MemoryDeleter> pair;
    pair = AllocatorGPU::New(d_bk->SizeMalloc());
    d_bk->set_data((BootstrappingKey::PointerType)pair.first);
    MemoryDeleter d_bk_deleter = pair.second;
    CuSafeCall(cudaMemcpy(d_bk->data(), bk->data(), d_bk->SizeMalloc(),
                          cudaMemcpyHostToDevice));

    Assert(bk_ntt == nullptr);
    bk_ntt = new BootstrappingKeyNTT(bk->n(), bk->k(), bk->l(), bk->w(), bk->t());
    pair = AllocatorGPU::New(bk_ntt->SizeMalloc());
    bk_ntt->set_data((BootstrappingKeyNTT::PointerType)pair.first);
    bk_ntt_deleter = pair.second;

    Assert(ntt_handler == nullptr);
    ntt_handler = new CuNTTHandler<LARGE_N, NEGATIVE_CYCLIC_CONVOLUTION>();
    ntt_handler->Create(LARGE_N);
    ntt_handler->CreateConstant();
    cudaDeviceSynchronize();
    CuCheckError();

#define MAX_THREAD_PER_BLOCK 1024 // 512

#define NBLOCK (LARGE_N / (2 * N_BUTTERFLY) < MAX_THREAD_PER_BLOCK ? LARGE_N / (2 * N_BUTTERFLY) : MAX_THREAD_PER_BLOCK)

#if 0
#define NGRID_X ((LARGE_N / (2 * N_BUTTERFLY)) % MAX_THREAD_PER_BLOCK == 0 ? (LARGE_N / (2 * N_BUTTERFLY)) / MAX_THREAD_PER_BLOCK : (LARGE_N / (2 * N_BUTTERFLY)) / MAX_THREAD_PER_BLOCK + 1)
#endif
#define NGRID_X (((LARGE_N / (2 * N_BUTTERFLY)) + MAX_THREAD_PER_BLOCK - 1) / MAX_THREAD_PER_BLOCK)

    dim3 grid(NGRID_X);
    dim3 block(NBLOCK);

    for (int i = 0; i < bk->t(); i++)
    {
      for (int j = 0; j < (bk->k() + 1) * bk->l(); j++)
      {
        for (int k = 0; k < 2; k++)
        {
          for (uint32_t s = 0; s < log2f(LARGE_N); s++)
          {
            __BootstrappingKeyToNTT__<LARGE_N><<<grid, block>>>(*bk_ntt, *d_bk, *ntt_handler, i, j, k, s);
            if (NGRID_X > 1)
              cudaDeviceSynchronize();
            CuCheckError();
          }
        }
      }
    }
    printf("BKNTT generated\n");

    d_bk_deleter(d_bk->data());
    delete d_bk;
  }

  void DeleteBootstrappingKeyNTT()
  {
    bk_ntt_deleter(bk_ntt->data());
    delete bk_ntt;
    bk_ntt = nullptr;

    ntt_handler->Destroy();
    delete ntt_handler;
  }

  void KeySwitchingKeyToDevice(const KeySwitchingKey *ksk)
  {
    Assert(ksk_dev == nullptr);
    ksk_dev = new KeySwitchingKey(ksk->n(), ksk->l(), ksk->w(), ksk->m());
    std::pair<void *, MemoryDeleter> pair;
    pair = AllocatorGPU::New(ksk_dev->SizeMalloc());
    ksk_dev->set_data((KeySwitchingKey::PointerType)pair.first);
    ksk_dev_deleter = pair.second;
    CuSafeCall(cudaMemcpy(ksk_dev->data(), ksk->data(), ksk->SizeMalloc(),
                          cudaMemcpyHostToDevice));
  }

  void DeleteKeySwitchingKey()
  {
    ksk_dev_deleter(ksk_dev->data());
    delete ksk_dev;
    ksk_dev = nullptr;
  }

  __device__ inline uint32_t ModSwitch2048(uint32_t a)
  {
    return (((uint64_t)a << 32) + (0x1UL << 52)) >> 53;
  }

  __device__ inline uint32_t ModSwitch_2N(uint32_t a, uint32_t log2N)
  {
    uint32_t roundoffset = 1 << 30 - log2N;
    return (a + roundoffset >> (31 - log2N));
  }

  // Public-functinoal key switching integrated with Sample Extract
  template <uint32_t lwe_n = 500, uint32_t tlwe_n = 1024,
            uint32_t decomp_bits = 2, uint32_t decomp_size = 8>
  __device__ inline void KeySwitch(Torus *lwe, Torus *tlwe, Torus *ksk)
  {
    static const Torus decomp_mask = (1u << decomp_bits) - 1;                  // OK
    static const Torus decomp_offset = 1u << (31 - decomp_size * decomp_bits); // OK
    uint32_t tid = ThisBlockRankInGrid() * ThisBlockSize() + ThisThreadRankInBlock();
    uint32_t bdim = ThisGridSize() * ThisBlockSize();
    uint32_t log2n = __log2f(Align512(lwe_n));
    uint32_t log2t = __log2f(decomp_size);
    Torus tmp, res, val;
    uint32_t m = 1u << decomp_bits;

#pragma unroll 0
    for (int i = tid; i <= lwe_n; i += bdim)
    {                                        // loop for n
      res = (i == lwe_n ? tlwe[tlwe_n] : 0); // i==lwe_n means b.
      for (int j = 0; j < tlwe_n; j++)
      {
        tmp = (j == 0 ? tlwe[0] : -tlwe[tlwe_n - j]); // anti-periodic index.
        tmp += decomp_offset;
        for (int k = 0; k < decomp_size; k++)
        { // loop for t
          val = (tmp >> (32 - (k + 1) * decomp_bits)) & decomp_mask;
          assert(val < m);
          if (val != 0)
          {
            uint32_t idx = (((j * decomp_size + k) << decomp_bits | val) << log2n) | i;
#ifdef DEBUG
            assert(idx < Align512(lwe_n) * decomp_size * m * tlwe_n);
#endif
            res -= ksk[idx];
          }
        }
      }
      lwe[i] = res;
    }
    __threadfence();
    __syncthreads();
  }

  template <uint32_t lwe_n = 500, uint32_t tlwe_n = 1024,
            uint32_t tgsw_decomp_bits = 10, uint32_t tgsw_decomp_size = 2>
  __device__ void Accumulate(Torus *tlwe,
                             FFP *sh_acc_ntt,
                             FFP *sh_res_ntt,
                             uint32_t a_bar,
                             FFP *tgsw_ntt,
                             CuNTTHandler<tlwe_n, NEGATIVE_CYCLIC_CONVOLUTION> ntt,
                             int state, int digit, int stage)
  {
    static const uint32_t decomp_bits = tgsw_decomp_bits;
    const uint32_t log2N = (uint32_t)__log2f(tlwe_n);
    static const uint32_t decomp_mask = (1 << decomp_bits) - 1; // ok
    // add half of decompision width and then bit-shift-right
    static const int32_t decomp_half = 1 << (decomp_bits - 1); // ok
    // offset for rounding
    static const uint32_t decomp_offset = (0x1u << (31 - decomp_bits));
    register uint32_t tid = ThisBlockRankInGrid() * ThisBlockSize() +
                            ThisThreadRankInBlock();
    uint32_t bdim = ThisGridSize() * ThisBlockSize();

    // temp[2] = sh_acc[2] * (x^exp - 1)
    // sh_acc_ntt[0, 1] = Decomp(temp[0])
    // sh_acc_ntt[2, 3] = Decomp(temp[1])
    // This algorithm is tested in cpp.
    switch (state)
    {
    case 1: // Rotate acc tlwe by a_bar Gudget decomposion
      Torus temp;
#pragma unroll
      for (int i = tid; i < tlwe_n; i += bdim)
      {
        uint32_t cmp = (uint32_t)(i < (a_bar & (tlwe_n - 1))); // i < a_bar % N
        uint32_t neg = (cmp ^ (a_bar >> log2N));               // xor(cmp, a_bar/N)
        uint32_t pos = -((1 - cmp) ^ (a_bar >> log2N));
#pragma unroll
        for (int j = 0; j < SMALL_K + 1; j++)
        {
          temp = tlwe[(j << log2N) | ((i - a_bar) & (tlwe_n - 1))]; // (j << log2N) for j-th poly
          temp = temp * (1 - neg) + (-temp) * neg;
          temp -= tlwe[(j << log2N) | i];
#ifdef DEBUG_ROTATE
          __syncthreads(); // must
          tlwe[(j << log2N) | i] = temp;
#else
          // decomp temp
          temp += decomp_offset;
#pragma unroll
          for (int l = 0; l < tgsw_decomp_size; l++)
          {
            sh_acc_ntt[(tgsw_decomp_size * j + l) * tlwe_n + i] =
                FFP(Torus(((temp >> (32 - (l + 1) * decomp_bits)) & decomp_mask) - decomp_half));
          }
#endif
        }
      }
      __threadfence();
      __syncthreads(); // must
      return;

    case 2: // (k+1)*l NTTs in parallel
#ifdef DEBUG_ROTATE
#else
            // FFP* tar = sh_acc_ntt;
      // FFP* tar = &sh_acc_ntt[digit*(SMALL_K+1) << LARGE_N];
      FFP *tar = &sh_acc_ntt[digit * (SMALL_K + 1) << log2N];
      ntt.template NTT_<FFP, N_BUTTERFLY>(tar, tar, stage, 0);
      __threadfence();
      __syncthreads();
#endif
      return;

    case 3: // Multiply with bootstrapping key in global memory.
#ifdef DEBUG_ROTATE
#else
#pragma unroll
      for (int i = tid; i < tlwe_n; i += bdim)
      {
#pragma unroll
        for (int j = 0; j < SMALL_K + 1; j++)
        { // for j-th TRLWE sample
          FFP *ptr = &sh_res_ntt[tgsw_decomp_size * (SMALL_K + 1) + j << log2N];
          ptr[i] = 0;
#pragma unroll
          for (int l = 0; l < tgsw_decomp_size; l++)
          {
            ptr[i] += sh_acc_ntt[(tgsw_decomp_size * j + l << log2N) + i] *
                      tgsw_ntt[(tgsw_decomp_size * j + l << log2N) + i];
          }
        }
      }
      __threadfence();
      __syncthreads(); // new
#endif
      return;

    case 4: // 2 NTTInvs
#ifdef DEBUG_ROTATE
#else
      FFP *src = &sh_res_ntt[(SMALL_K + 1) * tgsw_decomp_size << log2N];
      ntt.template NTTInv_<FFP, N_BUTTERFLY>(src, src, stage, 0);
      __threadfence();
      __syncthreads();
#endif
      return;

    case 5: // add 2 NTTInvs to acc
#ifdef DEBUG_ROTATE
#else
      uint32_t nthreads = gridDim.x * blockDim.x;
      assert(nthreads == tlwe_n);
      uint32_t idx = tid % nthreads; // local thread id
      FFP *ptr_res = &sh_res_ntt[((SMALL_K + 1) * tgsw_decomp_size << log2N) +
                                 (tid / nthreads) * nthreads];
      Torus *ptr_acc = &tlwe[(tid / nthreads) * nthreads];

      uint64_t med = FFP::kModulus() / 2;
      ptr_acc[idx] += Torus(ptr_res[idx].val() - (ptr_res[idx].val() >= med));
      __threadfence();
      __syncthreads();
#endif
      return;

    default:
      printf("Unknown state");
      assert(false);
    }
  }

#if 0
__global__
void __Bootstrap__(Torus* out, Torus* in, Torus mu,
                   FFP* bk,
                   Torus* ksk,
                   CuNTTHandler<> ntt) {
#else
  template <uint32_t length = 1024,
            ConvKind conv_kind = NEGATIVE_CYCLIC_CONVOLUTION>
  __global__ void __Bootstrap__(Torus *out, Torus *in, Torus mu,
                                FFP *bk,
                                Torus *ksk,
                                CuNTTHandler<length, conv_kind> ntt)
  {
#endif
//  Assert(bk.k() == 1);
//  Assert(bk.l() == 2);
//  Assert(bk.n() == 1024);
#if 0
  __shared__ FFP sh[6 * 1024];
#endif
    //  FFP* sh_acc_ntt[4] = { sh, sh + 1024, sh + 2048, sh + 3072 };
    //  FFP* sh_res_ntt[2] = { sh, sh + 4096 };
    Torus *tlwe = (Torus *)&sh[5120];

    // test vector
    // acc.a = 0; acc.b = vec(mu) * x ^ (in.b()/2048)
    register int32_t bar = 2048 - ModSwitch2048(in[500]);
    register uint32_t tid = ThisThreadRankInBlock();
    register uint32_t bdim = ThisBlockSize();
    register uint32_t cmp, neg, pos;
#pragma unroll
    for (int i = tid; i < 1024; i += bdim)
    {
      tlwe[i] = 0; // part a
      if (bar == 2048)
        tlwe[i + 1024] = mu;
      else
      {
        cmp = (uint32_t)(i < (bar & 1023));
        neg = -(cmp ^ (bar >> 10));
        pos = -((1 - cmp) ^ (bar >> 10));
        tlwe[i + 1024] = (mu & pos) + ((-mu) & neg); // part b
      }
    }
    __syncthreads();
// accumulate
#pragma unroll
    for (int i = 0; i < 500; i++)
    { // 500 iterations
      bar = ModSwitch2048(in[i]);
      Accumulate(tlwe, sh, sh, bar, bk + (i << 13), ntt, 0, 0, 0);
    }

    static const uint32_t lwe_n = 500;
    static const uint32_t tlwe_n = 1024;
    static const uint32_t ks_bits = 2;
    static const uint32_t ks_size = 8;
    KeySwitch<lwe_n, tlwe_n, ks_bits, ks_size>(out, tlwe, ksk);
  }

  template <uint32_t lwe_n = 500, uint32_t tlwe_n = 1024,
            uint32_t tgsw_decomp_bits = 10, uint32_t tgsw_decomp_size = 2,
            uint32_t decomp_bits = 2, uint32_t decomp_size = 8>
  __global__ void __NandBootstrap__(Torus *out, Torus *in0, Torus *in1, Torus mu, Torus fix,
                                    FFP *bk, Torus *ksk, CuNTTHandler<tlwe_n, NEGATIVE_CYCLIC_CONVOLUTION> ntt,
                                    int state, int polynum, int digit, int stage)
  {
    Torus *tlwe = (Torus *)&sh[((SMALL_K + 1) * tgsw_decomp_size + (SMALL_K + 1)) * tlwe_n];
    const uint32_t log2N = (uint32_t)__log2f(tlwe_n);
    const uint32_t log2l = (uint32_t)__log2f(tgsw_decomp_size);
    const uint32_t log2k1 = (uint32_t)__log2f(SMALL_K + 1);
    register uint32_t bar = 2 * tlwe_n -
                            ModSwitch_2N(fix - in0[lwe_n] - in1[lwe_n], log2N); // -b
#if 0                                                                           // assert settinsg for in0=1 and in1=1
  assert(in0[lwe_n] == fix && in1[lwe_n] == fix);
  assert(bar == 2*tlwe_n - ModSwitch_2N(-fix, log2N));
  assert(bar < 2*tlwe_n);
  assert(bar % 2*tlwe_n < tlwe_n);
#endif
    register uint32_t tid = ThisBlockRankInGrid() * ThisBlockSize() +
                            ThisThreadRankInBlock();
    register uint32_t bdim = ThisGridSize() * ThisBlockSize();
    register uint32_t cmp, neg, pos;

    switch (state)
    {

    case 0: // test vector rotated by b (bar)
#pragma unroll
      for (int i = tid; i < tlwe_n; i += bdim)
      {
        tlwe[i] = 0;
        // calculate source coefficient.
        cmp = (i < (bar & (tlwe_n - 1)));
        neg = (cmp ^ (bar >> log2N));
        tlwe[i + tlwe_n] = mu * (1 - neg) + (-mu) * neg;
#ifdef DEBUG
        assert((neg == 0 || neg == 1) &&
               (tlwe[i + tlwe_n] == mu || tlwe[i + tlwe_n] == -mu));
        printf("tv[%d]=%d\n", i, tlwe[i + tlwe_n]);
#endif
      }
      __threadfence();
      __syncthreads();
      return;
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
#ifdef DEBUG_ROTATE_A0
#else
    bar = ModSwitch_2N(0 - in0[polynum] - in1[polynum], log2N);
    Accumulate<lwe_n, tlwe_n, tgsw_decomp_bits, tgsw_decomp_size>(tlwe, sh, sh, bar,
                                                                  bk + tlwe_n * (SMALL_K + 1) * tgsw_decomp_size * polynum, ntt, state, digit, stage);
#endif
      return;
    case 6:
      KeySwitch<lwe_n, tlwe_n, decomp_bits, decomp_size>(out, tlwe, ksk);
      return;
    default:
      printf("Unknown state");
      assert(false);
    }
  }

#if 0
__global__
void __OrBootstrap__(Torus* out, Torus* in0, Torus* in1, Torus mu, Torus fix,
                       FFP* bk, Torus* ksk, CuNTTHandler<> ntt) {
#else
  template <uint32_t lwe_n = 500, uint32_t tlwe_n = 1024,
            uint32_t tgsw_decomp_bits = 10, uint32_t tgsw_decomp_size = 2,
            uint32_t decomp_bits = 2, uint32_t decomp_size = 8>
  __global__ void __OrBootstrap__(Torus *out, Torus *in0, Torus *in1, Torus mu, Torus fix,
                                  FFP *bk, Torus *ksk, CuNTTHandler<tlwe_n, NEGATIVE_CYCLIC_CONVOLUTION> ntt)
  {
#endif
    __shared__ FFP sh[6 * 1024];
    Torus *tlwe = (Torus *)&sh[5120];
    // test vector: acc.a = 0; acc.b = vec(mu) * x ^ (in.b()/2048)
    register int32_t bar = 2048 - ModSwitch2048(fix + in0[500] + in1[500]);
    register uint32_t tid = ThisThreadRankInBlock();
    register uint32_t bdim = ThisBlockSize();
    register uint32_t cmp, neg, pos;
#pragma unroll
    for (int i = tid; i < 1024; i += bdim)
    {
      tlwe[i] = 0; // part a
      if (bar == 2048)
        tlwe[i + 1024] = mu;
      else
      {
        cmp = (uint32_t)(i < (bar & 1023));
        neg = -(cmp ^ (bar >> 10));
        pos = -((1 - cmp) ^ (bar >> 10));
        tlwe[i + 1024] = (mu & pos) + ((-mu) & neg); // part b
      }
    }
    __threadfence();
    __syncthreads();
// accumulate
#pragma unroll
    for (int i = 0; i < 500; i++)
    { // 500 iterations
      bar = ModSwitch2048(0 + in0[i] + in1[i]);
      Accumulate(tlwe, sh, sh, bar, bk + (i << 13), ntt, 0, 0, 0);
    }
    KeySwitch<500, 1024, 2, 8>(out, tlwe, ksk);
  }

#if 0
__global__
void __AndBootstrap__(Torus* out, Torus* in0, Torus* in1, Torus mu, Torus fix,
                       FFP* bk, Torus* ksk, CuNTTHandler<> ntt) {
#else
  template <uint32_t lwe_n = 500, uint32_t tlwe_n = 1024,
            uint32_t tgsw_decomp_bits = 10, uint32_t tgsw_decomp_size = 2,
            uint32_t decomp_bits = 2, uint32_t decomp_size = 8>
  __global__ void __AndBootstrap__(Torus *out, Torus *in0, Torus *in1, Torus mu, Torus fix,
                                   FFP *bk, Torus *ksk, CuNTTHandler<tlwe_n, NEGATIVE_CYCLIC_CONVOLUTION> ntt)
  {
#endif
    __shared__ FFP sh[6 * 1024];
    Torus *tlwe = (Torus *)&sh[5120];
    // test vector: acc.a = 0; acc.b = vec(mu) * x ^ (in.b()/2048)
    register int32_t bar = 2048 - ModSwitch2048(fix + in0[500] + in1[500]);
    register uint32_t tid = ThisThreadRankInBlock();
    register uint32_t bdim = ThisBlockSize();
    register uint32_t cmp, neg, pos;
#pragma unroll
    for (int i = tid; i < 1024; i += bdim)
    {
      tlwe[i] = 0; // part a
      if (bar == 2048)
        tlwe[i + 1024] = mu;
      else
      {
        cmp = (uint32_t)(i < (bar & 1023));
        neg = -(cmp ^ (bar >> 10));
        pos = -((1 - cmp) ^ (bar >> 10));
        tlwe[i + 1024] = (mu & pos) + ((-mu) & neg); // part b
      }
    }
    __threadfence();
    __syncthreads();
// accumulate
#pragma unroll
    for (int i = 0; i < 500; i++)
    { // 500 iterations
      bar = ModSwitch2048(0 + in0[i] + in1[i]);
      Accumulate(tlwe, sh, sh, bar, bk + (i << 13), ntt, 0, 0, 0);
    }
    KeySwitch<500, 1024, 2, 8>(out, tlwe, ksk);
  }

#if 0
__global__
void __NorBootstrap__(Torus* out, Torus* in0, Torus* in1, Torus mu, Torus fix,
                       FFP* bk, Torus* ksk, CuNTTHandler<> ntt) {
#else
  template <uint32_t lwe_n = 500, uint32_t tlwe_n = 1024,
            uint32_t tgsw_decomp_bits = 10, uint32_t tgsw_decomp_size = 2,
            uint32_t decomp_bits = 2, uint32_t decomp_size = 8>
  __global__ void __NorBootstrap__(Torus *out, Torus *in0, Torus *in1, Torus mu, Torus fix,
                                   FFP *bk, Torus *ksk, CuNTTHandler<tlwe_n, NEGATIVE_CYCLIC_CONVOLUTION> ntt)
  {
#endif
    __shared__ FFP sh[6 * 1024];
    Torus *tlwe = (Torus *)&sh[5120];
    // test vector: acc.a = 0; acc.b = vec(mu) * x ^ (in.b()/2048)
    register int32_t bar = 2048 - ModSwitch2048(fix - in0[500] - in1[500]);
    register uint32_t tid = ThisThreadRankInBlock();
    register uint32_t bdim = ThisBlockSize();
    register uint32_t cmp, neg, pos;
#pragma unroll
    for (int i = tid; i < 1024; i += bdim)
    {
      tlwe[i] = 0; // part a
      if (bar == 2048)
        tlwe[i + 1024] = mu;
      else
      {
        cmp = (uint32_t)(i < (bar & 1023));
        neg = -(cmp ^ (bar >> 10));
        pos = -((1 - cmp) ^ (bar >> 10));
        tlwe[i + 1024] = (mu & pos) + ((-mu) & neg); // part b
      }
    }
    __threadfence();
    __syncthreads();
// accumulate
#pragma unroll
    for (int i = 0; i < 500; i++)
    { // 500 iterations
      bar = ModSwitch2048(0 - in0[i] - in1[i]);
      Accumulate(tlwe, sh, sh, bar, bk + (i << 13), ntt, 0, 0, 0);
    }
    KeySwitch<500, 1024, 2, 8>(out, tlwe, ksk);
  }

#if 0
__global__
void __XorBootstrap__(Torus* out, Torus* in0, Torus* in1, Torus mu, Torus fix,
                       FFP* bk, Torus* ksk, CuNTTHandler<> ntt) {
#else
  template <uint32_t lwe_n = 500, uint32_t tlwe_n = 1024,
            uint32_t tgsw_decomp_bits = 10, uint32_t tgsw_decomp_size = 2,
            uint32_t decomp_bits = 2, uint32_t decomp_size = 8>
  __global__ void __XorBootstrap__(Torus *out, Torus *in0, Torus *in1, Torus mu, Torus fix,
                                   FFP *bk, Torus *ksk, CuNTTHandler<tlwe_n, NEGATIVE_CYCLIC_CONVOLUTION> ntt)
  {
#endif
    __shared__ FFP sh[6 * 1024];
    Torus *tlwe = (Torus *)&sh[5120];
    // test vector: acc.a = 0; acc.b = vec(mu) * x ^ (in.b()/2048)
    register int32_t bar = 2048 - ModSwitch2048(fix + 2 * in0[500] + 2 * in1[500]);
    register uint32_t tid = ThisThreadRankInBlock();
    register uint32_t bdim = ThisBlockSize();
    register uint32_t cmp, neg, pos;
#pragma unroll
    for (int i = tid; i < 1024; i += bdim)
    {
      tlwe[i] = 0; // part a
      if (bar == 2048)
        tlwe[i + 1024] = mu;
      else
      {
        cmp = (uint32_t)(i < (bar & 1023));
        neg = -(cmp ^ (bar >> 10));
        pos = -((1 - cmp) ^ (bar >> 10));
        tlwe[i + 1024] = (mu & pos) + ((-mu) & neg); // part b
      }
    }
    __threadfence();
    __syncthreads();
// accumulate
#pragma unroll
    for (int i = 0; i < 500; i++)
    { // 500 iterations
      bar = ModSwitch2048(0 + 2 * in0[i] + 2 * in1[i]);
      Accumulate(tlwe, sh, sh, bar, bk + (i << 13), ntt, 0, 0, 0);
    }
    KeySwitch<500, 1024, 2, 8>(out, tlwe, ksk);
  }

#if 0
__global__
void __XnorBootstrap__(Torus* out, Torus* in0, Torus* in1, Torus mu, Torus fix,
                       FFP* bk, Torus* ksk, CuNTTHandler<> ntt) {
#else
  template <uint32_t lwe_n = 500, uint32_t tlwe_n = 1024,
            uint32_t tgsw_decomp_bits = 10, uint32_t tgsw_decomp_size = 2,
            uint32_t decomp_bits = 2, uint32_t decomp_size = 8>
  __global__ void __XnorBootstrap__(Torus *out, Torus *in0, Torus *in1, Torus mu, Torus fix,
                                    FFP *bk, Torus *ksk, CuNTTHandler<tlwe_n, NEGATIVE_CYCLIC_CONVOLUTION> ntt)
  {
#endif
    __shared__ FFP sh[6 * 1024];
    Torus *tlwe = (Torus *)&sh[5120];
    // test vector: acc.a = 0; acc.b = vec(mu) * x ^ (in.b()/2048)
    register int32_t bar = 2048 - ModSwitch2048(fix - 2 * in0[500] - 2 * in1[500]);
    register uint32_t tid = ThisThreadRankInBlock();
    register uint32_t bdim = ThisBlockSize();
    register uint32_t cmp, neg, pos;
#pragma unroll
    for (int i = tid; i < 1024; i += bdim)
    {
      tlwe[i] = 0; // part a
      if (bar == 2048)
        tlwe[i + 1024] = mu;
      else
      {
        cmp = (uint32_t)(i < (bar & 1023));
        neg = -(cmp ^ (bar >> 10));
        pos = -((1 - cmp) ^ (bar >> 10));
        tlwe[i + 1024] = (mu & pos) + ((-mu) & neg); // part b
      }
    }
    __threadfence();
    __syncthreads();
// accumulate
#pragma unroll
    for (int i = 0; i < 500; i++)
    { // 500 iterations
      bar = ModSwitch2048(0 - 2 * in0[i] - 2 * in1[i]);
      Accumulate(tlwe, sh, sh, bar, bk + (i << 13), ntt, 0, 0, 0);
    }
    KeySwitch<500, 1024, 2, 8>(out, tlwe, ksk);
  }

  void Bootstrap(LWESample *out,
                 LWESample *in,
                 Torus mu,
                 cudaStream_t st)
  {
    dim3 grid(1);
    dim3 block(512);
    __Bootstrap__<<<grid, block, 0, st>>>(out->data(), in->data(), mu,
                                          bk_ntt->data(), ksk_dev->data(), *ntt_handler);
    CuCheckError();
  }

  // Added state and phase parameters.
  void NandBootstrap(LWESample *out, LWESample *in0, LWESample *in1,
                     Torus mu, Torus fix, cudaStream_t st)
  {
#if N_BUTTERFLY == 1 && LARGE_N > 1024
    dim3 grid_ntt(NGRID_X, SMALL_K + 1); // grid_ntt(NGRID_X, (SMALL_K + 1) * SMALL_L);
#else
  dim3 grid_ntt(NGRID_X, (SMALL_K + 1) * SMALL_L);
#endif
    dim3 grid_intt(NGRID_X, 2);
    dim3 block_ntt(NBLOCK);
    dim3 block_intt(NBLOCK);
#ifdef DEBUG
    printf("N=%d, n=%d, k=%d, l=%d, NGRID_X=%d, NBLOCK=%d, N_BUTTERFLY=%d\n",
           LARGE_N, SMALL_N, SMALL_K, SMALL_L, NGRID_X, NBLOCK, N_BUTTERFLY);
#endif
#undef NBLOCK
#undef NGRID_X
#define NBLOCK (LARGE_N < MAX_THREAD_PER_BLOCK ? LARGE_N : MAX_THREAD_PER_BLOCK)
#define NGRID_X (LARGE_N % MAX_THREAD_PER_BLOCK == 0 ? LARGE_N / MAX_THREAD_PER_BLOCK : LARGE_N / MAX_THREAD_PER_BLOCK + 1)
    dim3 grid_acc(NGRID_X, 2);
    dim3 block_acc(NBLOCK);
    dim3 grid_ks(NGRID_X);
    dim3 block_ks(NBLOCK);
    int state;
    // std::cout << grid_ntt.x << " " << grid_ntt.y << " " << block_ntt.x << std::endl;
    // std::cout << state << endl;

    // Blind Rotate - calculate ACC*X^(-b)
    __NandBootstrap__<SMALL_N, LARGE_N, B_G_BIT, SMALL_L, B_KS_BIT, SMALL_T>
        <<<grid_ntt, block_ntt, 0, st>>>(out->data(), in0->data(), in1->data(), mu, fix,
                                         bk_ntt->data(), ksk_dev->data(), *ntt_handler, state = 0, 0, 0, 0);
    if (grid_ntt.x > 1)
      cudaDeviceSynchronize();
    CuCheckError();
    for (int i = 0; i < SMALL_N; i++)
    { // inside Accumulate
      // Blind Rotate - calculate ACC*X^a_i - ACC
      __NandBootstrap__<SMALL_N, LARGE_N, B_G_BIT, SMALL_L, B_KS_BIT, SMALL_T>
          <<<grid_ntt, block_ntt, 0, st>>>(out->data(), in0->data(), in1->data(), mu, fix,
                                           bk_ntt->data(), ksk_dev->data(), *ntt_handler, state = 1, i, 0, 0);
      if (grid_ntt.x > 1)
        cudaDeviceSynchronize();
      CuCheckError();
      // CMux - NTT
      int gadget_decomp_loop = (grid_ntt.x > 1 ? SMALL_L : 1); // 1;
      for (int l = 0; l < gadget_decomp_loop; l++)
      {
        for (int j = 0; j < log2f(LARGE_N); j++)
        {
          __NandBootstrap__<SMALL_N, LARGE_N, B_G_BIT, SMALL_L, B_KS_BIT, SMALL_T>
              <<<grid_ntt, block_ntt, 0, st>>>(out->data(), in0->data(), in1->data(), mu, fix,
                                               bk_ntt->data(), ksk_dev->data(), *ntt_handler, state = 2, i, l, j);
          if (grid_ntt.x > 1)
            cudaDeviceSynchronize();
          CuCheckError();
        }
      }
      // CMux - Multiply ACC with BK
      __NandBootstrap__<SMALL_N, LARGE_N, B_G_BIT, SMALL_L, B_KS_BIT, SMALL_T>
          <<<grid_ntt, block_ntt, 0, st>>>(out->data(), in0->data(), in1->data(), mu, fix,
                                           bk_ntt->data(), ksk_dev->data(), *ntt_handler, state = 3, i, 0, 0);
      if (grid_ntt.x > 1)
        cudaDeviceSynchronize();
      CuCheckError();
      // CMux - INTT
      for (int j = 0; j < log2f(LARGE_N); j++)
      {
        __NandBootstrap__<SMALL_N, LARGE_N, B_G_BIT, SMALL_L, B_KS_BIT, SMALL_T>
            <<<grid_intt, block_intt, 0, st>>>(out->data(), in0->data(), in1->data(), mu, fix,
                                               bk_ntt->data(), ksk_dev->data(), *ntt_handler, state = 4, i, 0, j);
        if (grid_intt.x > 1)
          cudaDeviceSynchronize();
        CuCheckError();
      }
      // CMux - Add INTT results to ACC
      cudaDeviceSynchronize();
      __NandBootstrap__<SMALL_N, LARGE_N, B_G_BIT, SMALL_L, B_KS_BIT, SMALL_T>
          <<<grid_ks, block_ks, 0, st>>>(out->data(), in0->data(), in1->data(), mu, fix,
                                         bk_ntt->data(), ksk_dev->data(), *ntt_handler, state = 5, i, 0, 0);
      if (grid_intt.x > 1)
        cudaDeviceSynchronize();
      CuCheckError();
    }
    // Sample Extract and Key Switch
    __NandBootstrap__<SMALL_N, LARGE_N, B_G_BIT, SMALL_L, B_KS_BIT, SMALL_T>
        <<<grid_ks, block_ks, 0, st>>>(out->data(), in0->data(), in1->data(), mu, fix,
                                       bk_ntt->data(), ksk_dev->data(), *ntt_handler, state = 6, 0, 0, 0);
    cudaDeviceSynchronize();
    CuCheckError();
  }

  void OrBootstrap(LWESample *out, LWESample *in0, LWESample *in1,
                   Torus mu, Torus fix, cudaStream_t st)
  {
#if 0
  __OrBootstrap__<<<1, 512, 0, st>>>(out->data(), in0->data(),
      in1->data(), mu, fix, bk_ntt->data(), ksk_dev->data(), *ntt_handler);
#else
#ifdef N16384
    __OrBootstrap__<636, 16384, 6, 5, 2, 7>
        <<<40, 512, (2 * 5 + 2) * 16384 * sizeof(FFP), st>>>(out->data(), in0->data(),
                                                             in1->data(), mu, fix, bk_ntt->data(), ksk_dev->data(), *ntt_handler);
#else
    __OrBootstrap__<500, 1024, 10, 2, 2, 8>
        <<<1, 512, (2 * 2 + 2) * 1024 * sizeof(FFP), st>>>(out->data(), in0->data(),
                                                           in1->data(), mu, fix, bk_ntt->data(), ksk_dev->data(), *ntt_handler);
#endif
#endif
    CuCheckError();
  }

  void AndBootstrap(LWESample *out, LWESample *in0, LWESample *in1,
                    Torus mu, Torus fix, cudaStream_t st)
  {
#if 0
  __AndBootstrap__<<<1, 512, 0, st>>>(out->data(), in0->data(),
      in1->data(), mu, fix, bk_ntt->data(), ksk_dev->data(), *ntt_handler);
#else
#ifdef N16384
    __AndBootstrap__<636, 16384, 6, 5, 2, 7>
        <<<40, 512, (2 * 5 + 2) * 16384 * sizeof(FFP), st>>>(out->data(), in0->data(),
                                                             in1->data(), mu, fix, bk_ntt->data(), ksk_dev->data(), *ntt_handler);
#else
    __AndBootstrap__<500, 1024, 10, 2, 2, 8>
        <<<1, 512, (2 * 2 + 2) * 1024 * sizeof(FFP), st>>>(out->data(), in0->data(),
                                                           in1->data(), mu, fix, bk_ntt->data(), ksk_dev->data(), *ntt_handler);
#endif
#endif
    CuCheckError();
  }

  void NorBootstrap(LWESample *out, LWESample *in0, LWESample *in1,
                    Torus mu, Torus fix, cudaStream_t st)
  {
#if 0
  __NorBootstrap__<<<1, 512, 0, st>>>(out->data(), in0->data(),
      in1->data(), mu, fix, bk_ntt->data(), ksk_dev->data(), *ntt_handler);
#else
#ifdef N16384
    __NorBootstrap__<636, 16384, 6, 5, 2, 7>
        <<<40, 512, (2 * 5 + 2) * 16384 * sizeof(FFP), st>>>(out->data(), in0->data(),
                                                             in1->data(), mu, fix, bk_ntt->data(), ksk_dev->data(), *ntt_handler);
#else
    __NorBootstrap__<500, 1024, 10, 2, 2, 8>
        <<<1, 512, (2 * 2 + 2) * 1024 * sizeof(FFP), st>>>(out->data(), in0->data(),
                                                           in1->data(), mu, fix, bk_ntt->data(), ksk_dev->data(), *ntt_handler);
#endif
#endif
    CuCheckError();
  }

  void XorBootstrap(LWESample *out, LWESample *in0, LWESample *in1,
                    Torus mu, Torus fix, cudaStream_t st)
  {
#if 0
  __XorBootstrap__<<<1, 512, 0, st>>>(out->data(), in0->data(),
      in1->data(), mu, fix, bk_ntt->data(), ksk_dev->data(), *ntt_handler);
#else
#ifdef N16384
    __XorBootstrap__<636, 16384, 6, 5, 2, 7>
        <<<40, 512, (2 * 5 + 2) * 16384 * sizeof(FFP), st>>>(out->data(), in0->data(),
                                                             in1->data(), mu, fix, bk_ntt->data(), ksk_dev->data(), *ntt_handler);
#else
    __XorBootstrap__<500, 1024, 10, 2, 2, 8>
        <<<1, 512, (2 * 2 + 2) * 1024 * sizeof(FFP), st>>>(out->data(), in0->data(),
                                                           in1->data(), mu, fix, bk_ntt->data(), ksk_dev->data(), *ntt_handler);
#endif
#endif
    CuCheckError();
  }

  void XnorBootstrap(LWESample *out, LWESample *in0, LWESample *in1,
                     Torus mu, Torus fix, cudaStream_t st)
  {
#if 0
  __XnorBootstrap__<<<1, 512, 0, st>>>(out->data(), in0->data(),
      in1->data(), mu, fix, bk_ntt->data(), ksk_dev->data(), *ntt_handler);
#else
#ifdef N16384
    __XnorBootstrap__<636, 16384, 6, 5, 2, 7>
        <<<40, 512, (2 * 5 + 2) * 16384 * sizeof(FFP), st>>>(out->data(), in0->data(),
                                                             in1->data(), mu, fix, bk_ntt->data(), ksk_dev->data(), *ntt_handler);
#else
    __XnorBootstrap__<500, 1024, 10, 2, 2, 8>
        <<<1, 512, (2 * 2 + 2) * 1024 * sizeof(FFP), st>>>(out->data(), in0->data(),
                                                           in1->data(), mu, fix, bk_ntt->data(), ksk_dev->data(), *ntt_handler);
#endif
#endif
    CuCheckError();
  }

} // namespace cufhe
