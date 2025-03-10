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

#pragma once

#include "ntt_ffp.cuh"
#include "ntt_single_thread.cuh"
#include "ntt_shifting.cuh"
#include <include/details/utils_gpu.cuh>
#if 1
#include <include/cufhe.h>
#undef DEBUG
#endif

namespace cufhe {

// Large-sized twiddles cannot be placed in shared memory in L1.
#if 1
__device__ FFP con_twd[LARGE_N];
__device__ FFP con_twd_inv[LARGE_N];
__device__ FFP con_twd_sqrt[LARGE_N];
__device__ FFP con_twd_sqrt_inv[LARGE_N];
#else
__constant__ FFP con_twd[1024];
__constant__ FFP con_twd_inv[1024];
__constant__ FFP con_twd_sqrt[1024];
__constant__ FFP con_twd_sqrt_inv[1024];
#endif

__device__ inline
void NTT1024Core(FFP* r,
                 FFP* s,
                 const FFP* twd,
                 const FFP* twd_sqrt,
                 const uint32_t& t1d,
                 const uint3& t3d) {
  FFP *ptr = nullptr;
  FFP *ptr_twd = nullptr;
  FFP *ptr_twd_sqrt = nullptr;
  FFP t[4]; // twiddle factor
  // split into even odd 
  ptr = &s[t1d<<2];
  ptr_twd_sqrt = &con_twd_sqrt[t1d<<2];
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    r[i] *= ptr_twd_sqrt[((i >> 2) << 9) + (i&0x3)];  // (i div 4)*512+i
  // NTT1024
  #pragma unroll
  for (int i = 0; i < 4; i ++)
    t[i] = con_twd[(t1d<<2) + i];
  NTT8(r,t);
  //NTT8x2Lsh(r, t3d.z); // if (t1d >= 64) NTT8x2<1>(r);
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    ptr[((i >> 2) << 9) + (i&0x3)] = r[i];
  __threadfence();
  __syncthreads();

  // choose one of the first 64 entries in a block of 512 entries
  ptr = &s[t3d.z*512 + (t3d.y<<3) + t3d.x];
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    r[i] = ptr[i << 6];
  #pragma unroll
  for (int i = 0; i < 4; i ++)
    t[i] = con_twd[((t3d.y<<3) + t3d.x) << 3];
  NTT2(r[0], r[1], t[0]);
  NTT2(r[2], r[3], t[1]);
  NTT2(r[4], r[6], t[2]);
  NTT2(r[6], r[7], t[3]);
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    ptr[i << 6] = r[i];
  __threadfence();
  __syncthreads();

  #pragma unroll
  // choose one of the first 8 entries in a block of 64 entries
  // there are 16 such a enties.
  ptr = &s[t3d.z*512 + (t3d.y<<6) + (t3d.x<<2)];
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    r[i] = ptr[((i >> 2) << 5) + (i&0x3)];  // i/4*32 + i%4
  #pragma unroll
  for (int i = 0; i < 4; i ++)
    t[i] = con_twd[((t3d.x<<2) + i) << 4];
  NTT8(r, t);
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    ptr[((i >> 2) << 5) + (i&0x3)] = r[i];
  __threadfence();
  __syncthreads();

  ptr = &s[t1d << 3];
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    r[i] = ptr[i];
  #pragma unroll
  for (int i = 0; i < 4; i ++)
    t[i] = con_twd[i << 7];
  //NTT8x8Lsh(r, t1d >> 4); // less divergence if put here!
  NTT8(r, t);
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    ptr[i] = r[i];
  __threadfence();
  __syncthreads();
  ptr = &s[t1d<<2];
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    r[i] = ptr[((i >> 2) << 9) + (i&0x3)];
}

__device__ inline
void NTTInv1024Core(FFP* r,
                    FFP* s,
                    const FFP* twd_inv,
                    const FFP* twd_sqrt_inv,
                    const uint32_t& t1d,
                    const uint3& t3d) {
  FFP *ptr = nullptr;
  FFP t[4]; // twiddle factor
  ptr = &s[t1d << 3];
  for (int i = 0; i < 8; i ++)
    r[i] = ptr[i];
  for (int i = 0; i < 4; i ++)
    t[i] = con_twd_inv[i << 7];
  NTTInv8(r, t);
  NTTInv8x2Lsh(r, t3d.z); // if (t1d >= 64) NTT8x2<1>(r);
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    ptr[i] = r[i];
  __threadfence();
  __syncthreads();

  ptr = &s[t3d.z*512 + (t3d.y<<64) + (t3d.x<<2)];
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    r[i] = ptr[((i >> 2) << 5) + (i&0x3)];  // i/4*32 + i%4
  #pragma unroll
  for (int i = 0; i < 4; i ++)
    t[i] = con_twd[((t3d.x<<2) + i) << 4];
  NTTInv8(r, t);
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    ptr[((i >> 2) << 5) + (i&0x3)] = r[i];
  __threadfence();
  __syncthreads();

  // choose one of the first 64 entries in a block of 512 entries
  ptr = &s[t3d.z*512 + (t3d.y<<3) + t3d.x];
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    r[i] = ptr[i << 6];
  #pragma unroll
  for (int i = 0; i < 4; i ++)
    t[i] = con_twd[((t3d.y<<3) + t3d.x) << 3];
  NTTInv2(r[0], r[1], t[0]);
  NTTInv2(r[2], r[3], t[1]);
  NTTInv2(r[4], r[6], t[2]);
  NTTInv2(r[6], r[7], t[3]);
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    ptr[i << 6] = r[i];
  __threadfence();
  __syncthreads();

  ptr = &s[t1d*4];
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    r[i] = ptr[((i >> 2) << 9) + (i&0x3)]; // mult twiddle sqrt
  #pragma unroll
  for (int i = 0; i < 4; i ++)
    t[i] = con_twd[(t1d<<2) + i];
  NTTInv8(r, t);
  NTT8x2Lsh(r, t3d.z); // if (t1d >= 64) NTT8x2<1>(r);
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    r[i] *= con_twd_sqrt_inv[((i >> 2) << 9) + (i&0x3)]; // mult twiddle sqrt
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    ptr[((i >> 2) << 9) + (i&0x3)] = r[i];
  __threadfence();
  __syncthreads();
  ptr = &s[t1d << 3]; 
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    r[i] = ptr[i];
}

template <typename T>
__device__
void NTT1024(FFP* out,
             T* in,
             FFP* temp_shared,
             FFP* twd,
             FFP* twd_sqrt,
             uint32_t leading_thread) {
  uint32_t t1d = ThisThreadRankInBlock() - leading_thread;
  uint3 t3d;
  Index3DFrom1D<8, 8, 2>(t3d, t1d);
  register FFP r[8];
  #pragma unroll
  for (int i = 0; i < 8; i ++) {
    r[i] = FFP((T)in[(t1d<<2) + ((i >> 2) << 9) + (i&0x3)]);
  }
  __syncthreads();
  NTT1024Core(r, temp_shared, twd, twd_sqrt, t1d, t3d);
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    out[(t1d<<2) + ((i >> 2) << 9) + (i&0x3)] = r[i];
  __threadfence();
  __syncthreads();
}

// INTT2 using Cookey-Tukey Butterfly
__device__
void NTTInv2_(FFP* out, FFP* in, uint32_t length, uint32_t stage, 
		FFP* twd_inv, FFP* twd_sqrt_inv) {
  uint32_t tid = ThisBlockRankInGrid()*ThisBlockSize() + ThisThreadRankInBlock();
  // uint32_t total_threads = ThisGridSize()*ThisBlockSize();
  uint32_t nthreads = gridDim.x * blockDim.x;
  assert(nthreads == length/2);
  uint32_t ltid = tid % nthreads;  // local thread id
  uint32_t log2N = __log2f(length);
  uint32_t distance = length>>(log2N-stage); // Butterfly distance
  //uint32_t idx = (ltid/distance)*(distance<<1) + (ltid % distance);
  uint32_t idx = (tid/nthreads)*length + 
	  (ltid/distance)*(distance<<1) + (ltid % distance);
  uint32_t twd_idx = (ltid % distance) << (log2N-stage-1);
  register FFP r[2];
  FFP* ptr = &in[(tid/nthreads) * nthreads];
  FFP* ptr_out = &out[(tid/nthreads) * nthreads];
  #pragma unroll
  for (int i=0; i<2; i++) 
    r[i] = ptr[idx+i*distance];
  NTTInv2(r[0], r[1], twd_inv[twd_idx]);
  #pragma unroll
  if (stage == log2N-1) {
    #pragma unroll
    for (int i=0; i<2; i++) 
      r[i] *= twd_sqrt_inv[idx+i*distance];
  }
  #pragma unroll
  for (int i=0; i<2; i++) 
    ptr_out[idx + i*distance] = r[i];
  __syncthreads();
  __threadfence();
}

// NTT2 using Gentleman-Sande Butterfly, with 2X batterflies in one thread.
template <typename T, uint32_t N_B>
__device__
void NTT2X_(FFP* out, T* in, uint32_t length, uint32_t stage,
		FFP* twd, FFP* twd_sqrt, int test=0) {
  if (test == 2) return;
  uint32_t tid = ThisBlockRankInGrid()*ThisBlockSize() + ThisThreadRankInBlock();
  uint32_t bdim = ThisGridSize()*ThisBlockSize();
  uint32_t nthreads = gridDim.x * blockDim.x;
  assert(nthreads == length/(2*N_B));
  uint32_t ltid = tid % nthreads;  // local thread id
  uint32_t log2N = __log2f(length);
  uint32_t distance = length>>stage+1; // Butterfly distance
  FFP r[2*N_B];
  T* ptr = &in[(tid/nthreads) * nthreads * N_B];
  FFP* ptr_out = &out[(tid/nthreads) * nthreads * N_B];
  //printf("tid = %d\n", tid);
  for (int i=0; i<N_B; i++) {
    uint32_t idx = (tid/nthreads) * length + 
	    ((ltid*N_B+i)/distance)*(distance<<1) + ((ltid*N_B+i) % distance);
    // idx: index of the 1st input of butterfly for this thread
    uint32_t twd_idx = ((ltid*N_B+i) % distance) << stage;
    #pragma unroll
    for (int j=0; j<2; j++)
      r[2*i+j] = FFP(ptr[idx+j*distance]);
    if (stage == 0 && test == 0) {
      #pragma unroll
      for (int j=0; j<2; j++)
         r[2*i+j] *= twd_sqrt[idx+j*distance];
    }
    if (test == 0)
    NTT2(r[2*i], r[2*i+1], twd[twd_idx]);
    #pragma unroll
    for (int j=0; j<2; j++)
      ptr_out[idx + j*distance] = r[2*i+j];
  };
  __threadfence();
  __syncthreads();
}

// INTT2 using Cookey-Tukey Butterfly, with 2X batterflies in one thread.
template <typename T, uint32_t N_B>
__device__
void NTTInv2X_(T* out, FFP* in, uint32_t length, uint32_t stage,
		FFP* twd_inv, FFP* twd_sqrt_inv, int test=0) {
  if (test == 2) return;
  uint32_t tid = ThisBlockRankInGrid()*ThisBlockSize() + ThisThreadRankInBlock();
  // uint32_t total_threads = ThisGridSize()*ThisBlockSize();
  uint32_t nthreads = gridDim.x * blockDim.x;
  assert(nthreads == length/(2*N_B));
  uint32_t ltid = tid % nthreads;  // local thread id
  uint32_t log2N = __log2f(length);
  uint32_t distance = length>>(log2N-stage); // Butterfly distance

  FFP r[2*N_B];
  FFP* ptr = &in[(tid/nthreads) * nthreads * N_B];
  T* ptr_out = &out[(tid/nthreads) * nthreads * N_B];
  #pragma unroll
  for (int i=0; i<N_B; i++) {
    uint32_t idx = (tid/nthreads) * length + 
	    ((ltid*N_B+i)/distance)*(distance<<1) + ((ltid*N_B+i) % distance);
    // idx: index of the 1st input of butterfly for this thread
    uint32_t twd_idx = ((ltid*N_B+i) % distance) << (log2N-stage-1);
    #pragma unroll
    for (int j=0; j<2; j++)
      r[2*i+j] = ptr[idx+j*distance];
    if (test == 0) 
    NTTInv2(r[2*i], r[2*i+1], twd_inv[twd_idx]);
    if (stage == log2N-1 && test == 0) {
      #pragma unroll
      for (int j=0; j<2; j++)
         r[2*i+j] *= twd_sqrt_inv[idx+j*distance];
    }
    #pragma unroll
    for (int j=0; j<2; j++)
      ptr_out[idx + j*distance] = T(r[2*i+j].val());
  }
  __threadfence();
  __syncthreads();
}

template <typename T>
__device__
void NTT1024Decomp(FFP* out,
                   T* in,
                   FFP* temp_shared,
                   FFP* twd,
                   FFP* twd_sqrt,
                   uint32_t rsh_bits,
                   T mask,
                   T offset,
                   uint32_t leading_thread) {
  uint32_t t1d = ThisThreadRankInBlock() - leading_thread;
  uint3 t3d;
  Index3DFrom1D<8, 8, 2>(t3d, t1d);
  register FFP r[8];
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    r[i] = FFP(((in[(i << 7) | t1d] >> rsh_bits) & mask) - offset);
  __syncthreads();
  NTT1024Core(r, temp_shared, twd, twd_sqrt, t1d, t3d);
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    out[(i << 7) | t1d] = r[i];
  __threadfence();
  __syncthreads();
}

template <typename T>
__device__
void NTTInv1024(T* out,
                FFP* in,
                FFP* temp_shared,
                FFP* twd_inv,
                FFP* twd_sqrt_inv,
                uint32_t leading_thread) {
  uint32_t t1d = ThisThreadRankInBlock() - leading_thread;
  uint3 t3d;
  Index3DFrom1D<8, 8, 2>(t3d, t1d);
  register FFP r[8];
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    r[i] = in[(t1d << 3) + i];
  __syncthreads();
  NTTInv1024Core(r, temp_shared, twd_inv, twd_sqrt_inv, t1d, t3d);
  // mod 2^32 specifically
  uint64_t med = FFP::kModulus() / 2;
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    out[(t1d << 3) + i] = r[i];
  __threadfence();
  __syncthreads();
}

template <typename T>
__device__
void NTTInv1024Add(T* out,
                   FFP* in,
                   FFP* temp_shared,
                   FFP* twd_inv,
                   FFP* twd_sqrt_inv,
                   uint32_t leading_thread) {
  uint32_t t1d = ThisThreadRankInBlock() - leading_thread;
  uint3 t3d;
  Index3DFrom1D<8, 8, 2>(t3d, t1d);
  register FFP r[8];
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    r[i] = in[(t1d << 3) + i];
  __syncthreads();
  NTTInv1024Core(r, temp_shared, twd_inv, twd_sqrt_inv, t1d, t3d);
  // mod 2^32 specifically
  uint64_t med = FFP::kModulus() / 2;
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    out[(t1d << 3) + i] += T(r[i].val() - (r[i].val() >= med));
  __threadfence();
  __syncthreads();
}

__device__ inline
void NTT16384Core(FFP* r,
                 FFP* s,
                 const FFP* twd,
                 const FFP* twd_sqrt,
                 const uint32_t& t1d
                 ) {
  FFP *ptr = nullptr;
  FFP *ptr_twd = nullptr;
  FFP *ptr_twd_sqrt = nullptr;
  FFP t[4]; // twiddle factor
  ptr = &s[t1d<<2];
  ptr_twd_sqrt = &con_twd_sqrt[t1d<<2];
  uint32_t logN = 14;
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    r[i] *= ptr_twd_sqrt[((i >> 2) << (logN-1)) + (i&0x3)];
  // NTT16384
  #pragma unroll
  for (int i = 0; i < 4; i ++)
    t[i] = con_twd[(t1d<<2) + i];
  NTT8(r,t);
  //NTT8x2Lsh(r, t3d.z); // if (t1d >= 64) NTT8x2<1>(r);
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    ptr[((i >> 2) << (logN -1)) + (i&0x3)] = r[i];
  __threadfence();
  __syncthreads();

  // choose one of the first 1024 entries in a block of 8192 entries
  ptr = &s[(t1d%2)*8192 + (t1d/2)];
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    r[i] = ptr[i << 10];
  #pragma unroll
  for (int i = 0; i < 4; i ++)
    t[i] = con_twd[t1d>>1];
  NTT2(r[0], r[1], t[0]);
  NTT2(r[2], r[3], t[1]);
  NTT2(r[4], r[6], t[2]);
  NTT2(r[6], r[7], t[3]);
  #pragma unroll
  for (int i = 0; i < 8; i ++)
  ptr[i << 10] = r[i];
  __threadfence();
  __syncthreads();

  // NTT1024
  #pragma unroll
  ptr = &s[1024*(t1d / 128)];   //partition into 16 1024-blocks
  for (int i = 0; i < 8; i ++)
    r[i] = ptr[((i >> 2) << 9) + (i&0x3)];
  uint32_t idx = t1d % 128;
  uint3 t3d;
  Index3DFrom1D<8, 8, 2>(t3d, idx);
  NTT1024Core(r, ptr, twd, twd_sqrt, idx, t3d);
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    r[i] = ptr[((i >> 2) << (logN - 1)) + (i&0x3)];
}

__device__ inline
void NTTInv16384Core(FFP* r,
                    FFP* s,
                    const FFP* twd_inv,
                    const FFP* twd_sqrt_inv,
                    const uint32_t& t1d
                    ) {
  FFP *ptr = nullptr;
  FFP t[4]; // twiddle factor
  uint32_t logN = 14;
  // NTTInv1024Core 
  ptr = &s[1024*(t1d / 128)];   //partition into 16 1024-blocks
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    r[i] = ptr[((i >> 2) << 9) + (i&0x3)];
  //for (int i = 0; i < 8; i ++)
  //  r[i] = s[(t1d << 3) + i];
  __syncthreads();
  uint32_t idx = t1d % 128;
  uint3 t3d;
  Index3DFrom1D<8, 8, 2>(t3d, idx);
  //NTTInv1024Core(r, s, twd_inv, twd_sqrt_inv, t1d, t3d);
  NTTInv1024Core(r, ptr, twd_inv, twd_sqrt_inv, idx, t3d);
  // choose one of the first 1024 entries in a block of 8192 entries
  ptr = &s[(t1d%2)*8192 + (t1d/2)];
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    r[i] = ptr[i << 10];
  #pragma unroll
  for (int i = 0; i < 4; i ++)
    t[i] = con_twd[(t1d>>1)];
    //t[i] = con_twd[t1d];
  NTT2(r[0], r[1], t[0]);
  NTT2(r[2], r[3], t[1]);
  NTT2(r[4], r[6], t[2]);
  NTT2(r[6], r[7], t[3]);
  #pragma unroll
  for (int i = 0; i < 8; i ++)
  ptr[i << 10] = r[i];
  __threadfence();
  __syncthreads();

  ptr = &s[t1d<<2];
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    r[i] = ptr[((i >> 2) << (logN-1)) + (i&0x3)];
  // NTT16384
  #pragma unroll
  for (int i = 0; i < 4; i ++)
    t[i] = con_twd[(t1d<<2) + i];
  NTT8(r,t);
  //NTT8x2Lsh(r, t3d.z); // if (t1d >= 64) NTT8x2<1>(r);
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    ptr[((i >> 2) << (logN -1)) + (i&0x3)] = r[i];
  __threadfence();
  __syncthreads();
}

template <typename T>
__device__
void NTT16384(FFP* out,
             T* in,
             FFP* temp_shared,
             FFP* twd,
             FFP* twd_sqrt,
             uint32_t leading_thread) {
  uint32_t t1d = ThisBlockRankInGrid()*ThisBlockSize() 
		  + ThisThreadRankInBlock() - leading_thread;
  uint32_t logN = 14;
  uint3 t3d;
  //Index3DFrom1D<8, 8, 2>(t3d, t1d);
  register FFP r[8];
  #pragma unroll
  for (int i = 0; i < 8; i ++) {
    r[i] = FFP((T)in[(t1d<<2) + ((i >> 2) << (logN - 1)) + (i&0x3)]);
  }
  __syncthreads();
  NTT16384Core(r, temp_shared, twd, twd_sqrt, t1d);
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    out[(t1d<<2) + ((i >> 2) << (logN - 1)) + (i&0x3)] = r[i];
  __threadfence();
  __syncthreads();
}

template <typename T>
__device__
void NTTInv16384Add(T* out,
                   FFP* in,
                   FFP* temp_shared,
                   FFP* twd_inv,
                   FFP* twd_sqrt_inv,
                   uint32_t leading_thread) {
  uint32_t t1d = ThisBlockRankInGrid()*ThisBlockSize() 
		  + ThisThreadRankInBlock() - leading_thread;
  uint3 t3d;
//  Index3DFrom1D<8, 8, 2>(t3d, t1d);
  register FFP r[8];
  uint32_t logN = 14;
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    r[i] = in[(t1d<<2) + ((i >> 2) << (logN - 1)) + (i&0x3)];
  __syncthreads();
  NTTInv16384Core(r, temp_shared, twd_inv, twd_sqrt_inv, t1d);
  // mod 2^32 specifically
  uint64_t med = FFP::kModulus() / 2;
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    out[(t1d<<2) + ((i >> 2) << (logN - 1)) + (i&0x3)] 
	    += T(r[i].val() - (r[i].val() >= med));
  __threadfence();
  __syncthreads();
}

template <typename T>
__device__
void NTT4_(FFP* out, T* in, FFP* twd, FFP* twd_sqrt) {
  register FFP r[4];
  FFP t[2]; // twiddle factor
  #pragma unroll
  for (int i = 0; i < 4; i ++) {
    r[i] = FFP((T)in[i]);
    r[i] *= twd_sqrt[i];
  }
  for (int i = 0; i < 2; i ++)
    t[i] = twd[i];
  // ----- 1st stage -----
  NTT2(r[0], r[2], t[0]);
  NTT2(r[1], r[3], t[1]);
  // ----- 2nd stage -----
  NTT2(r[0], r[1], t[0]);
  NTT2(r[2], r[3], t[0]);
  for (int i = 0; i < 4; i ++)
    out[i] = r[i];
}

template <typename T>
__device__
void NTTInv4_(T* out, FFP* in, FFP* twd_inv, FFP* twd_sqrt_inv) {
  register FFP r[4];
  FFP t[2]; // twiddle factor
  #pragma unroll
  for (int i = 0; i < 4; i ++) 
    r[i] = in[i];
  for (int i = 0; i < 2; i ++) 
    t[i] = twd_inv[i];
  // ----- 1st stage -----
  NTTInv2(r[0], r[1], t[0]);
  NTTInv2(r[2], r[3], t[0]);
  // ----- 2nd stage -----
  NTTInv2(r[0], r[2], t[0]);
  NTTInv2(r[1], r[3], t[1]);
  for (int i = 0; i < 4; i ++) {
    r[i] *= twd_sqrt_inv[i];
    out[i] = T(r[i].val());
  }
}

template <typename T>
__device__
void NTT8_(FFP* out, T* in, FFP* twd, FFP* twd_sqrt) {
  register FFP r[8];
  FFP t[4]; // twiddle factor
  #pragma unroll
  for (int i = 0; i < 8; i ++) {
    r[i] = FFP((T)in[i]);
    r[i] *= twd_sqrt[i];
  }
  for (int i = 0; i < 4; i ++)
    t[i] = twd[i];
  // ----- 1st stage -----
  NTT2(r[0], r[4], t[0]);
  NTT2(r[1], r[5], t[1]);
  NTT2(r[2], r[6], t[2]);
  NTT2(r[3], r[7], t[3]);
#ifdef DEBUG
  printf("----- NTT8 1st stage ----\n");
  printf("r[0]=%d, r[1]=%d\n", r[0], r[1]);
  printf("r[2]=%d, r[3]=%d\n", r[2], r[3]);
  printf("r[4]=%d, r[5]=%d\n", r[4], r[5]);
  printf("r[6]=%d, r[7]=%d\n", r[6], r[7]);
#endif
  // ----- 2nd stage -----
  NTT2(r[0], r[2], t[0]);
  NTT2(r[1], r[3], t[2]);
  NTT2(r[4], r[6], t[0]);
  NTT2(r[5], r[7], t[2]);
#ifdef DEBUG
  printf("----- 2nd stage ----\n");
  printf("r[0]=%d, r[1]=%d\n", r[0], r[1]);
  printf("r[2]=%d, r[3]=%d\n", r[2], r[3]);
  printf("r[4]=%d, r[5]=%d\n", r[4], r[5]);
  printf("r[6]=%d, r[7]=%d\n", r[6], r[7]);
#endif
  // ----- 3rd stage -----
  NTT2(r[0], r[1], t[0]);
  NTT2(r[2], r[3], t[0]);
  NTT2(r[4], r[5], t[0]);
  NTT2(r[6], r[7], t[0]);
#ifdef DEBUG
  printf("----- 3rd stage ----\n");
  printf("r[0]=%d, r[1]=%d\n", r[0], r[1]);
  printf("r[2]=%d, r[3]=%d\n", r[2], r[3]);
  printf("r[4]=%d, r[5]=%d\n", r[4], r[5]);
  printf("r[6]=%d, r[7]=%d\n", r[6], r[7]);
#endif
  for (int i = 0; i < 8; i ++)
    out[i] = r[i];
}

template <typename T>
__device__
void NTTInv8_(T* out, FFP* in, FFP* twd_inv, FFP* twd_sqrt_inv) {
  register FFP r[8];
  FFP t[4]; // twiddle factor
  #pragma unroll
  for (int i = 0; i < 8; i ++) 
    r[i] = in[i];
  for (int i = 0; i < 8; i ++) 
    t[i] = twd_inv[i];
  // ----- 1st stage -----
  NTTInv2(r[0], r[1], t[0]);
  NTTInv2(r[2], r[3], t[0]);
  NTTInv2(r[4], r[5], t[0]);
  NTTInv2(r[6], r[7], t[0]);
#ifdef DEBUG
  printf("----- NTTInv8 1st stage ----\n");
  printf("r[0]=%d, r[1]=%d\n", r[0], r[1]);
  printf("r[2]=%d, r[3]=%d\n", r[2], r[3]);
  printf("r[4]=%d, r[5]=%d\n", r[4], r[5]);
  printf("r[6]=%d, r[7]=%d\n", r[6], r[7]);
#endif
  // ----- 2nd stage -----
  NTTInv2(r[0], r[2], t[0]);
  NTTInv2(r[1], r[3], t[2]);
  NTTInv2(r[4], r[6], t[0]);
  NTTInv2(r[5], r[7], t[2]);
#ifdef DEBUG
  printf("----- 2nd stage ----\n");
  printf("r[0]=%d, r[1]=%d\n", r[0], r[1]);
  printf("r[2]=%d, r[3]=%d\n", r[2], r[3]);
  printf("r[4]=%d, r[5]=%d\n", r[4], r[5]);
  printf("r[6]=%d, r[7]=%d\n", r[6], r[7]);
#endif
  // ----- 3rd stage -----
  NTTInv2(r[0], r[4], t[0]);
  NTTInv2(r[1], r[5], t[1]);
  NTTInv2(r[2], r[6], t[2]);
  NTTInv2(r[3], r[7], t[3]);
  for (int i = 0; i < 8; i ++) {
    r[i] *= twd_sqrt_inv[i];
    out[i] = T(r[i].val());
  }
#ifdef DEBUG
  printf("----- 3rd stage ----\n");
  printf("r[0]=%d, r[1]=%d\n", r[0], r[1]);
  printf("r[2]=%d, r[3]=%d\n", r[2], r[3]);
  printf("r[4]=%d, r[5]=%d\n", r[4], r[5]);
  printf("r[6]=%d, r[7]=%d\n", r[6], r[7]);
#endif
}
} // namespace cufhe
