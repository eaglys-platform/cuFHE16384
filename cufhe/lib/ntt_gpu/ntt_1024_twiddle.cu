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

#include <include/ntt_gpu/ntt_1024_twiddle.cuh>
#include <include/details/error_gpu.cuh>
#include <include/details/assert.h>
#include <include/details/allocator_gpu.cuh>
#include <include/details/utils_gpu.cuh>


namespace cufhe {

__global__
#if 0
void __GenTwd__(FFP* twd, FFP* twd_inv) {
#else
void __GenTwd__(FFP* twd, FFP* twd_inv, uint32_t n) {
#endif
#if 0
  uint32_t n = 1024;
#endif
  uint32_t idx;
  uint32_t cid;
  FFP w = FFP::Root(n);
  FFP t;
  uint32_t e;
#if 1
  register uint32_t t1d = ThisBlockRankInGrid()*ThisBlockSize()+
                                ThisThreadRankInBlock();
  if (t1d<n) {
  e = idx = t1d;
  twd[idx] = FFP::Pow(w, e);
  twd_inv[idx] = FFP::Pow(w, (n - e) % n);
  //printf("twd[%d]=%lld, ", idx, twd[idx].val());
  //printf("twd_inv[%d]=%lld\n", idx, twd_inv[idx].val());
  }
  __syncthreads();
#else
  cid = (threadIdx.z << 6) + (threadIdx.y << 3) + threadIdx.x;
  for (int i = 0; i < 8; i ++) {
    e = (threadIdx.z * 8 + threadIdx.y / 4 * 4 + (threadIdx.x % 4))
      * (i * 8 + (threadIdx.y % 4) * 2 + threadIdx.x / 4);
    idx = (i * n / 8) + cid;
    twd[idx] = FFP::Pow(w, e);
    twd_inv[idx] = FFP::Pow(w, (n - e) % n);
  }
#endif
}

__global__
#if 0
void __GenTwdSqrt__(FFP* twd_sqrt, FFP* twd_sqrt_inv) {
#else
void __GenTwdSqrt__(FFP* twd_sqrt, FFP* twd_sqrt_inv, uint32_t n) {
#endif
#if 0
  uint32_t n = 1024;
  uint32_t idx = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
  FFP w = FFP::Root(2 * n);
  FFP n_inv = FFP::InvPow2(10);
  twd_sqrt[idx] = FFP::Pow(w, idx);
  twd_sqrt_inv[idx] = FFP::Pow(w, (2 * n - idx) % (2 * n)) * n_inv;
#else
  register uint32_t t1d = ThisBlockRankInGrid()*ThisBlockSize()+
                                ThisThreadRankInBlock();
  uint32_t log2N = __log2f(n);
  uint32_t idx = t1d;
  if (t1d<n) {
  FFP w = FFP::Root(2 * n);
  FFP n_inv = FFP::InvPow2(log2N);
  twd_sqrt[idx] = FFP::Pow(w, idx);
  twd_sqrt_inv[idx] = FFP::Pow(w, (2 * n - idx) % (2 * n)) * n_inv;
  //printf("twd_sqrt[%d]=%lld, ", idx, twd_sqrt[idx].val());
  //printf("twd_sqrt_inv[%d]=%lld\n", idx, twd_sqrt_inv[idx].val());
  } 
  __syncthreads();
#endif
}

template <>
void CuTwiddle<NEGATIVE_CYCLIC_CONVOLUTION>::Create(uint32_t size) {
  assert(this->twd_ == nullptr);
#if 0
  size_t nbytes = sizeof(FFP) * 1024 * 4;
#else
  size_t nbytes = sizeof(FFP) * size * 4;
#endif
#if 0
  this->twd_ = (FFP*)AllocatorGPU::New(nbytes).first;
#else
  CuSafeCall(cudaMalloc(&this->twd_, nbytes));
#endif
#if 0
  this->twd_inv_ = this->twd_ + 1024;
  this->twd_sqrt_ = this->twd_inv_ + 1024;
  this->twd_sqrt_inv_ = this->twd_sqrt_ + 1024;
#else
  this->twd_inv_ = this->twd_ + size;
  this->twd_sqrt_ = this->twd_inv_ + size;
  this->twd_sqrt_inv_ = this->twd_sqrt_ + size;
#endif
#if 0
  __GenTwd__<<<1, dim3(8, 8, 2)>>>(this->twd_, this->twd_inv_);
  __GenTwdSqrt__<<<16, 64>>>(this->twd_sqrt_, this->twd_sqrt_inv_);
#else
  __GenTwd__<<<32, 512>>>(this->twd_, this->twd_inv_, size);
  __GenTwdSqrt__<<<32, 512>>>(this->twd_sqrt_, this->twd_sqrt_inv_, size);
#endif
  cudaDeviceSynchronize();
  CuCheckError();
}

template <>
void CuTwiddle<NEGATIVE_CYCLIC_CONVOLUTION>::Destroy() {
  assert(this->twd_ != nullptr);
  CuSafeCall(cudaFree(this->twd_));
  this->twd_ = nullptr;
  this->twd_inv_ = nullptr;
  this->twd_sqrt_ = nullptr;
  this->twd_sqrt_inv_ = nullptr;
}

} // namespace cufhe
