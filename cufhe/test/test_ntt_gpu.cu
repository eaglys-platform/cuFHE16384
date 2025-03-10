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

// Include these two files for GPU computing.
#include <include/cufhe_gpu.cuh>
#include <include/ntt_gpu/ntt.cuh>
#include <include/bootstrap_gpu.cuh>
#include <include/details/error_gpu.cuh>

using namespace cufhe;

#include <iostream>
#include <unistd.h>
#include <stdio.h>
#include <math.h>

#define N_BUTTERFLY 1

#define MAX_THREAD_PER_BLOCK 1024

#define NBLOCK (LARGE_N / (2 * N_BUTTERFLY) < MAX_THREAD_PER_BLOCK ? LARGE_N / (2 * N_BUTTERFLY) : MAX_THREAD_PER_BLOCK)

#define NGRID_X ((LARGE_N / (2 * N_BUTTERFLY)) % MAX_THREAD_PER_BLOCK == 0 ? (LARGE_N / (2 * N_BUTTERFLY)) / MAX_THREAD_PER_BLOCK : (LARGE_N / (2 * N_BUTTERFLY)) / MAX_THREAD_PER_BLOCK + 1)

using namespace std;

using clock_value_t = long long;

void CuErrorCheck(const char *file, const int line)
{
  cudaError err = cudaGetLastError();
  if (cudaSuccess != err)
  {
    fprintf(stderr, "Cuda Error: %s at at %s:%i\n", cudaGetErrorString(err), file, line);
    exit(-1);
  }
}

__device__ void sleep(clock_value_t sleep_cycle)
{
  clock_value_t start = clock64();
  clock_value_t cycles_elapsed;
  do
  {
    cycles_elapsed = clock64() - start;
  } while (cycles_elapsed < sleep_cycle);
}

// template <uint32_t LARGE_N, uint32_t SMALL_K, uint32_t SMALL_L>
struct Ctx
{
  Ctx(uint32_t length)
  {
    F = nullptr;
    F_ = nullptr;
    bufsize = length * sizeof(FFP);
    cudaMallocHost(&F, bufsize);
    cudaMalloc(&F_, bufsize);
  }
  ~Ctx()
  {
    cudaFreeHost(F);
    cudaFree(F_);
  }
  FFP *F;  // host
  FFP *F_; // device
  uint32_t bufsize;
};

__global__ void __do_ntt(FFP *out, FFP *in, uint32_t stage,
                         CuNTTHandler<LARGE_N, NEGATIVE_CYCLIC_CONVOLUTION> ntt,
                         int test)
{
  ntt.template NTT_<FFP, N_BUTTERFLY>(out, in, stage, test);
}

__global__ void __do_intt(FFP *out, FFP *in, uint32_t stage,
                          CuNTTHandler<LARGE_N, NEGATIVE_CYCLIC_CONVOLUTION> ntt,
                          int test)
{
  ntt.template NTTInv_<FFP, N_BUTTERFLY>(out, in, stage, test); // same as above
}

void do_check(Ctx *ctx)
{
  bool correct = true;
  for (int i = 0; i < LARGE_N; i++)
  {
    Torus val = Torus(ctx->F[i].val());
    printf("f[%d]=%ld\n", i, val);
    if (val != i)
      correct = false;
  }
  if (correct)
    cout << "PASS" << endl;
  else
    cout << "FAIL" << endl;
}

void do_ntt(Ctx *ctx,
            CuNTTHandler<LARGE_N, NEGATIVE_CYCLIC_CONVOLUTION> *ntt, int test)
{
  uint32_t log2N = (uint32_t)log2f(LARGE_N);
  cudaMemcpy(ctx->F_, ctx->F, ctx->bufsize, cudaMemcpyHostToDevice);
#if 1 // set to 1 for when cheking ntt correctness
#if N_BUTTERFLY == 1 && LARGE_N > 1024
  dim3 grid_ntt(NGRID_X, (SMALL_K + 1) * SMALL_L, 1);
#else
  dim3 grid_ntt(NGRID_X, (SMALL_K + 1) * SMALL_L, 1);
#endif
  dim3 grid_intt(NGRID_X, 2, 1);
#else
  dim3 grid_ntt(NGRID_X, 1, 1);
  dim3 grid_intt(NGRID_X, 1, 1);
#endif

  for (int i = 0; i < SMALL_N; i++)
  {
    int gadget_decomp_loop = 1; // (N_BUTTERFLY == 1 && LARGE_N > 1024 ? SMALL_L : 1);
    for (uint32_t l = 0; l < gadget_decomp_loop; l++)
    {
      for (uint32_t j = 0; j < log2N; j++)
      {
        // cout<< "------ Doing NTT(" << j << ")------" <<endl;
        __do_ntt<<<grid_ntt, NBLOCK, 0, 0>>>(ctx->F_ + (SMALL_K + 1) * l, ctx->F_ + (SMALL_K + 1) * l, j, *ntt, test);
        if (NGRID_X > 1)
          cudaDeviceSynchronize();
        CuErrorCheck(__FILE__, __LINE__);
      }
    }
    for (uint32_t j = 0; j < log2N; j++)
    {
      // cout<< "------ Doing INTT(" << j << ")------" <<endl;
      __do_intt<<<grid_intt, NBLOCK, 0, 0>>>(ctx->F_, ctx->F_, j, *ntt, test);
      if (NGRID_X > 1)
        cudaDeviceSynchronize();
      CuErrorCheck(__FILE__, __LINE__);
    }
    // cout<< "------ INTT Done ------" <<endl;
  }
  cudaMemcpy(ctx->F, ctx->F_, ctx->bufsize, cudaMemcpyDeviceToHost);
  // do_check(ctx); // uncomment for when checking ntt correctness
}

int main()
{
  cudaSetDevice(0);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  uint32_t kNumSMs = prop.multiProcessorCount;
#if 1
  uint32_t kNumTests = 100; // kNumSMs * 32;// * 8;
#else
  uint32_t kNumTests = 1; // kNumSMs * 32;// * 8;
#endif
  bool correct;

  correct = true;

  printf("N=%d, n=%d, k=%d, l=%d, NGRID_X=%d, NBLOCK=%d, N_BUTTERFLY=%d\n",
         LARGE_N, SMALL_N, SMALL_K, SMALL_L, NGRID_X, NBLOCK, N_BUTTERFLY);
  Ctx *ctx = new Ctx(LARGE_N * ((SMALL_K + 1) * SMALL_L));
  CuNTTHandler<LARGE_N, NEGATIVE_CYCLIC_CONVOLUTION> *ntt =
      new CuNTTHandler<LARGE_N, NEGATIVE_CYCLIC_CONVOLUTION>();
  ntt->Create(LARGE_N);
#if 1
  for (int test = 0; test < 3; test++)
  {
#else
  for (int test = 0; test < 1; test++)
  {
#endif
#if 1 // set to 0 for when cheking ntt correctness
    for (int j = 0; j < LARGE_N; j++)
      ((Torus *)ctx->F)[j] = Torus(j);
#else
    for (int j = 0; j < LARGE_N; j++)
      ctx->F[j] = FFP(j);
#endif
    switch (test)
    {
    case 0:
      cout << "----- test for real NTT/INTT -----" << endl;
      break;
    case 1:
      cout << "----- test for pseudo NTT/INTT w R/W only -----" << endl;
      break;
    case 2:
      cout << "----- test for pseudo NTT/INTT w thread sync only -----" << endl;
      break;
    }

    float et;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    for (int i = 0; i < kNumTests; i++)
      do_ntt(ctx, ntt, test);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&et, start, stop);
    cout << et / kNumTests << " ms / Blind Rotate" << endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
  delete ctx;
  ntt->Destroy();
  delete ntt;

  return 0;
}
