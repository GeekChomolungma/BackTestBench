#ifndef TECHNICALANALYSIS_CUH
#define TECHNICALANALYSIS_CUH

#include <dtos/kline.h>

__global__ void test_kernel(Kline* inputK, int stockNum, int* deviceStartIndex, int* deviceEndIndex, int length, float alpha);

__device__ void ema_cuda_klines(int idx, Kline* inputK, float* output, int klineSize, int length, float alpha);

__device__ void rma_cuda_klines(int idx, Kline* inputK, int start, int end, int length, int rmaType);

__device__ void tr_cuda_klines(int idx, Kline* inputK, int start, int end);

__device__ void st_cuda_klines(int idx, Kline* inputK, float factor, int start, int end);

#endif