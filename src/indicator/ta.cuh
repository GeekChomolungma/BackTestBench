#ifndef TECHNICALANALYSIS_CUH
#define TECHNICALANALYSIS_CUH

#include <dtos/kline.h>

__global__ void ema_kernel(const Kline* inputK, float* output, int stockNum, int klineSize, int length, float alpha);

__device__ void ema_cuda_klines(int idx, const Kline* inputK, float* output, int klineSize, int length, float alpha);

__device__ void rma_cuda_klines(int idx, const Kline* inputK, float* output, int klineSize, int length);
__device__ void rma_cuda(int idx, const float* input, float* output, int klineSize, int length);

__device__ void tr_cuda_klines(int idx, const Kline* inputK, float* output, int klineSize, int length);

__device__ void atr_cuda_klines(int idx, const Kline* inputK, float* output, int klineSize, int length);

#endif