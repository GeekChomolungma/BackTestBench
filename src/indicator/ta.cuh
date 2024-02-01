#ifndef TECHNICALANALYSIS_CUH
#define TECHNICALANALYSIS_CUH

#include <dtos/kline.h>

__global__ void ema_kernel(const Kline* inputK, float* output, int stockNum, int klineSize, int length, float alpha);

#endif