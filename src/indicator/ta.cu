#include "indicator/ta.cuh"
#include <stdio.h>

__global__ void ema_kernel(const Kline* inputK, float* output, int stockNum, int klineSize, int length, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < stockNum) {
        for (auto i = 0 + idx * klineSize; i < (idx + 1) * klineSize; i++) {
            if (i < length+idx * klineSize) {
                output[i] = inputK[i].High;
            } else {
                output[i] = alpha * inputK[i].High + (1 - alpha) * output[i - 1];
            }
            printf("stock number: %d, %dth EMA, \t\t :%f\n", idx, i, output[i]);
        }
    }
}
