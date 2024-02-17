#include "indicator/ta.cuh"
#include <stdio.h>

__global__ void ema_kernel(const Kline* inputK, float* output, int stockNum, int klineSize, int length, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < stockNum) {
        ema_cuda_klines(idx, inputK, output, klineSize, length, alpha);
        rma_cuda_klines(idx, inputK, output, klineSize, length);
    }
}

__device__ void ema_cuda_klines(int idx, const Kline* inputK, float* output, int klineSize, int length, float alpha) {
    for (auto i = 0 + idx * klineSize; i < (idx + 1) * klineSize; i++) {
        if (i < length + idx * klineSize) {
            output[i] = inputK[i].High;
        }
        else {
            output[i] = alpha * inputK[i].High + (1 - alpha) * output[i - 1];
        }
        printf("ema_cuda stock number: %d, %dth EMA, \t\t :%f\n", idx, i, output[i]);
    }
}

__device__ void rma_cuda_klines(int idx, const Kline* inputK, float* output, int klineSize, int length) {
    auto rmaAlpha = 1.0 / float(length);
    for (auto i = 0 + idx * klineSize; i < (idx + 1) * klineSize; i++) {
        if (i < length + idx * klineSize) {
            output[i] = inputK[i].High;
        }
        else {
            output[i] = rmaAlpha * inputK[i].High + (1 - rmaAlpha) * output[i - 1];
        }
        printf("rma_cuda_klines stock number: %d, %dth RMA, \t\t :%f\n", idx, i, output[i]);
    }
}

__device__ void rma_cuda(int idx, const float* input, float* output, int klineSize, int length) {
    auto rmaAlpha = 1.0 / float(length);
    for (auto i = 0 + idx * klineSize; i < (idx + 1) * klineSize; i++) {
        if (i < length + idx * klineSize) {
            output[i] = input[i];
        }
        else {
            output[i] = rmaAlpha * input[i] + (1 - rmaAlpha) * output[i - 1];
        }
        printf("rma_cuda stock number: %d, %dth RMA, \t\t :%f\n", idx, i, output[i]);
    }
}

__device__ void tr_cuda_klines(int idx, const Kline* inputK, float* output, int klineSize) {
    for (auto i = 0 + idx * klineSize; i < (idx + 1) * klineSize; i++) {
        if i == 0 + idx * klineSize{
            output[i] = inputK[i].High - inputK[i].Low;
        }
        else {
            output[i] = max(
                max(inputK[i].High - inputK[i].Low, fabs(inputK[i].High - inputK[i - 1].Close)),
                fabs(inputK[i].Low - inputK[i - 1].Close)
            )
        }
    }
}

__device__ void atr_cuda_klines(int idx, const Kline* inputK, float* output, int klineSize, int length) {
    tr_cuda_klines(idx, inputK, output, klineSize);
    rma_cuda(idx, output, output, klineSize, length)
}