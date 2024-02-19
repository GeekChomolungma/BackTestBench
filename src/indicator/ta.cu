#include "indicator/ta.cuh"
#include <stdio.h>

__global__ void test_kernel(Kline* inputK, float* output, int stockNum, int klineSize, int length, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < stockNum) {
        tr_cuda_klines(idx, inputK, klineSize);
        rma_cuda_klines(idx, inputK, output, klineSize, length, 1);
    }
}

__device__ void ema_cuda_klines(int idx, Kline* inputK, float* output, int klineSize, int length, float alpha) {
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

__device__ void rma_cuda_klines(int idx, Kline* inputK, float* output, int klineSize, int length, int rmaType) {
    auto rmaAlpha = 1.0 / float(length);
    // 0. High
    // 1. Atr

    switch (rmaType) {
        case 0:
            for (auto i = 0 + idx * klineSize; i < (idx + 1) * klineSize; i++) {
                if (i < length + idx * klineSize) {
                    output[i] = inputK[i].High;
                }
                else {
                    output[i] = rmaAlpha * inputK[i].High + (1 - rmaAlpha) * output[i - 1];
                }
                printf("rma_cuda_klines stock number: %d, %dth RMA, \t\t :%f\n", idx, i, output[i]);
            }
            break;
        case 1:
            for (auto i = 0 + idx * klineSize; i < (idx + 1) * klineSize; i++) {
                if (i < length + idx * klineSize) {
                    if (inputK[i].AveTrueRange == 0.0) {
                        // initial ATR
                        output[i] = inputK[i].TrueRange;
                        inputK[i].AveTrueRange = inputK[i].TrueRange;
                    }
                    else {
                        output[i] = inputK[i].AveTrueRange;
                    }
                }
                else {
                    output[i] = rmaAlpha * inputK[i].TrueRange + (1 - rmaAlpha) * output[i - 1];
                    inputK[i].AveTrueRange = rmaAlpha * inputK[i].TrueRange + (1 - rmaAlpha) * inputK[i - 1].AveTrueRange;
                }
                printf("rma_cuda_klines type: AveTrueRange, stock number: %d, %dth RMA, \t\t :%f\n", idx, i, output[i]);
            }
            break;
        default:
            break;
    }
}

__device__ void tr_cuda_klines(int idx, Kline* inputK, int klineSize) {
    for (auto i = 0 + idx * klineSize; i < (idx + 1) * klineSize; i++) {
        if (i == 0 + idx * klineSize){
            if (inputK[i].TrueRange == 0.0) {
                // initial TR
                inputK[i].TrueRange = inputK[i].High - inputK[i].Low;
            }
        }
        else {
            inputK[i].TrueRange = max(
                max(inputK[i].High - inputK[i].Low, fabs(inputK[i].High - inputK[i - 1].Close)),
                fabs(inputK[i].Low - inputK[i - 1].Close)
            );
        }
        printf("tr_cuda_klines, stock number: %d, %dth High: %f, Low: %f, Close: %f, TrueRange: %f\n", idx, i, 
            inputK[i].High, inputK[i].Low, inputK[i].Close, inputK[i].TrueRange);
    }
}
