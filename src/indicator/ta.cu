#include "indicator/ta.cuh"
#include <stdio.h>

__global__ void test_kernel(Kline* inputK, float* output, int stockNum, int* deviceStartIndex, int* deviceEndIndex, int length, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < stockNum) {
        tr_cuda_klines(idx, inputK, deviceStartIndex[idx], deviceEndIndex[idx]);
        rma_cuda_klines(idx, inputK, output, deviceStartIndex[idx], deviceEndIndex[idx], length, 1);
        st_cuda_klines(idx, inputK, 5.0, deviceStartIndex[idx], deviceEndIndex[idx]);
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
        // printf("ema_cuda stock number: %d, %dth EMA, \t\t :%f\n", idx, i, output[i]);
    }
}

__device__ void rma_cuda_klines(int idx, Kline* inputK, float* output, int start, int end, int length, int rmaType) {
    auto rmaAlpha = 1.0 / float(length);
    // we make sure the first element is either 0 element or a calculated history element
    // 0. High
    // 1. Atr

    switch (rmaType) {
        case 0:
            for (auto i = start; i < end + 1; i++) {
                if (i < start + length) {
                    output[i] = inputK[i].High;
                }
                else {
                    output[i] = rmaAlpha * inputK[i].High + (1 - rmaAlpha) * output[i - 1];
                }
                printf("rma_cuda_klines stock number: %d, %dth RMA, \t\t :%f\n", idx, i, output[i]);
            }
            break;
        case 1:
            for (auto i = start; i < end + 1; i++) {
                if (i == start) {
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
                // printf("rma_cuda_klines type: AveTrueRange, stock number: %d, %dth RMA, \t\t :%f\n", idx, i, output[i]);
            }
            break;
        default:
            break;
    }
}

__device__ void tr_cuda_klines(int idx, Kline* inputK, int start, int end) {
    for (auto i = start; i < end + 1; i++) {
        if (i == start){
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
        //printf("tr_cuda_klines, stock number: %d, %dth High: %f, Low: %f, Close: %f, TrueRange: %f\n", idx, i, 
        //    inputK[i].High, inputK[i].Low, inputK[i].Close, inputK[i].TrueRange);
    }
}


// 
//method st(bar b, simple float factor, simple int len) = >
//    float atr = b.atr(len)
//    float up = b.src('hl2') + factor * atr
//    up : = up < nz(up[1]) or b.c[1] > nz(up[1]) ? up : nz(up[1])
//    float dn = b.src('hl2') - factor * atr
//    dn : = dn > nz(dn[1]) or b.c[1] < nz(dn[1]) ? dn : nz(dn[1])
//
//    float st = na
//    int   dir = na
//    dir : = switch
//    na(atr[1]) = > 1
//    st[1] == nz(up[1]) = > dir : = b.c > up ? -1 : +1
//    = > dir : = b.c < dn ? +1 : -1
//    st : = dir == -1 ? dn : up
//
//    supertrend.new(st, dir)
__device__ void st_cuda_klines(int idx, Kline* inputK, float factor, int start, int end) {
    // kline.atr has involved length, 
    // so this SuperTrend Calculated should be called and coupled with Atr func

    for (auto i = start; i < end + 1; i++) {

        double hl2 = (inputK[i].High + inputK[i].Low) / 2.0;
        double facAtr = factor * inputK[i].AveTrueRange;

        double stUpper = hl2 + facAtr;
        double stLower = hl2 - facAtr;

        if (i == start) {
            // StUp and StDown:
            // 0.0 nothing to do
            // non 0.0, nothing to do, keep it
            if (inputK[i].AveTrueRange == 0) { inputK[i].STDirection = 1; }
        }
        else {
            inputK[i].StUp = (stUpper < inputK[i - 1].StUp || inputK[i - 1].Close > inputK[i - 1].StUp) ? stUpper : inputK[i - 1].StUp;
            inputK[i].StDown = (stLower > inputK[i - 1].StDown || inputK[i - 1].Close < inputK[i - 1].StDown) ? stLower : inputK[i - 1].StDown;
            if (inputK[i - 1].SuperTrendValue == inputK[i - 1].StUp) { 
                inputK[i].STDirection = (inputK[i].Close > inputK[i].StUp) ? -1 : 1; 
            }
            else {
                inputK[i].STDirection = (inputK[i].Close < inputK[i].StDown) ? 1 : -1;
            }
            
            //alerts a = alerts.new(
            //    math.sign(ta.change(st.d)) == 1,
            //    math.sign(ta.change(st.d)) == -1)
            inputK[i].Action = ((inputK[i].STDirection - inputK[i - 1].STDirection) > 0) ? 1 : (((inputK[i].STDirection - inputK[i - 1].STDirection) < 0) ? 2 : 0);
        }
        
        inputK[i].SuperTrendValue = (inputK[i].STDirection == -1) ? inputK[i].StDown : inputK[i].StUp;

        //printf("st_cuda_klines, stock number: %d, %dth StUp: %f, StDown: %f, SuperTrendValue: %f, STDirection: %d, Action: %d\n", idx, i,
        //    inputK[i].StUp, inputK[i].StDown, inputK[i].SuperTrendValue, inputK[i].STDirection, inputK[i].Action);
    }
}
