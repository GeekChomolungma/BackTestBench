#include "strategy/myStrategy.h"
#include <iostream>
#include <sstream>
#include <algorithm> // For std::max
#include <chrono>
#include <cmath>

void kernel_wrapper(int argc, const char* argv[], std::vector<Kline>& rawData, std::vector<std::pair<int, int>>& dataIndexes);

void tr_klines(std::vector<Kline>& inputK, int start, int end) {
    for (auto i = start; i < end + 1; i++) {
        if (i == start) {
            if (inputK[i].TrueRange == 0.0) {
                // initial TR
                inputK[i].TrueRange = inputK[i].High - inputK[i].Low;
            }
        }
        else {
            inputK[i].TrueRange = std::max(
                std::max(inputK[i].High - inputK[i].Low, fabs(inputK[i].High - inputK[i - 1].Close)),
                fabs(inputK[i].Low - inputK[i - 1].Close)
            );
        }
    }
}

void rma_klines(std::vector<Kline>& inputK, int start, int end, int length, int rmaType) {
    auto rmaAlpha = 1.0 / float(length);
    // we make sure the first element is either 0 element or a calculated history element
    // 0. High
    // 1. Atr

    switch (rmaType) {
    case 0:
        break;
    case 1:
        for (auto i = start; i < end + 1; i++) {
            if (i == start) {
                if (inputK[i].AveTrueRange == 0.0) {
                    // initial ATR
                    inputK[i].AveTrueRange = inputK[i].TrueRange;
                }
            }
            else {
                inputK[i].AveTrueRange = rmaAlpha * inputK[i].TrueRange + (1 - rmaAlpha) * inputK[i - 1].AveTrueRange;
            }
        }
        break;
    default:
        break;
    }
}

void st_klines(std::vector<Kline>& inputK, float factor, int start, int end) {
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
    }
}

MyStrategy::MyStrategy(int64_t startTime, int64_t endTime, std::string symbol) :
    startTime(startTime), endTime(endTime), symbol(symbol){
}

void MyStrategy::initialize() {
}

void MyStrategy::onMarketData(std::vector<Kline>& rawData, std::vector<std::pair<int, int>>& dataIndexes) {

    std::cout << "targetKlines size is " << rawData.size() << std::endl;

    this->executeCUDACalculation(rawData, dataIndexes);

    auto start = std::chrono::high_resolution_clock::now();
    this->onMarketData_HostBenchMark(rawData, dataIndexes);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "onMarketData on Host, Elapsed time: " << elapsed.count() << " ms\n";
}

void MyStrategy::onMarketData_HostBenchMark(std::vector<Kline>& data, std::vector<std::pair<int, int>>& dataIndexes) {
    for (auto idxPair : dataIndexes) {
            tr_klines(data, idxPair.first, idxPair.second);
            rma_klines(data, idxPair.first, idxPair.second, 10, 1);
            st_klines(data, 5.0, idxPair.first, idxPair.second);
    }
}

void MyStrategy::onBar() {
}

void  MyStrategy::finalize() {
}

void  MyStrategy::executeCUDACalculation(std::vector<Kline>& rawData, std::vector<std::pair<int, int>>& dataIndexes) {
    // for cuda process
    int argc = 0;
    const char* argv[1] = { "My strategy start running!" };
    kernel_wrapper(0, argv, rawData, dataIndexes);
    //int i = 0;
    //for (auto kline : rawData) {
    //    std::cout << i << "th kline Atr is " << kline.AveTrueRange << std::endl;
    //    i++;
    //}
}