#include "strategy/myStrategy.h"
#include <iostream>
#include <sstream>

void kernel_wrapper(int argc, const char* argv[], std::vector<Kline>& rawData, std::vector<std::pair<int, int>>& dataIndexes);

MyStrategy::MyStrategy(int64_t startTime, int64_t endTime, std::string symbol) :
    startTime(startTime), endTime(endTime), symbol(symbol){
}

void MyStrategy::initialize() {
}

void MyStrategy::onMarketData(std::vector<Kline>& rawData, std::vector<std::pair<int, int>>& dataIndexes) {

    std::cout << "targetKlines size is " << rawData.size() << std::endl;

    this->executeCUDACalculation(rawData, dataIndexes);
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
    int i = 0;
    for (auto kline : rawData) {
        std::cout << i << "th kline Atr is " << kline.AveTrueRange << std::endl;
        i++;
    }
}