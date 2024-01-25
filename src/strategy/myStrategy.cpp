#include "MyStrategy.h"
#include <iostream>
#include <sstream>

void kernel_wrapper(int argc, const char* argv[], std::vector<Kline>& rawData);

MyStrategy::MyStrategy(int64_t startTime, int64_t endTime, std::string symbol) :
    startTime(startTime), endTime(endTime), symbol(symbol){
}

void MyStrategy::initialize() {
}

void MyStrategy::onMarketData(std::vector<Kline>& rawData) {
    //std::stringstream ss;
    //ss << "maket data:" << this->symbol << "span from" << this->startTime << "to" << this->endTime;
    //std::cout << ss.str() << std::endl;
    this->executeCUDACalculation(rawData);
}

void MyStrategy::onBar() {
}

void  MyStrategy::finalize() {
}

void  MyStrategy::executeCUDACalculation(std::vector<Kline>& rawData) {
    // for cuda process
    int argc = 0;
    const char* argv[1] = { "Burning GPU!!!" };
    kernel_wrapper(0, argv, rawData);   
}