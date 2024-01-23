#include "MyStrategy.h"
#include <iostream>
#include <sstream>

MyStrategy::MyStrategy(int64_t startTime, int64_t endTime, std::string symbol) :
    startTime(startTime), endTime(endTime), symbol(symbol){
}

void MyStrategy::initialize() {
}

void MyStrategy::onMarketData(const Kline& data) {
    std::stringstream ss;
    ss << "maket data:" << this->symbol << "span from" << this->startTime << "to" << this->endTime;
    std::cout << ss.str() << std::endl;
}

void MyStrategy::onBar() {
}

void  MyStrategy::finalize() {
}

void  MyStrategy::executeCUDACalculation() {
    
}