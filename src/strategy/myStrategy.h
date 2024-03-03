#ifndef MYSTRATEGY_H
#define MYSTRATEGY_H

#include "baseStrategy.h"
#include "dtos/kline.h"
#include <string>

class MyStrategy : public BaseStrategy<Kline> {
public:
    MyStrategy(int64_t startTime, int64_t endTime, std::string symbol);

    void initialize() override;

    void onMarketData(std::vector<Kline>& data, std::vector<std::pair<int, int>>& dataIndexes) override;
   
    void onMarketData_HostBenchMark(std::vector<Kline>& data, std::vector<std::pair<int, int>>& dataIndexes);

    void onBar() override;

    void finalize() override;

private:
    void executeCUDACalculation(std::vector<Kline>& rawData, std::vector<std::pair<int, int>>& dataIndexes);
    int64_t startTime;
    int64_t endTime;
    std::string symbol;
};

#endif
