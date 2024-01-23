#ifndef MYSTRATEGY_H
#define MYSTRATEGY_H

#include "baseStrategy.h"
#include "dtos/kline.h"

class MyStrategy : public BaseStrategy<Kline> {
public:
    MyStrategy(int64_t startTime, int64_t endTime, std::string symbol);

    void initialize() override;

    void onMarketData(const Kline& data) override;

    void onBar() override;

    void finalize() override;

private:
    void executeCUDACalculation();
    int64_t startTime;
    int64_t endTime;
    std::string symbol;
};

#endif
