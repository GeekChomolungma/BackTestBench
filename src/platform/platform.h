#include "strategy/baseStrategy.h"


class BacktestingPlatform {
public:
    // run
    template <typename T> void runBacktest(BaseStrategy<T>* strategy, std::vector<T>& rawData) {
        strategy->onMarketData(rawData);
    };
};