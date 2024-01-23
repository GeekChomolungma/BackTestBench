#include "strategy/baseStrategy.h"


class BacktestingPlatform {
public:
    // run
    template <typename T> void runBacktest(BaseStrategy<T>* strategy, T data) {
        strategy->onMarketData(data);
    };
};