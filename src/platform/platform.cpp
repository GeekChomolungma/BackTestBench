#include "platform/platform.h"
#include "dtos/kline.h"

template <typename T>
void BacktestingPlatform::runBacktest(BaseStrategy<T>* strategy, T data) {
    strategy->onMarketData(data);
}

