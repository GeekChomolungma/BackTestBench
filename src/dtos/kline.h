#ifndef KLINE_H
#define KLINE_H

#include <string>

class Kline {
public:
    int64_t StartTime;
    int64_t EndTime;
    std::string Symbol;
    std::string Interval;
    int64_t FirstTradeID;
    int64_t LastTradeID;
    std::string Open;
    std::string Close;
    std::string High;
    std::string Low;
    std::string Volume;
    int64_t TradeNum;
    bool IsFinal;
    std::string QuoteVolume;
    std::string ActiveBuyVolume;
    std::string ActiveBuyQuoteVolume;
};
#endif
