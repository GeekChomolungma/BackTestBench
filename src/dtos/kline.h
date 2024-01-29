#ifndef KLINE_H
#define KLINE_H

#include <string>

class Kline {
public:
    int64_t StartTime;
    int64_t EndTime;
    char Symbol[16];
    char Interval[16];
    int64_t FirstTradeID;
    int64_t LastTradeID;
    double  Open;
    double  Close;
    double  High;
    double  Low;
    double  Volume;
    int64_t  TradeNum;
    bool IsFinal;
    double  QuoteVolume;
    double  ActiveBuyVolume;
    double  ActiveBuyQuoteVolume;
};
#endif
