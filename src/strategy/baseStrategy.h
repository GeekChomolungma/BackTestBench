#ifndef BASESTRATEGY_H
#define BASESTRATEGY_H
#include <vector>

class BaseStrategyInterface {
public:
    virtual ~BaseStrategyInterface() {}
};

template<typename DataT>
class BaseStrategy : public BaseStrategyInterface {
    public:
        virtual void initialize() = 0;
        virtual void onMarketData(std::vector<DataT>& data) = 0;
        virtual void onBar() = 0;
        virtual void finalize() = 0;
};

#endif