#include "strategy/baseStrategy.h"
#include "db/mongoManager.h"
#include <sstream>
#include <iostream>


class BacktestingPlatform {
public:
    BacktestingPlatform(std::string uriCfg) :dbManager(uriCfg){
    }

    // run
    template <typename T> void runBacktest(BaseStrategy<T>* strategy, std::vector<T>& rawData) {
        strategy->onMarketData(rawData, 1);
    };

    template <typename T> void runStrategyTask(
        BaseStrategy<T>* strategyInst, int64_t startTime, int64_t endTime, std::string dbName, std::vector<std::string> symbols, std::string interval
    ) {
        std::vector<T> targetData;
        std::vector<std::pair<int,int>> dataIndexes;
        std::vector<std::string> colNameList;
        int startIndex = 0;
        

        for (auto s : symbols) {
            std::ostringstream oss;
            oss << "Binance-" << s << "-" << interval;
            std::string colName = oss.str();
            colNameList.push_back(colName);

            this->dbManager.GetKline(startTime, endTime, dbName, colName, targetData);
            if (targetData.size() == startIndex) {
                continue;
            }

            int endIndex = targetData.size() - 1;
            dataIndexes.push_back(std::pair<int, int>(startIndex, endIndex));
            startIndex = targetData.size();
        }
       
        int i = 0;
        for (auto k : targetData) {
            std::cout << "targetKlines" << i << " th element, start time is: " << k.StartTime << " open is: " << k.Open
                << " close is: " << k.Close << " high is: " << k.High << " low is: " << k.Low << std::endl;
            i++;
        }

        // exec the calculation
        strategyInst->onMarketData(targetData, dataIndexes);

        // update Kline one by one with Bulk
        int colNameIndex = 0;
        for (auto dIndex : dataIndexes) {
            std::vector<T> unitData(targetData.begin() + dIndex.first, targetData.begin() + dIndex.second + 1);
            this->dbManager.BulkWriteByIds(dbName, colNameList[colNameIndex], unitData);
            colNameIndex++;
        }
    };

    MongoManager dbManager;
};