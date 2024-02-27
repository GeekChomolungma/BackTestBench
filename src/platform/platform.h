#include "strategy/baseStrategy.h"
#include "db/mongoManager.h"
#include <sstream>
#include <iomanip> // std::setw, std::setfill
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
            
            this->dbManager.GetKline(startTime, endTime, dbName, colName, targetData);
            std::cout << "GetKline colName:" << colName << ", size is:" << targetData.size() - startIndex << "\n" << std::endl;
            if (targetData.size() == startIndex) {
                continue;
            }

            colNameList.push_back(colName);
            int endIndex = targetData.size() - 1;
            dataIndexes.push_back(std::pair<int, int>(startIndex, endIndex));
            startIndex = targetData.size();
        }
       
        int i = 0;
        for (auto dIndex : dataIndexes) {
            for (int start = dIndex.first; start <= dIndex.second; start++) {
                auto k = targetData[start];
                std::cout << "Pending Calculate Collection: " << colNameList[i] << " " << start << " th Kline, start time is : " << k.StartTime << " open is : " << k.Open
                    << " close is: " << k.Close << " high is: " << k.High << " low is: " << k.Low << "\n" << std::endl;
            }
            i++;
        }

        // exec the calculation
        strategyInst->onMarketData(targetData, dataIndexes);

        // update Kline one by one with Bulk
        int colNameIndex = 0;
        std::cout << interval + " dataIndexes size is: " << targetData.size() << "\n" << std::endl;
        for (auto dIndex : dataIndexes) {

            //Kline kline0 = static_cast<Kline>(targetData[dIndex.first]);
            //std::cout << "Write Back collection: " << colNameList[colNameIndex] << " first kline ID is:";
            //for (int i = 0; i < 12; ++i) {
            //    std::cout << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(kline0.Id[i]);
            //}
            //std::cout << std::dec << "\n" << std::endl;

            std::vector<T> unitData(targetData.begin() + dIndex.first, targetData.begin() + dIndex.second + 1);
            this->dbManager.BulkWriteByIds(dbName, colNameList[colNameIndex], unitData);
            colNameIndex++;
        }
    };

    MongoManager dbManager;
};