#include "strategy/baseStrategy.h"
#include "db/mongoManager.h"
#include "settlement/settlement.h"
#include "threadPool/threadPool.h"

#include <sstream>
#include <iomanip> // std::setw, std::setfill
#include <iostream>
#include <map>


class BacktestingPlatform {
public:
    BacktestingPlatform(std::string uriCfg) :dbManager(uriCfg), settleInstance(dbManager){
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
       
        //int i = 0;
        //for (auto dIndex : dataIndexes) {
        //    for (int start = dIndex.first; start <= dIndex.second; start++) {
        //        auto k = targetData[start];
        //        std::cout << "Pending Calculate Collection: " << colNameList[i] << " " << start << " th Kline, start time is : " << k.StartTime << " open is : " << k.Open
        //            << " close is: " << k.Close << " high is: " << k.High << " low is: " << k.Low << "\n" << std::endl;
        //    }
        //    i++;
        //}

        // send orders
        try {
            // exec the calculation
            strategyInst->onMarketData(targetData, dataIndexes);

            // exec settlement
            this->runSettlement(targetData, colNameList, dataIndexes);

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
        }
        catch (const std::exception& e) {
            // error 
            std::cerr << "exception: " << e.what() << std::endl;
        }
        catch (...) {
            std::cerr << "unkown error:" << std::endl;
        }
    };

    template <typename T> void runStrategyRealTime(
        BaseStrategy<T>* strategyInst, int64_t startTime, std::string dbName, std::vector<std::string> symbols, std::string interval
    ) {
        std::vector<Kline> targetData;
        std::vector<std::vector<Kline>&> watchDataGroup(symbols.size());
        std::vector<std::pair<int, int>> dataIndexes;
        std::vector<std::string> colNameList;
        ThreadPool TpInterval(symbols.size());



        auto i = 0;
        for (auto s : symbols) {
            std::ostringstream oss;
            oss << "Binance-" << s << "-" << interval;
            std::string colName = oss.str();
            colNameList.push_back(colName);

            std::vector<Kline> prevTwoKlines;
            watchDataGroup[i] = prevTwoKlines;

            auto realTimeUpdateWatchTask = boost::bind(&MongoManager::WatchKlineUpdate,
                &this->dbManager,
                dbName,
                colName,
                prevTwoKlines);

            TpInterval.Enqueue(realTimeUpdateWatchTask);

            i++
        }
        TpInterval.WaitAll();
        
        // combine watchDataGroup element to targetData
        int startIndex = 0;
        for (auto i = 0; i < symbols.size() i++) {
            if (watchDataGroup[i].size() == 0) {
                colNameList.erase(colNameList.begin() + i);
                continue;
            }
            targetData.insert(targetData.end(), watchDataGroup[i].begin(), watchDataGroup[i].end());
            
            int endIndex = targetData.size() - 1;
            dataIndexes.push_back(std::pair<int, int>(startIndex, endIndex));
            startIndex = targetData.size();
        }

        try {
            // exec the calculation
            strategyInst->onMarketData(targetData, dataIndexes);

            // exec settlement
            this->runSettlement(targetData, colNameList, dataIndexes);

            // update Kline one by one with Bulk
            int colNameIndex = 0;
            std::cout << interval + " dataIndexes size is: " << targetData.size() << "\n" << std::endl;
            for (auto dIndex : dataIndexes) {
                std::vector<T> unitData(targetData.begin() + dIndex.first, targetData.begin() + dIndex.second + 1);
                this->dbManager.BulkWriteByIds(dbName, colNameList[colNameIndex], unitData);
                colNameIndex++;
            }
        }
        catch (const std::exception& e) {
            // error 
            std::cerr << "exception: " << e.what() << std::endl;
        }
        catch (...) {
            std::cerr << "unkown error:" << std::endl;
        }
    };

    //void SendOrders();
    template <typename T> void runSettlement(std::vector<T>& targetData, std::vector<std::string>& colNameList, std::vector<std::pair<int, int>>& dataIndexes){
        int colNameIndex = 0;
        for (auto dIndex : dataIndexes) {
            std::cout << "RunSettlement, colName:" + colNameList[colNameIndex] << "\n" << std::endl;
            std::vector<T> unitData(targetData.begin() + dIndex.first, targetData.begin() + dIndex.second + 1);
            settleInstance.ExecSettlement(unitData, colNameList[colNameIndex]);
            colNameIndex++;
        }        
    };

private:
    MongoManager dbManager;
    SettlementModule settleInstance;
};