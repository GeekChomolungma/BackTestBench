#include "strategy/baseStrategy.h"
#include "db/mongoManager.h"
#include "settlement/settlement.h"
#include "threadPool/threadPool.h"

#include <sstream>
#include <iostream>
#include <iomanip> // std::setw, std::setfill
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
        BaseStrategy<T>* strategyInst, std::string dbName, std::vector<std::string> symbols, std::string interval
    ) {
        std::vector<int64_t> currentStartTimes(symbols.size());
        while (true) {
            std::vector<Kline> targetData;
            std::vector<std::pair<int,int>> dataIndexes;
            std::vector<std::string> colNameList;
            int startIndex = 0;

            auto i = 0;
            for (auto s : symbols) {
                std::ostringstream oss;
                oss << "Binance-" << s << "-" << interval;
                std::string colName = oss.str();

                // fetch the synced time as get line end time.
                auto syncedTime = this->dbManager.GetSynedFlag("marketSyncFlag" , colName);

                std::vector<Kline> fetchedDataPerCol;
                //this->dbManager.GetLatestSyncedKlines(syncedTime, 200, dbName, colName, fetchedDataPerCol);
                this->dbManager.GetKline(currentStartTimes[i], syncedTime, 5000, 1, dbName, colName, fetchedDataPerCol);
                std::ostringstream ss;
                ss << "runStrategyTask, GetKline colName:" << colName << ", size is:" << fetchedDataPerCol.size();
                ss << "  index is: " << i << " start time is: " << currentStartTimes[i] << ", synced time is: " << syncedTime;
                ss << "\n" << std::endl;
                std::cout << ss.str();
                if (fetchedDataPerCol.size() == 0) {
                    continue;
                }

                if (fetchedDataPerCol.front().StartTime == syncedTime) {
                    continue;
                }

                currentStartTimes[i] = fetchedDataPerCol.back().StartTime;
                i++;

                //for (auto data : fetchedDataPerCol) {
                //    std::cout << "runStrategyTask: GetLatestSyncedKlines for Col: " + colName << " start time is: " << data.StartTime << "\n" << std::endl;
                //}

                targetData.insert(targetData.end(), fetchedDataPerCol.begin(), fetchedDataPerCol.end());
                colNameList.push_back(colName);
                int endIndex = targetData.size() - 1;
                dataIndexes.push_back(std::pair<int, int>(startIndex, endIndex));
                startIndex = targetData.size();
            }

            if (targetData.size() == 0) {
                return;
            }
            
            // send orders
            try {
                // exec the calculation
                strategyInst->onMarketData(targetData, dataIndexes);

                // exec settlement
                this->runSettlement(targetData, colNameList, dataIndexes);

                // update Kline one by one with Bulk
                int colNameIndex = 0;
                std::ostringstream ss;
                 ss << interval + " dataIndexes size is: " << targetData.size() << "\n" << std::endl;
                std::cout << ss.str();
           
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
        }
    };

    template <typename T> void runStrategyRealTime(
        BaseStrategy<T>* strategyInst, int64_t startTime, std::string dbName, std::vector<std::string> symbols, std::string interval
    ) {
        std::ostringstream ss;
        ss << "runStrategyRealTime begin for interval: " + interval << std::endl;
        std::cout << ss.str();

        try {
            std::vector<Kline> targetData;
            std::vector<std::vector<Kline>> watchDataGroup(symbols.size());
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
                std::ostringstream ss;
                ss << "realTimeUpdateWatchTask begin for collection: " + colName + "\n" << std::endl;
                std::cout << ss.str();
                auto realTimeUpdateWatchTask = boost::bind(&MongoManager::GetKlineUpdate,
                    &(this->dbManager),
                    dbName,
                    colName,
                    boost::ref(watchDataGroup[i]));

                TpInterval.Enqueue(realTimeUpdateWatchTask);

                i++;
            }
            TpInterval.WaitAll();

            // combine watchDataGroup element to targetData
            int startIndex = 0;
            int eraseOffset = 0;
            for (auto i = 0; i < symbols.size(); i++) {
                std::ostringstream ss;
                ss << colNameList[i - eraseOffset] + " watchDataGroup i: " << i << " size is: " << watchDataGroup[i].size() << "\n" << std::endl;
                std::cout << ss.str();

                if (watchDataGroup[i].size() == 0) {
                    colNameList.erase(colNameList.begin() + i - eraseOffset);
                    eraseOffset++;
                    continue;
                }
                targetData.insert(targetData.end(), watchDataGroup[i].begin(), watchDataGroup[i].end());
            
                int endIndex = targetData.size() - 1;
                dataIndexes.push_back(std::pair<int, int>(startIndex, endIndex));
                startIndex = targetData.size();
                std::ostringstream ss2;
                ss2 << "dataIndexes size is: " << dataIndexes.size() << "\n" << std::endl;
                std::cout << ss2.str();
            }

            // exec the calculation
            strategyInst->onMarketData(targetData, dataIndexes);

            // exec settlement
            this->runSettlement(targetData, colNameList, dataIndexes);

            // update Kline one by one with Bulk
            int colNameIndex = 0;
            std::ostringstream ss;
            ss << interval + " targetData size is: " << targetData.size() << "\n" << std::endl;
            std::cout << ss.str();
            
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
            std::cerr << interval + " TpInterval start up with error:" << std::endl;
        }
    };

    //void SendOrders();
    template <typename T> void runSettlement(std::vector<T>& targetData, std::vector<std::string>& colNameList, std::vector<std::pair<int, int>>& dataIndexes){
        int colNameIndex = 0;
        for (auto dIndex : dataIndexes) {
            std::ostringstream ss;
            ss << "RunSettlement, colName:" + colNameList[colNameIndex] << "\n" << std::endl;
            std::cout << ss.str();

            std::vector<T> unitData(targetData.begin() + dIndex.first, targetData.begin() + dIndex.second + 1);
            settleInstance.ExecSettlement(unitData, colNameList[colNameIndex]);
            colNameIndex++;
        }        
    };

private:
    MongoManager dbManager;
    SettlementModule settleInstance;
};