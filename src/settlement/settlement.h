#ifndef SETTLEMENTMODULE_H
#define SETTLEMENTMODULE_H

#include <vector>
#include <map>
#include <string>
#include <sstream>
#include <iostream>
#include <boost/thread/shared_mutex.hpp>
#include <boost/thread/locks.hpp>
#include "dtos/settlementItem.h"
#include "db/mongoManager.h"

#include "dtos/kline.h"

class SettlementModule
{
public:
	SettlementModule(MongoManager& dbManager) : dbM(dbManager), dbName("settlement") {}
	
	void SetPrivousSettleItem(std::string& keySymbol, SettlementItem item);
	bool GetPrivousSettleItem(std::string& keySymbol, SettlementItem& item);

	template <typename T> void ExecSettlement(std::vector<T> oneCoinData, std::string colName) {
        auto klineDataList = static_cast<std::vector<Kline>>(oneCoinData);
        for (auto kline : klineDataList) {
            if (kline.Action != 0) {
                // read previous item from cache
                SettlementItem prevItem;
                auto prevItemExist = this->GetPrivousSettleItem(colName, prevItem);
                if ((prevItemExist) && (prevItem.StartTime == kline.StartTime)){
                    std::ostringstream ss;
                    ss << "ExecSettlement, Same kline in " + colName + " with the prev action Item, start time: " << kline.StartTime << "\n" << std::endl;
                    std::cout << ss.str();
                    continue;
                }
                SettlementItem currItem;
                currItem.StartTime = kline.StartTime;
                currItem.EndTime = kline.EndTime;
                for (auto i = 0; i < 16; i++) {
                    currItem.Symbol[i] = kline.Symbol[i];
                    currItem.Interval[i] = kline.Interval[i];
                }
                currItem.Action = kline.Action;
                currItem.ExecPrice = kline.Close;

                // create the an item when cache is empty
                if (!prevItemExist) {
                    if (currItem.Action == 1) {
                        // the first trade should not be "1.sell"
                        continue;
                    }
                    // create a item and insert into DB
                    currItem.PreviousId = "";
                    currItem.ExecVolume = 10000.0 / currItem.ExecPrice;
                    currItem.SumValue = 10000.0;
                    currItem.SumAmout = currItem.ExecVolume;
                    currItem.ProfitValue = 0.0;
                    
                    // insert into DB
                    auto insertedID = this->dbM.SetSettlementItems(this->dbName, colName, currItem);
                    if (insertedID.empty()) {
                        std::cout << "insertedID is empty" << std::endl;
                    }
                    else {
                        // update cache
                        currItem.Id = insertedID;
                        std::cout << "Insert settlement items, col: " + colName + " insertedID:" + insertedID << std::endl;
                        this->SetPrivousSettleItem(colName, currItem);
                    }
                }
                else {
                    if (currItem.Action != prevItem.Action) {
                        currItem.PreviousId = prevItem.Id;
                        if (currItem.Action == 1) {
                            // sell
                            currItem.ExecVolume = prevItem.SumAmout;
                            currItem.SumValue = prevItem.SumAmout * currItem.ExecPrice;
                            currItem.SumAmout = 0.0;
                            currItem.ProfitValue = (currItem.SumValue - 10000.0) / 10000.0;
                        }
                        else {
                            // buy
                            currItem.ExecVolume = prevItem.SumValue / currItem.ExecPrice;
                            currItem.SumValue = 0.0;
                            currItem.SumAmout = currItem.ExecVolume;
                            currItem.ProfitValue = 0.0;
                        }
                        // insert into DB
                        auto insertedID = this->dbM.SetSettlementItems(this->dbName, colName, currItem);
                        if (insertedID.empty()) {
                            std::cout << "insertedID is empty" << std::endl;
                        }
                        else {
                            // update cache
                            currItem.Id = insertedID;
                            this->SetPrivousSettleItem(colName, currItem);
                        }
                    }
                    else {
                        std::cout << "same action between currItem and prevItem" << std::endl;
                    }
                }
            }
        }
    };
	//bool GetCurrentSettleItemFromDB();

private:
	mutable boost::shared_mutex mutex;
	std::map<std::string, SettlementItem> previousSettleMap;
	MongoManager& dbM;
    std::string dbName;
};

#endif
