#ifndef MONGOMANAGER_H
#define MONGOMANAGER_H

#include <vector>
#include <bsoncxx/json.hpp>
#include <bsoncxx/builder/basic/document.hpp>
#include <bsoncxx/builder/basic/kvp.hpp>
#include <mongocxx/client.hpp>
#include <mongocxx/instance.hpp>
#include <mongocxx/uri.hpp>
#include <mongocxx/pool.hpp>
#include <mongocxx/change_stream.hpp>
#include <mongocxx/options/find.hpp>
#include <mongocxx/exception/exception.hpp>

#include "dtos/kline.h"
#include "dtos/settlementItem.h"

using bsoncxx::to_json;
using bsoncxx::builder::basic::make_document;
using bsoncxx::builder::basic::kvp;

class MongoManager {
public:
    MongoManager(std::string uriStr);
    
    void GetSynedFlag();
    
    void ParseKline(const bsoncxx::v_noabi::document::view& doc, Kline& klineInst);

    void GetKline(int64_t startTime, int64_t endTime, std::string dbName, std::string colName, std::vector<Kline>& targetKlineList);

    void BulkWriteByIds(std::string dbName, std::string colName, std::vector<Kline>& rawData);

    std::string SetSettlementItems(std::string dbName, std::string colName,SettlementItem& data);

    void WatchKlineUpdate(std::string dbName, std::string colName, std::vector<Kline>& PreviousTwoKlines);

private:
    std::string uriStr;
    mongocxx::instance inst;
    //mongocxx::client mongoClient;
    mongocxx::pool mongoPool;
};

#endif