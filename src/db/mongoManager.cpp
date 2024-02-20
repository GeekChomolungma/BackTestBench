#include "db/mongoManager.h"
#include <iomanip> // std::setw, std::setfill
#include <sstream>

using bsoncxx::builder::basic::kvp;
using bsoncxx::builder::basic::make_document;

MongoManager::MongoManager(const std::string uriStr) {
    this->uriStr = uriStr;

    const auto uri = mongocxx::uri{ this->uriStr.c_str() };
    // Set the version of the Stable API on the client.
    mongocxx::options::client client_options;
    const auto api = mongocxx::options::server_api{ mongocxx::options::server_api::version::k_version_1 };
    client_options.server_api_opts(api);

    // Setup the connection and get a handle on the "admin" database.
    this->mongoClient = mongocxx::client{ uri, client_options };
};

void MongoManager::GetSynedFlag() {
    auto mongoDB = this->mongoClient["marketSyncFlag"];
    // Ping the database.
    const auto ping_cmd = make_document(kvp("ping", 1));
    mongoDB.run_command(ping_cmd.view());
    std::cout << "Pinged your deployment. You successfully connected to MongoDB!" << std::endl;

    try
    {
        auto col = mongoDB["ETCUSDT"];
        auto find_one_result = col.find_one({});
        if (find_one_result) {
            auto extractedValue = *find_one_result;
            auto eViewElement = extractedValue["starttime"];
            auto st = eViewElement.get_int64();
            std::cout << "Got synced flag time:" << st << std::endl;
        }
    }
    catch (const std::exception& e)
    {
        // Handle errors.
        std::cout << "Exception: " << e.what() << std::endl;
    }
}

void MongoManager::GetKline(int64_t startTime, int64_t endTime, std::string dbName, std::string colName, std::vector<Kline>& targetKlineList) {
    auto db = this->mongoClient[dbName.c_str()];
    auto col = db[colName.c_str()];
    auto cursor_filtered = 
        col.find({ make_document(kvp("kline.starttime", make_document(kvp("$gte", startTime), kvp("$lt", endTime)))) });

    for (auto doc : cursor_filtered) {        
        Kline klineInst;

        const auto& oidBytesVec = doc["_id"].get_oid().value.bytes();
        for (size_t i = 0; i < 12; ++i) {
            klineInst.Id[i] = oidBytesVec[i];
        }

        std::cout << "OID in hex: ";
        for (int i = 0; i < 12; ++i) {
            std::cout << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(klineInst.Id[i]);
        }
        std::cout << std::endl;

        auto klineContent = doc["kline"];
        klineInst.StartTime = klineContent["starttime"].get_int64().value;
        klineInst.EndTime = klineContent["endtime"].get_int64().value;
        bsoncxx::stdx::string_view symbolTmp = klineContent["symbol"].get_string().value;
        std::string symbolStr(symbolTmp);

        strcpy(klineInst.Symbol, symbolStr.c_str());

        bsoncxx::stdx::string_view intervalTmp = klineContent["interval"].get_string().value;
        std::string intervalStr(intervalTmp);
        strcpy(klineInst.Interval, intervalStr.c_str());

        bsoncxx::stdx::string_view openStrTmp = klineContent["open"].get_string().value;
        klineInst.Open = std::stod(std::string(openStrTmp));

        bsoncxx::stdx::string_view closeStrTmp = klineContent["close"].get_string().value;
        klineInst.Close = std::stod(std::string(closeStrTmp));

        bsoncxx::stdx::string_view highStrTmp = klineContent["high"].get_string().value;
        klineInst.High = std::stod(std::string(highStrTmp));

        bsoncxx::stdx::string_view lowStrTmp = klineContent["low"].get_string().value;
        klineInst.Low = std::stod(std::string(lowStrTmp));

        auto trElement = klineContent["truerange"];
        if (trElement && trElement.type() == bsoncxx::type::k_double) {
            klineInst.TrueRange = trElement.get_double().value;
        }
        else {
            klineInst.TrueRange = 0.0;
        }
       
        auto atrElement = klineContent["avetruerange"];
        if (atrElement && atrElement.type() == bsoncxx::type::k_double) {
            klineInst.AveTrueRange = atrElement.get_double().value;
        }
        else {
            klineInst.AveTrueRange = 0.0;
        }

        bsoncxx::stdx::string_view volStrTmp = klineContent["volume"].get_string().value;
        klineInst.Volume = std::stod(std::string(volStrTmp));

        klineInst.TradeNum = klineContent["tradenum"].get_int64();
        klineInst.IsFinal = klineContent["isfinal"].get_bool();

        bsoncxx::stdx::string_view qutovStrTmp = klineContent["quotevolume"].get_string().value;
        klineInst.QuoteVolume = std::stod(std::string(qutovStrTmp));

        targetKlineList.push_back(klineInst);
    }
}

void MongoManager::BulkWriteByIds(std::string dbName, std::string colName, std::vector<Kline>& rawData) {
    // locate the coll
    auto db = this->mongoClient[dbName.c_str()];
    auto col = db[colName.c_str()];
    

    // create bulk
    auto bulk = col.create_bulk_write();
    for (auto kline : rawData) {
        bsoncxx::builder::basic::document filter_builder, update_builder;
        bsoncxx::oid docID(&kline.Id[0], 12);

        // format tr and atr to string
        std::ostringstream oss1, oss2;
        oss1 << std::fixed << std::setprecision(6) << kline.TrueRange;
        oss2 << std::fixed << std::setprecision(6) << kline.AveTrueRange;
        std::string trStr = oss1.str();
        std::string atrStr = oss2.str();
    
        // create filter and update
        filter_builder.append(kvp("_id", docID));
        update_builder.append(kvp("$set", 
            make_document(
                kvp("truerange", trStr),
                kvp("avetruerange", atrStr)
            )
        ));
        mongocxx::model::update_one upsert_op(filter_builder.view(), update_builder.view());
        upsert_op.upsert(true);

        bulk.append(upsert_op);
    }
    auto result = bulk.execute();

    if (!result) {
        std::cout << "create_bulk_write failed!!!" << std::endl;
    }
}