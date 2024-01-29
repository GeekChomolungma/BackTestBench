#include "db/mongoManager.h"

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
        // assert(doc["_id"].type() == bsoncxx::type::k_oid);
        auto klineContent = doc["kline"];

        Kline klineInst;
        klineInst.StartTime = klineContent["starttime"].get_int64();
        klineInst.EndTime = klineContent["endtime"].get_int64();
        std::string symbolStr(klineContent["symbol"].get_string());
        strcpy(klineInst.Symbol, symbolStr.c_str());
        std::string intervalStr(klineContent["interval"].get_string());
        strcpy(klineInst.Interval, intervalStr.c_str());

        klineInst.Open = std::stod(std::string(klineContent["open"].get_string()));
        klineInst.Close = std::stod(std::string(klineContent["close"].get_string()));
        klineInst.High = std::stod(std::string(klineContent["high"].get_string()));
        klineInst.Low = std::stod(std::string(klineContent["low"].get_string()));
        klineInst.Volume = std::stod(std::string(klineContent["volume"].get_string()));
        klineInst.TradeNum = klineContent["tradenum"].get_int64();
        klineInst.IsFinal = klineContent["isfinal"].get_bool();
        klineInst.QuoteVolume = std::stod(std::string(klineContent["quotevolume"].get_string()));
        targetKlineList.push_back(klineInst);
    }
}