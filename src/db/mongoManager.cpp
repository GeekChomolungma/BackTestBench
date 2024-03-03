#include "db/mongoManager.h"
#include <iomanip> // std::setw, std::setfill
#include <sstream>

using bsoncxx::builder::basic::kvp;
using bsoncxx::builder::basic::make_document;

MongoManager::MongoManager(const std::string uriStr):uriStr(uriStr), mongoPool(mongocxx::uri{ this->uriStr.c_str() }){
    // this->uriStr = uriStr;
    // const auto uri = mongocxx::uri{ this->uriStr.c_str() };
    // Set the version of the Stable API on the client.
    // mongocxx::options::client client_options;
    // const auto api = mongocxx::options::server_api{ mongocxx::options::server_api::version::k_version_1 };
    // client_options.server_api_opts(api);
    // Setup the connection and get a handle on the "admin" database.
    // this->mongoClient = mongocxx::client{ uri, client_options };
    // this->mongoPool = mongocxx::pool{ uri };
};

void MongoManager::GetSynedFlag() {
    auto client = this->mongoPool.acquire();
    auto mongoDB = (*client)["marketSyncFlag"];

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
    auto client = this->mongoPool.acquire();
    auto db = (*client)[dbName.c_str()];
    auto col = db[colName.c_str()];
    auto cursor_filtered = 
        col.find({ make_document(kvp("kline.starttime", make_document(kvp("$gte", startTime), kvp("$lt", endTime)))) });

    for (auto doc : cursor_filtered) {        
        Kline klineInst;

        const auto& oidBytesVec = doc["_id"].get_oid().value.bytes();
        for (size_t i = 0; i < 12; ++i) {
            klineInst.Id[i] = oidBytesVec[i];
        }

        // std::cout << "OID in hex: ";
        //for (int i = 0; i < 12; ++i) {
        //    std::cout << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(klineInst.Id[i]);
        //}
        //std::cout << std::dec << std::endl;

        auto klineContent = doc["kline"];
        klineInst.StartTime = klineContent["starttime"].get_int64().value;
        klineInst.EndTime = klineContent["endtime"].get_int64().value;
        bsoncxx::stdx::string_view symbolTmp = klineContent["symbol"].get_string().value;
        std::string symbolStr(symbolTmp);
        bsoncxx::stdx::string_view intervalTmp = klineContent["interval"].get_string().value;
        std::string intervalStr(intervalTmp);

        #ifdef _WIN32
            strcpy_s(klineInst.Symbol, symbolStr.c_str());
            strcpy_s(klineInst.Interval, intervalStr.c_str());
        #else
            strcpy(klineInst.Symbol, symbolStr.c_str());
            strcpy(klineInst.Interval, intervalStr.c_str());
        #endif

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
            if (trElement && trElement.type() == bsoncxx::type::k_string) {
                bsoncxx::stdx::string_view trTmp = trElement.get_string().value;
                klineInst.TrueRange = std::stod(std::string(trTmp));
            }
            else {
                klineInst.TrueRange = 0.0;
            }
        }
       
        auto atrElement = klineContent["avetruerange"];
        if (atrElement && atrElement.type() == bsoncxx::type::k_double) {
            klineInst.AveTrueRange = atrElement.get_double().value;
        }
        else {
            if (atrElement && atrElement.type() == bsoncxx::type::k_string) {
                bsoncxx::stdx::string_view atrTmp = atrElement.get_string().value;
                klineInst.AveTrueRange = std::stod(std::string(atrTmp));
            }
            else {
                klineInst.AveTrueRange = 0.0;
            }
        }

        auto stElement = klineContent["supertrendvalue"];
        if (stElement && stElement.type() == bsoncxx::type::k_string) {
            bsoncxx::stdx::string_view stTmp = stElement.get_string().value;
            klineInst.SuperTrendValue = std::stod(std::string(stTmp));
        }
        else {
            klineInst.SuperTrendValue = 0.0;
        }

        auto stupElement = klineContent["stup"];
        if (stupElement && stupElement.type() == bsoncxx::type::k_string) {
            bsoncxx::stdx::string_view stupTmp = stupElement.get_string().value;
            klineInst.StUp = std::stod(std::string(stupTmp));
        }
        else {
            klineInst.StUp = 0.0;
        }

        auto stdownElement = klineContent["stdown"];
        if (stdownElement && stdownElement.type() == bsoncxx::type::k_string) {
            bsoncxx::stdx::string_view stdownTmp = stdownElement.get_string().value;
            klineInst.StDown = std::stod(std::string(stdownTmp));
        }
        else {
            klineInst.StDown = 0.0;
        }

        auto dirElement = klineContent["stdirection"];
        if (dirElement && dirElement.type() == bsoncxx::type::k_int32) {
            klineInst.STDirection = dirElement.get_int32().value;
        }
        else {
            klineInst.STDirection = 0;
        }

        auto actElement = klineContent["action"];
        if (actElement && actElement.type() == bsoncxx::type::k_int32) {
            klineInst.Action = actElement.get_int32().value;
        }
        else {
            klineInst.Action = 0;
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
    try{
        // locate the coll
        auto client = this->mongoPool.acquire();
        auto db = (*client)[dbName.c_str()];
        auto col = db[colName.c_str()];

        // create bulk
        auto bulk = col.create_bulk_write();
        for (auto kline : rawData) {
            bsoncxx::builder::basic::document filter_builder, update_builder;
            bsoncxx::oid docID(&kline.Id[0], 12);

            // format tr and atr to string
            std::ostringstream oss1, oss2, oss3, oss4, oss5;
            oss1 << std::fixed << std::setprecision(6) << kline.TrueRange;
            oss2 << std::fixed << std::setprecision(6) << kline.AveTrueRange;
            oss3 << std::fixed << std::setprecision(6) << kline.SuperTrendValue;
            oss4 << std::fixed << std::setprecision(6) << kline.StUp;
            oss5 << std::fixed << std::setprecision(6) << kline.StDown;
            std::string trStr = oss1.str();
            std::string atrStr = oss2.str();
            std::string stStr = oss3.str();
            std::string stUpStr = oss4.str();
            std::string stDownStr = oss5.str();

            // create filter and update
            filter_builder.append(kvp("_id", docID));
            update_builder.append(kvp("$set",
                make_document(
                    kvp("truerange", trStr),
                    kvp("avetruerange", atrStr),
                    kvp("supertrendvalue", stStr),
                    kvp("stup", stUpStr),
                    kvp("stdown", stDownStr),
                    kvp("stdirection", kline.STDirection),
                    kvp("action", kline.Action)
                )
            ));
            mongocxx::model::update_one upsert_op(filter_builder.view(), update_builder.view());
            upsert_op.upsert(true);

            bulk.append(upsert_op);
        }
        auto result = bulk.execute();

        if (!result) {
            std::cout << "create_bulk_write failed!!!\n" << std::endl;
        }
    }
    catch (const mongocxx::exception& e) {
        std::cout << "BulkWriteByIds, An exception occurred: " << e.what() << std::endl;
    }
    catch (const std::runtime_error& e) {
        std::cerr << "runtime error: " << e.what() << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "error exception: " << e.what() << std::endl;
    }
    catch (...) {
        std::cerr << "unknown error!" << std::endl;
    }
}

std::string MongoManager::SetSettlementItems(std::string dbName, std::string colName, SettlementItem& data) {
    // locate the coll
    auto client = this->mongoPool.acquire();
    auto db = (*client)[dbName.c_str()];
    auto col = db[colName.c_str()];
    std::cout << "SetSettlementItems: " + colName + "\n" << std::endl;

    std::string symbol = data.Symbol;
    std::string interval = data.Interval;

    std::ostringstream oss1, oss2, oss3, oss4, oss5;
    oss1 << std::fixed << std::setprecision(6) << data.ExecPrice;
    oss2 << std::fixed << std::setprecision(6) << data.ExecVolume;
    oss3 << std::fixed << std::setprecision(6) << data.ProfitValue;
    oss4 << std::fixed << std::setprecision(6) << data.SumValue;
    oss5 << std::fixed << std::setprecision(6) << data.SumAmout;
    std::string ExecPriceStr = oss1.str();
    std::string ExecVolumeStr = oss2.str();
    std::string ProfitValueStr = oss3.str();
    std::string SumValueStr = oss4.str();
    std::string SumAmoutStr = oss5.str();
    std::cout << "ready to builder::basic: " + colName + "\n" << std::endl;

    auto docValueBuilder = bsoncxx::builder::basic::document{};
    docValueBuilder.append(
        kvp("start_time", data.StartTime),
        kvp("end_time", data.EndTime),
        kvp("symbol", symbol),
        kvp("interval", interval),
        kvp("action", data.Action),
        kvp("tradeID", data.TradeID),
        kvp("execTime", data.ExecTime),
        kvp("exec_price", ExecPriceStr),
        kvp("exec_volume", ExecVolumeStr),
        kvp("profit_value", ProfitValueStr),
        kvp("sum_value", SumValueStr),
        kvp("sum_amout", SumAmoutStr));

    if (!data.PreviousId.empty()) {
        bsoncxx::oid prevID(data.PreviousId);
        docValueBuilder.append(kvp("prevId", prevID));
    }
    bsoncxx::document::value InsertedDoc = docValueBuilder.extract();

    // We choose to move in our document here, which transfers ownership to insert_one()
    try {
        std::cout << "ready to insert_one: " + colName + "\n" << std::endl;
        auto res = col.insert_one(std::move(InsertedDoc));

        if (!res) {
            std::cout << "Unacknowledged write. No id available." << std::endl;
            return "";
        }

        if (res->inserted_id().type() == bsoncxx::type::k_oid) {
            bsoncxx::oid id = res->inserted_id().get_oid().value;
            std::string id_str = id.to_string();
            std::cout << "Inserted id: " << id_str << std::endl;
            return id_str;
        }
        else {
            std::cout << "Inserted id was not an OID type" << std::endl;
            return "";
        }
    }
    catch (const mongocxx::exception& e) {
        std::cout << "An exception occurred: " << e.what() << std::endl;
        return "";
    }
}