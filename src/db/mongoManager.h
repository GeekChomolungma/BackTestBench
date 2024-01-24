#include <vector>
#include <bsoncxx/json.hpp>
#include <bsoncxx/builder/basic/document.hpp>
#include <bsoncxx/builder/basic/kvp.hpp>
#include <mongocxx/client.hpp>
#include <mongocxx/instance.hpp>
#include <mongocxx/uri.hpp>

#include <dtos/kline.h>


using bsoncxx::to_json;
using bsoncxx::builder::basic::make_document;
using bsoncxx::builder::basic::kvp;

class MongoManager {
public:
    MongoManager(const std::string uriStr);
    
    void GetSynedFlag();

    void GetKline(int64_t startTime, int64_t endTime, std::string dbName, std::string colName, std::vector<Kline>& targetKlineList);

private:
    std::string uriStr;
    mongocxx::instance inst;
    mongocxx::client mongoClient;
};
