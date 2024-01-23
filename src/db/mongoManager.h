#include <vector>
#include <bsoncxx/json.hpp>
#include <bsoncxx/builder/basic/document.hpp>
#include <bsoncxx/builder/basic/kvp.hpp>
#include <mongocxx/client.hpp>
#include <mongocxx/instance.hpp>
#include <mongocxx/uri.hpp>


class MongoManager {
public:
    MongoManager(const std::string uriStr);
    
    void GetSynedFlag();

    template <typename T>
    std::vector<T> Get(int64_t startTime, int64_t endTime, std::string eventName, std::string symbol) {
       
    }

    template <typename T>
    std::vector<T> Set() {};
private:
    std::string uriStr;
    mongocxx::instance inst;
    mongocxx::client mongoClient;
};
