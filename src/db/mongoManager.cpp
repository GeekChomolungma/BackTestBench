#include "db/mongoManager.h"

using bsoncxx::to_json;
using bsoncxx::builder::basic::make_document;
using bsoncxx::builder::basic::kvp;

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