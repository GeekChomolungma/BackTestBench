// mongoandcuda.cpp : Defines the entry point for the application.
//

#include "config/config.h"

#ifdef _WIN32
    // Windows-specific includes and definitions
#else
    #include <cassert>
    // Other non-Windows includes and definitions
#endif

#include <stdio.h>
#include <bsoncxx/json.hpp>
#include <bsoncxx/builder/basic/document.hpp>
#include <bsoncxx/builder/basic/kvp.hpp>

#include <mongocxx/client.hpp>
#include <mongocxx/instance.hpp>
#include <mongocxx/uri.hpp>


using bsoncxx::to_json;
using namespace mongocxx;
using bsoncxx::builder::basic::make_document;
using bsoncxx::builder::basic::kvp;

// void kernel_wrapper(int argc,const char *argv[]);

int main()
{
    Config cfg("config.ini");
    const std::string uriCfg = cfg.getUri();

    try
    {
        // Create an instance.
        mongocxx::instance inst{};

        // Replace the connection string with your MongoDB deployment's connection string.
        const auto uri = mongocxx::uri{uriCfg.c_str()};

        // Set the version of the Stable API on the client.
        mongocxx::options::client client_options;
        const auto api = mongocxx::options::server_api{ mongocxx::options::server_api::version::k_version_1 };
        client_options.server_api_opts(api);

        // Setup the connection and get a handle on the "admin" database.
        mongocxx::client conn{ uri, client_options };
        mongocxx::database db = conn["marketSyncFlag"];

        // Ping the database.
        const auto ping_cmd = make_document(kvp("ping", 1));
        db.run_command(ping_cmd.view());
        std::cout << "Pinged your deployment. You successfully connected to MongoDB!" << std::endl;

        auto col = db["ETCUSDT"];
        auto find_one_result = col.find_one({});
        if (find_one_result) {
            auto extractedValue = *find_one_result;
            auto eViewElement = extractedValue["starttime"];
            auto st = eViewElement.get_int64();
            std::cout << "Got synced flag time:"<< st << std::endl;
        }
        assert(find_one_result);
    }
    catch (const std::exception& e) 
    {
        // Handle errors.
        std::cout<< "Exception: " << e.what() << std::endl;
    }
    
    // for cuda process
    int argc = 0;
    const char* argv [1]= {"Burning GPU!!!"};
    // kernel_wrapper(0, argv);
    
    return 0;
}
