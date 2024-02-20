// mongoandcuda.cpp : Defines the entry point for the application.
//

#include "config/config.h"
#include "db/mongoManager.h"

#ifdef _WIN32
    // Windows-specific includes and definitions
    #include "matplotlibcpp.h"
    namespace plt = matplotlibcpp;
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

#include "platform/platform.h"
#include "strategy/myStrategy.h"
//


int main()
{
    #ifdef _WIN32
        // Windows-specific includes and definitions
        //plt::plot({ 1,3,2,4 });
        //plt::show();
    #else
    #endif

    Config cfg("config.ini");
    const std::string uriCfg = cfg.getUri();

    MongoManager dbManager(uriCfg);
    dbManager.GetSynedFlag();

    std::vector<Kline> targetKlines;
    dbManager.GetKline(1692670260000, 1692676319999, "marketInfo", "ETCUSDT", targetKlines);
    int i = 0;
    for (auto k : targetKlines) {
        std::cout << "targetKlines" << i << " th element, start time is: " << k.StartTime << " open is: " << k.Open 
            << " close is: " << k.Close << " high is: " << k.High << " low is: " << k.Low << std::endl;
        i++;
    }
    
    // back test platform
    BacktestingPlatform BTP;
    int64_t startTime = 86400;
    int64_t endTime = 864000;

    MyStrategy* strategyInstance = new MyStrategy(startTime, endTime, "ETHUSDT");
    BTP.runBacktest(strategyInstance, targetKlines);

    // update Kline
    dbManager.BulkWriteByIds("marketInfo", "ETCUSDT", targetKlines);

    return 0;
}
