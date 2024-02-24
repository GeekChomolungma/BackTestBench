// mongoandcuda.cpp : Defines the entry point for the application.
//

#include "config/config.h"
#include "db/mongoManager.h"
#include "threadPool/threadPool.h"

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
    const std::vector<std::string> allSymbols = cfg.getMarketSubInfo("marketsub.symbols");
    const std::vector<std::string> allIntervals = cfg.getMarketSubInfo("marketsub.intervals");

    // back test platform
    BacktestingPlatform BTP(uriCfg);
    int64_t startTime = 86400;
    int64_t endTime = 864000;

    //MongoManager dbManager(uriCfg);
    BTP.dbManager.GetSynedFlag();
    MyStrategy* strategyInstance = new MyStrategy(startTime, endTime, "ETCUSDT");
    auto backTestTask = boost::bind(&BacktestingPlatform::runStrategyTask<Kline>,
        &BTP, 
        strategyInstance, 
        1692670260000, 
        1692676319999, 
        "marketInfo",
        allSymbols,
        "15m");

    ThreadPool tp(6);
    tp.enqueue(backTestTask);

    boost::this_thread::sleep_for(boost::chrono::seconds(20));

    return 0;
}
