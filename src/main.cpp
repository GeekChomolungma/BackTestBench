// mongoandcuda.cpp : Defines the entry point for the application.
//

#include "config/config.h"
#include "db/mongoManager.h"
#include "threadPool/threadPool.h"

#include <stdio.h>
#include <bsoncxx/json.hpp>
#include <bsoncxx/builder/basic/document.hpp>
#include <bsoncxx/builder/basic/kvp.hpp>

#include <mongocxx/client.hpp>
#include <mongocxx/instance.hpp>
#include <mongocxx/uri.hpp>

#include "platform/platform.h"
#include "strategy/myStrategy.h"
//#include <chrono>
//#include <thread>

#ifdef _WIN32
    // Windows-specific includes and definitions
    // #include "matplotlibcpp.h"
    // namespace plt = matplotlibcpp;
#else
#include <cassert>
// Other non-Windows includes and definitions
#endif
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
    //BTP.dbManager.GetSynedFlag();

    int64_t startTime = 0; // not used yet
    int64_t endTime = 0; // not used yet

    // create a thread group and push task
    ThreadPool tp(6);
    for (auto interval : allIntervals) {
        MyStrategy* strategyInstance = new MyStrategy(startTime, endTime, "");
        auto backTestTask = boost::bind(&BacktestingPlatform::runStrategyTask<Kline>,
            &BTP,
            strategyInstance,
            1640966400000,
            1642694399000,//1641398399000,
            "marketInfo",
            allSymbols,
            interval);

        tp.enqueue(backTestTask);
    }

    return 0;
}
