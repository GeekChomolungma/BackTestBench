﻿// mongoandcuda.cpp : Defines the entry point for the application.
//

#include "config/config.h"
#include "db/mongoManager.h"

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

#include "platform/platform.h"
#include "strategy/myStrategy.h"


int main()
{
    Config cfg("config.ini");
    const std::string uriCfg = cfg.getUri();

    MongoManager dbManager(uriCfg);
    dbManager.GetSynedFlag();

    BacktestingPlatform BTP;

    int64_t startTime = 86400;
    int64_t endTime = 864000;

    MyStrategy* strategyInstance = new MyStrategy(startTime, endTime, "ETHUSDT");
    Kline kData;
    BTP.runBacktest(strategyInstance, kData);

    return 0;
}
