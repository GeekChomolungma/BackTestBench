// mongoandcuda.cpp : Defines the entry point for the application.
//

#include "config/config.h"
#include "db/mongoManager.h"
#include "threadPool/threadPool.h"

#include <stdio.h>
#include <bsoncxx/json.hpp>
#include <bsoncxx/builder/basic/document.hpp>
#include <bsoncxx/builder/basic/kvp.hpp>

#include <iostream>
#include <chrono>
#include <iomanip>

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

#ifdef _DEBUG
// for memory leak check 
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#include <csignal>
int reportHook(int reportType, char* message, int* returnValue) {
    std::cout << message;
    std::ofstream logFile("leak_report.txt", std::ios_base::app);
    logFile << message;
    logFile.close();
    *returnValue = 0;
    return TRUE;
}

void signalHandler(int signum) {
    _CrtDumpMemoryLeaks(); // trigger dump
    exit(signum);
}

#else
// Release
#endif
#else
#include <cassert>
// Other non-Windows includes and definitions
#endif
//

void GenerateTestBench(BacktestingPlatform& BTP, 
    const std::vector<std::string> allIntervals, const std::vector<std::string> allSymbols, 
    int64_t startTime, int64_t endTime);

int main()
{
#ifdef _WIN32
#ifdef _DEBUG
_CrtSetReportHook(reportHook);
_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
signal(SIGINT, signalHandler);
#else
// Windows-specific includes and definitions
//plt::plot({ 1,3,2,4 });
//plt::show();
#endif
#else
#endif

    Config cfg("config.ini");
    const std::string uriCfg = cfg.getUri();
    const std::vector<std::string> allSymbols = cfg.getMarketSubInfo("marketsub.symbols");
    const std::vector<std::string> allIntervals = cfg.getMarketSubInfo("marketsub.intervals");

    // back test platform
    BacktestingPlatform BTP(uriCfg);
    //BTP.dbManager.GetSynedFlag();

    int64_t startTime = 1640966400000; // not used yet in MyStrategy
    int64_t endTime = 1642694399000; // not used yet in MyStrategy
    GenerateTestBench(BTP, allIntervals, allSymbols, startTime, endTime);

    MyStrategy* strategyInstance = new MyStrategy(startTime, endTime, "");
    ThreadPool tpRealTime(allIntervals.size());
    for (auto interval : allIntervals) {
        auto realTimetask = [&tpRealTime, &BTP, strategyInstance, interval, allSymbols]() {
            int64_t startTime = 1640966400000; // not used yet in MyStrategy
            int64_t endTime = 1642694399000; // not used yet in MyStrategy
            while (true) {
                std::ostringstream ss;
                auto now = std::chrono::system_clock::now();
                auto now_c = std::chrono::system_clock::to_time_t(now);
                auto now_tm = std::localtime(&now_c);
                ss << std::put_time(now_tm, "%Y-%m-%d %H:%M:%S");
                ss << " - New Real-Time Task: " + interval + " Started... \n" << std::endl;
                std::cout << ss.str();

                BTP.runStrategyRealTime(
                    strategyInstance,
                    1640966400000,
                    "marketInfo",
                    allSymbols,
                    interval);

                ss.str(""); // clean ss
                now = std::chrono::system_clock::now();
                now_c = std::chrono::system_clock::to_time_t(now);
                now_tm = std::localtime(&now_c);
                ss << std::put_time(now_tm, "%Y-%m-%d %H:%M:%S");
                ss << " - New Real-Time Task: " + interval + " Finshed! \n" << std::endl;
                std::cout << ss.str(); 
            }
        };
        tpRealTime.Enqueue(realTimetask);
    }

    return 0;
}

void GenerateTestBench(BacktestingPlatform& BTP, const std::vector<std::string> allIntervals, const std::vector<std::string> allSymbols, int64_t startTime, int64_t endTime) {
    // create a thread group and push task
    ThreadPool tpBackTest(allIntervals.size());
    for (auto interval : allIntervals) {
        MyStrategy* strategyInstance = new MyStrategy(startTime, endTime, "");
        auto backTestTask = boost::bind(&BacktestingPlatform::runStrategyTask<Kline>,
            &BTP,
            strategyInstance,
            "marketInfo",
            allSymbols,
            interval);

        tpBackTest.Enqueue(backTestTask);
    }
}