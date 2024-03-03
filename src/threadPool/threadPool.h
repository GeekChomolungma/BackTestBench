#ifndef THREADPOOL_BACKTEST_H
#define THREADPOOL_BACKTEST_H

#include <boost/asio.hpp>
#include <boost/bind/bind.hpp>
#include <boost/thread/thread.hpp>
#include <functional>
#include <iostream>
#include <memory> // std::unique_ptr

class ThreadPool {
public:
    ThreadPool(size_t threads) : work(new boost::asio::io_service::work(ioService)), serviceThread(nullptr) {
        for (size_t i = 0; i < threads; ++i) {
            workerGroup.create_thread(boost::bind(&boost::asio::io_service::run, &ioService));
        }
    }

    ~ThreadPool() {
        work.reset();
        //ioService.stop();
        workerGroup.join_all();
    }

    template<class F>
    void enqueue(F job) {
        ioService.post(job);
    }

private:
    boost::asio::io_service ioService;
    std::unique_ptr<boost::asio::io_service::work> work;
    boost::thread_group workerGroup;
    boost::thread* serviceThread;
};

#endif // !THREADPOOL_BACKTEST_H