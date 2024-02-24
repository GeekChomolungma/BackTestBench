#include <boost/asio.hpp>
#include <boost/bind/bind.hpp>
#include <boost/thread/thread.hpp>
#include <functional>
#include <iostream>

class ThreadPool {
public:
    ThreadPool(size_t threads) : work(ioService), serviceThread(nullptr) {
        for (size_t i = 0; i < threads; ++i) {
            workerGroup.create_thread(boost::bind(&boost::asio::io_service::run, &ioService));
        }
    }

    ~ThreadPool() {
        ioService.stop();
        workerGroup.join_all();
    }

    template<class F>
    void enqueue(F job) {
        ioService.post(job);
    }

private:
    boost::asio::io_service ioService;
    boost::asio::io_service::work work;
    boost::thread_group workerGroup;
    boost::thread* serviceThread;
};