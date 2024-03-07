#include "threadPool/threadPool.h"

void ThreadPool::WaitAll() {
    this->work.reset();
    this->workerGroup.join_all();
}