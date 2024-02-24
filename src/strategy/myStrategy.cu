// includes, system
#include <stdio.h>
#include <vector>
#include <utility> // for std::pair
#include <dtos/kline.h>

// includes CUDA Runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// indicator lib
#include "indicator/ta.cuh"

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h>  // helper utility functions

void kernel_wrapper(int argc, const char* argv[], std::vector<Kline>& rawData, std::vector<std::pair<int, int>>& dataIndexes) {
    int devID;
    cudaDeviceProp deviceProps;

    printf("[%s] - Starting...\n", argv[0]);
    printf("kline size is %zd\n", rawData.size());
    // This will pick the best possible CUDA capable device
    devID = findCudaDevice(argc, (const char**)argv);

    // get device name
    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
    printf("CUDA device [%s]\n", deviceProps.name);

    // allocate host memeory
    std::vector<Kline> hostSrc = rawData;
    size_t n = rawData.size();
    size_t nbytes = rawData.size() * sizeof(Kline);
    printf("rawData size is %zd, nbytes is %zd \n", n, nbytes);

    // calculate stock number and 
    // get start and end index of each stock
    size_t stockNumber = dataIndexes.size();
    int* startIndexes = new int[stockNumber];
    int* endIndexes = new int[stockNumber];
    for (auto i = 0; i < stockNumber; i++) {
        startIndexes[i] = dataIndexes[i].first;
        endIndexes[i] = dataIndexes[i].second;
    }

    // allocate device memory
    Kline* deviceRaw = 0;
    float* deviceEma = 0;
    int* deviceStartInd = 0;
    int* deviceEndInd = 0;
    checkCudaErrors(cudaMalloc((void**)&deviceRaw, nbytes));
    checkCudaErrors(cudaMalloc((void**)&deviceEma, n*sizeof(float)));
    checkCudaErrors(cudaMemset(deviceRaw, 255, nbytes));
    checkCudaErrors(cudaMemset(deviceRaw, 0, n * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&deviceStartInd, stockNumber*sizeof(int)));
    checkCudaErrors(cudaMemset(deviceStartInd, 0, stockNumber * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&deviceEndInd, stockNumber * sizeof(int)));
    checkCudaErrors(cudaMemset(deviceEndInd, 0, stockNumber * sizeof(int)));

    // set kernel launch configuration
    dim3 threads = dim3(32, 1);
    dim3 blocks = dim3(n / threads.x + 1, 1);

    // create cuda event handles
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    StopWatchInterface* timer = NULL;
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);

    checkCudaErrors(cudaDeviceSynchronize());
    float gpu_time = 0.0f;

    // new a stream for this task
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // asynchronously issue work to the GPU (all to stream 0)
    checkCudaErrors(cudaProfilerStart());
    sdkStartTimer(&timer);
    cudaEventRecord(start, stream);
    cudaMemcpyAsync(deviceRaw, rawData.data(), nbytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(deviceStartInd, startIndexes, stockNumber * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(deviceEndInd, endIndexes, stockNumber * sizeof(int), cudaMemcpyHostToDevice, stream);
    test_kernel <<<blocks, threads, 0, stream >>> (deviceRaw, deviceEma, stockNumber, deviceStartInd, deviceEndInd, 10, 0.2);
    cudaMemcpyAsync(rawData.data(), deviceRaw, nbytes, cudaMemcpyDeviceToHost, stream);
    std::vector<float> hostEma(n);
    cudaMemcpyAsync(hostEma.data(), deviceEma, n * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaEventRecord(stop, stream);
    sdkStopTimer(&timer);
    checkCudaErrors(cudaProfilerStop());

    //// have CPU do some work while waiting for stage 1 to finish
    //unsigned long int counter = 0;

    //while (cudaEventQuery(stop) == cudaErrorNotReady) {
    //    counter++;
    //}

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));

    // print the cpu and gpu times
    printf("time spent executing by the GPU: %.2f ms\n", gpu_time);
    printf("time spent by CPU in CUDA calls: %.2f ms\n", sdkGetTimerValue(&timer));
    printf("EMA in host length is: %zd, the 0th EMA is: %f, the 1th EMA is: %f \n", hostEma.size(), hostEma[0], hostEma[1]);

    // release resources
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    //checkCudaErrors(cudaFreeHost(hostSrc));
    checkCudaErrors(cudaFree(deviceRaw));
    checkCudaErrors(cudaFree(deviceEma));
    checkCudaErrors(cudaFree(deviceStartInd));
    checkCudaErrors(cudaFree(deviceEndInd));
    cudaStreamDestroy(stream);
    delete[] startIndexes;
    delete[] endIndexes;

    return;
   /* exit(bFinalResults ? EXIT_SUCCESS : EXIT_FAILURE);*/
}