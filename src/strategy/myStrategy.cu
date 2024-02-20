// includes, system
#include <stdio.h>
#include <vector>
#include <dtos/kline.h>

// includes CUDA Runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// indicator lib
#include "indicator/ta.cuh"

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h>  // helper utility functions

void kernel_wrapper(int argc, const char* argv[], std::vector<Kline>& rawData) {
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
    printf("rawData size is %d, nbytes is %d \n", n, nbytes);

    // allocate device memory
    Kline* deviceRaw = 0;
    float* deviceEma = 0;
    checkCudaErrors(cudaMalloc((void**)&deviceRaw, nbytes));
    checkCudaErrors(cudaMalloc((void**)&deviceEma, n*sizeof(float)));
    checkCudaErrors(cudaMemset(deviceRaw, 255, nbytes));
    checkCudaErrors(cudaMemset(deviceRaw, 0, n * sizeof(float)));

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

    // asynchronously issue work to the GPU (all to stream 0)
    checkCudaErrors(cudaProfilerStart());
    sdkStartTimer(&timer);
    cudaEventRecord(start, 0);
    cudaMemcpyAsync(deviceRaw, rawData.data(), nbytes, cudaMemcpyHostToDevice, 0);
    test_kernel <<<blocks, threads, 0, 0 >>> (deviceRaw, deviceEma,1, n, 10, 0.2);
    cudaMemcpyAsync(rawData.data(), deviceRaw, nbytes, cudaMemcpyDeviceToHost, 0);
    std::vector<float> hostEma(n);
    cudaMemcpyAsync(hostEma.data(), deviceEma, n * sizeof(float), cudaMemcpyDeviceToHost, 0);
    cudaEventRecord(stop, 0);
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
    printf("EMA in host length is: %d, the 0th EMA is: %f, the 1th EMA is: %f \n", hostEma.size(), hostEma[0], hostEma[1]);

    // release resources
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    //checkCudaErrors(cudaFreeHost(hostSrc));
    checkCudaErrors(cudaFree(deviceRaw));
    checkCudaErrors(cudaFree(deviceEma));

    return;
   /* exit(bFinalResults ? EXIT_SUCCESS : EXIT_FAILURE);*/
}