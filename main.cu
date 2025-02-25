#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <fstream>
#include <chrono>

#include "main.h"
#include "cuda_macros.h"
#include <objects.h>

cudaError_t execute(numb* data, numb* maps, int rangingCount, int variationSize, int variations, unsigned long int mapsSize, numb* previousData)
{
    unsigned long int size = variationSize * variations;

    //std::chrono::steady_clock::time_point before = std::chrono::steady_clock::now();
    //std::chrono::steady_clock::time_point precompute, incompute, postcompute;

    cudaError_t cudaStatus;

    int blocks = (int)ceil((float)variations / THREADS_PER_BLOCK);
    int threads = THREADS_PER_BLOCK;

    PreRanging ranging(kernel::VAR_COUNT, kernel::PARAM_COUNT);
    ranging.setRangingAndVariations(rangingCount, variations);
    int rangingIndex = 0;

    numb* cuda_data = 0;
    numb* cuda_params = 0;
    numb* cuda_maps = 0;
    PreRanging* cuda_ranging = 0;
    numb* cuda_prev_data = 0;
    MapData* cuda_map_data = 0;

    CUDA_SET_DEVICE;

    CUDA_MALLOC((void**)&cuda_data, size * sizeof(numb), "cudaMalloc data failed!");
    CUDA_MALLOC((void**)&cuda_prev_data, size * sizeof(numb), "cudaMalloc prev data failed!");
    CUDA_MALLOC((void**)&cuda_params, kernel::PARAM_COUNT * sizeof(numb), "cudaMalloc params failed!");
    CUDA_MALLOC((void**)&cuda_maps, mapsSize * sizeof(numb), "cudaMalloc maps failed!");
    CUDA_MALLOC((void**)&cuda_map_data, kernel::MAP_COUNT * sizeof(MapData), "cudaMalloc map data failed!");

    // Parameters array structure (PreRanging):
    for (int i = 0; i < kernel::VAR_COUNT; i++)
        if (kernel::VAR_RANGING[i])
            ranging.rangings[rangingIndex++].init(i, kernel::VAR_VALUES[i], kernel::VAR_STEPS[i], kernel::VAR_MAX[i], kernel::VAR_STEP_COUNTS[i]);
    for (int i = 0; i < kernel::PARAM_COUNT; i++)
        if (kernel::PARAM_RANGING[i])
            ranging.rangings[rangingIndex++].init(i + kernel::VAR_COUNT, kernel::PARAM_VALUES[i], kernel::PARAM_STEPS[i], kernel::PARAM_MAX[i], kernel::PARAM_STEP_COUNTS[i]);
    ranging.continuation = previousData != nullptr;

    CUDA_MALLOC((void**)&cuda_ranging, sizeof(ranging), "cudaMalloc ranging failed!");

    if (!ranging.continuation)
    {
        CUDA_MEMCPY(cuda_data, kernel::VAR_VALUES, cudaMemcpyHostToDevice, kernel::VAR_COUNT * sizeof(numb), "cudaMemcpy data failed!");
    }
    else
    {
        CUDA_MEMCPY(cuda_prev_data, previousData, cudaMemcpyHostToDevice, size * sizeof(numb), "cudaMemcpy prev data failed!");
    }

    CUDA_MEMCPY(cuda_params, kernel::PARAM_VALUES, cudaMemcpyHostToDevice, kernel::PARAM_COUNT * sizeof(numb), "cudaMemcpy params failed!");
    CUDA_MEMCPY(cuda_maps, maps, cudaMemcpyHostToDevice, mapsSize * sizeof(numb), "cudaMemcpy maps failed!");
    CUDA_MEMCPY(cuda_map_data, &kernel::MAP_DATA, cudaMemcpyHostToDevice, kernel::MAP_COUNT * sizeof(MapData), "cudaMemcpy map data failed!");
    CUDA_MEMCPY(cuda_ranging, &ranging, cudaMemcpyHostToDevice, sizeof(ranging), "cudaMemcpy ranging failed!");

    // Kernel execution
    //precompute = std::chrono::steady_clock::now();
    kernelProgram <<< blocks, threads >>> (cuda_data, cuda_params, cuda_maps, cuda_map_data, cuda_ranging, kernel::steps, kernel::stepSize, variationSize, !ranging.continuation ? 0 : cuda_prev_data);

    CUDA_LASTERROR;

    CUDA_SYNCHRONIZE;
    //incompute = std::chrono::steady_clock::now();

    CUDA_MEMCPY(data, cuda_data, cudaMemcpyDeviceToHost, size * sizeof(numb), "cudaMemcpy back failed!");
    CUDA_MEMCPY(maps, cuda_maps, cudaMemcpyDeviceToHost, mapsSize * sizeof(numb), "cudaMemcpy maps back failed!");

Error:
    cudaFree(cuda_data);
    cudaFree(cuda_prev_data);
    cudaFree(cuda_params);
    cudaFree(cuda_ranging);
    cudaFree(cuda_maps);

    //postcompute = std::chrono::steady_clock::now();
    //printf("Precompute time: %Ii ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(precompute - before).count());
    //printf("Incompute time: %Ii ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(incompute - precompute).count());
    //printf("Postcompute time: %Ii ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(postcompute - incompute).count());

    return cudaStatus;
}

int compute(void** dest, void** maps, numb* previousData, PostRanging* rangingData)
{
    std::chrono::steady_clock::time_point before = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point after;

    // Preparation
    unsigned long int variations = 1; // Parameter/variable variations (ranging steps)
    int currentVariations; // Variations of the current parameter/variable
    int rangingCount = 0;
    rangingData->clear();

    for (int i = 0; i < kernel::VAR_COUNT; i++)
    {
        if (kernel::VAR_RANGING[i])
        {
            currentVariations = calculateStepCount(kernel::VAR_VALUES[i], kernel::VAR_MAX[i], kernel::VAR_STEPS[i]);
            kernel::VAR_STEP_COUNTS[i] = currentVariations;
            variations *= currentVariations;

            rangingData->names.push_back(kernel::VAR_NAMES[i]);
            rangingData->min.push_back(kernel::VAR_VALUES[i]);
            rangingData->step.push_back(kernel::VAR_STEPS[i]);
            rangingData->max.push_back(kernel::VAR_MAX[i]);
            rangingData->stepCount.push_back(currentVariations);
            rangingData->currentStep.push_back(0);
            rangingData->currentValue.push_back(0);
            rangingCount++;
        }
        else
            kernel::VAR_STEP_COUNTS[i] = 1;
    }

    for (int i = 0; i < kernel::PARAM_COUNT; i++)
    {
        if (kernel::PARAM_RANGING[i])
        {
            currentVariations = calculateStepCount(kernel::PARAM_VALUES[i], kernel::PARAM_MAX[i], kernel::PARAM_STEPS[i]);
            kernel::PARAM_STEP_COUNTS[i] = currentVariations;
            variations *= currentVariations;

            rangingData->names.push_back(kernel::PARAM_NAMES[i]);
            rangingData->min.push_back(kernel::PARAM_VALUES[i]);
            rangingData->step.push_back(kernel::PARAM_STEPS[i]);
            rangingData->max.push_back(kernel::PARAM_MAX[i]);
            rangingData->stepCount.push_back(currentVariations);
            rangingData->currentStep.push_back(0);
            rangingData->currentValue.push_back(0);
            rangingCount++;
        }
        else
            kernel::PARAM_STEP_COUNTS[i] = 1;
    }

    rangingData->rangingCount = rangingCount;
    rangingData->totalVariations = variations;
    unsigned long int variationSize = kernel::VAR_COUNT * (kernel::steps + 1); // All steps for the current parameter/variable value combination
    unsigned long int size = variationSize * variations; // Entire data array size
    if (*dest == nullptr) *dest = (void*)(new numb[size]);

    // Maps allocate memory for a Xvariations*Yvariations matrix, where X and Y are MAP_X and MAP_Y

    unsigned long int mapsSize = 0;
    for (int i = 0; i < kernel::MAP_COUNT; i++)
    {
        int index = kernel::MAP_DATA->indexX;
        switch (kernel::MAP_DATA->typeX)
        {
        case VARIABLE:
            kernel::MAP_DATA[i].xSize = kernel::VAR_RANGING[index] ? calculateStepCount(kernel::VAR_VALUES[index], kernel::VAR_MAX[index], kernel::VAR_STEPS[index]) : 1;
            break;
        case PARAMETER:
            kernel::MAP_DATA[i].xSize = kernel::PARAM_RANGING[index] ? calculateStepCount(kernel::PARAM_VALUES[index], kernel::PARAM_MAX[index], kernel::PARAM_STEPS[index]) : 1;
            break;
        case STEP:
            kernel::MAP_DATA[i].xSize = kernel::steps;
            break;
        }

        index = kernel::MAP_DATA->indexY;
        switch (kernel::MAP_DATA->typeY)
        {
        case VARIABLE:
            kernel::MAP_DATA[i].ySize = kernel::VAR_RANGING[index] ? calculateStepCount(kernel::VAR_VALUES[index], kernel::VAR_MAX[index], kernel::VAR_STEPS[index]) : 1;
            break;
        case PARAMETER:
            kernel::MAP_DATA[i].ySize = kernel::PARAM_RANGING[index] ? calculateStepCount(kernel::PARAM_VALUES[index], kernel::PARAM_MAX[index], kernel::PARAM_STEPS[index]) : 1;
            break;
        case STEP:
            kernel::MAP_DATA[i].ySize = kernel::steps;
            break;
        }

        kernel::MAP_DATA[i].offset = mapsSize;
        mapsSize += kernel::MAP_DATA[i].xSize * kernel::MAP_DATA[i].ySize;
    }
    if (*maps == nullptr) *maps = (void*)(new numb[mapsSize]);

    bool hasFailed = false;

    // Execution

    cudaError_t cudaStatus = execute((numb*)(*dest), (numb*)(*maps), rangingCount, variationSize, variations, mapsSize, previousData);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "execute failed!\n"); hasFailed = true; }

    // Output

    after = std::chrono::steady_clock::now();
    std::chrono::steady_clock::duration elapsed = after - before;
    auto timeElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    //printf("CUDA ended in %Ii ms\n", timeElapsed);
    rangingData->timeElapsed = (float)timeElapsed;

#define WRITE 0
#if WRITE
    std::ofstream outputFile;
    outputFile.open("lorentz.txt", std::ios::out);
    for (unsigned long int i = 0; i < size / 3; i++)
    {
        outputFile << data[3 * i + kernel::x] << " " << data[3 * i + kernel::y] << " " << data[3 * i + kernel::z] << std::endl;
    }
#endif

    CUDA_RESET;

    return hasFailed;
}