#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <fstream>
#include <chrono>

#include "main.h"
#include "cuda_macros.h"
#include <objects.h>

cudaError_t execute(Computation* data)
{
    // Hi-res only requires limited amount of variations
    unsigned long long variations = !data->isHires ? CUDA_marshal.totalVariations : data->variationsInCurrentExecute;
    unsigned long long size = CUDA_marshal.variationSize * variations;
    int totalMapValues = !data->isHires ? CUDA_marshal.totalMapValuesPerVariation : 1; // We always only calculate one map in hi-res, controlled by "toCompute"s

    //std::chrono::steady_clock::time_point before = std::chrono::steady_clock::now();
    //std::chrono::steady_clock::time_point precompute, incompute, postcompute;

    cudaError_t cudaStatus;

    int blocks = (int)ceil((float)variations / KERNEL_TPB);
    int threads = KERNEL_TPB;
    data->threads_per_block = KERNEL_TPB;

    // We create a dummy CUDA Computation with a Marshal to store the pointers in
    Computation* cuda_computation = nullptr;
    numb* cuda_trajectory = nullptr;
    numb* cuda_parameters = nullptr;
    int* cuda_stepIndices = nullptr;
    numb* cuda_maps = nullptr;

    CUDA_SET_DEVICE;

    // Allocation memory on the device for the Computation struct and the buffers
    CUDA_MALLOC(&cuda_computation, sizeof(Computation), "cudaMalloc computation failed!");
    CUDA_MALLOC(&cuda_trajectory, size * sizeof(numb), "cudaMalloc data failed!");
    CUDA_MALLOC(&cuda_parameters, variations * CUDA_kernel.PARAM_COUNT * sizeof(numb), "cudaMalloc params failed!");
    CUDA_MALLOC(&cuda_stepIndices, variations * CUDA_ATTR_COUNT * sizeof(int), "cudaMalloc indices failed!");
    if (totalMapValues > 0 && variations > 1) CUDA_MALLOC(&cuda_maps, variations * totalMapValues * sizeof(numb), "cudaMalloc maps failed!");

    // Copying the Computation struct to the device
    CUDA_MEMCPY(cuda_computation, data, cudaMemcpyHostToDevice, sizeof(Computation), "cudaMemcpy computation failed!");
    // Copying addresses of the device-side buffers to the device-side Computation
    CUDA_MEMCPY(&(cuda_computation->marshal.trajectory), &cuda_trajectory, cudaMemcpyHostToDevice, sizeof(numb*), "cudaMemcpy trajectory address failed!");
    CUDA_MEMCPY(&(cuda_computation->marshal.parameterVariations), &cuda_parameters, cudaMemcpyHostToDevice, sizeof(numb*), "cudaMemcpy parameter address failed!");
    CUDA_MEMCPY(&(cuda_computation->marshal.stepIndices), &cuda_stepIndices, cudaMemcpyHostToDevice, sizeof(int*), "cudaMemcpy indices address failed!");
    if (totalMapValues > 0 && variations > 1) CUDA_MEMCPY(&(cuda_computation->marshal.maps), &cuda_maps, cudaMemcpyHostToDevice, sizeof(numb*), "cudaMemcpy maps address failed!");

    // Copying the values in the buffers themselves to the device
    CUDA_MEMCPY(cuda_trajectory, CUDA_marshal.trajectory, cudaMemcpyHostToDevice, size * sizeof(numb), "cudaMemcpy data failed!");
    CUDA_MEMCPY(cuda_parameters, CUDA_marshal.parameterVariations, cudaMemcpyHostToDevice, variations * CUDA_kernel.PARAM_COUNT * sizeof(numb), "cudaMemcpy params failed!");
    CUDA_MEMCPY(cuda_stepIndices, CUDA_marshal.stepIndices, cudaMemcpyHostToDevice, variations * CUDA_ATTR_COUNT * sizeof(int), "cudaMemcpy indices failed!");
    // We don't need to account for multiple maps since we only calculate one at once
    if (variations > 1) CUDA_MEMCPY(cuda_maps, CUDA_marshal.maps + (!data->isHires ? 0 : data->startVariationInCurrentExecute),
        cudaMemcpyHostToDevice, variations * totalMapValues * sizeof(numb), "cudaMemcpy maps failed!");

    // Kernel execution
    //precompute = std::chrono::steady_clock::now();
    KERNEL_PROG <<< blocks, threads >>> (cuda_computation);
    CUDA_LASTERROR;
    CUDA_SYNCHRONIZE;
    //incompute = std::chrono::steady_clock::now();

    // Copying the trajectories and the maps back to the host
    CUDA_MEMCPY(CUDA_marshal.trajectory, cuda_trajectory, cudaMemcpyDeviceToHost, size * sizeof(numb), "cudaMemcpy back failed!");
    if (variations > 1) CUDA_MEMCPY(CUDA_marshal.maps + (!data->isHires ? 0 : data->startVariationInCurrentExecute), cuda_maps,
        cudaMemcpyDeviceToHost, variations * totalMapValues * sizeof(numb), "cudaMemcpy maps back failed!");

Error:
    if (cuda_trajectory != nullptr) cudaFree(cuda_trajectory);
    if (cuda_parameters != nullptr) cudaFree(cuda_parameters);
    if (cuda_stepIndices != nullptr) cudaFree(cuda_stepIndices);
    if (cuda_maps != nullptr) cudaFree(cuda_maps);
    if (cuda_computation != nullptr) cudaFree(cuda_computation);

    CUDA_RESET;

    //postcompute = std::chrono::steady_clock::now();
    //printf("Precompute time: %Ii ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(precompute - before).count());
    //printf("Incompute time: %Ii ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(incompute - precompute).count());
    //printf("Postcompute time: %Ii ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(postcompute - incompute).count());

    return cudaStatus;
}

int compute(Computation* data)
{
    std::chrono::steady_clock::time_point before = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point after;

    // Preparation
    unsigned long long variations = 1; // Parameter/variable variations (ranging steps)

    for (int i = 0; i < CUDA_kernel.VAR_COUNT; i++)
        if (CUDA_kernel.variables[i].TrueStepCount() > 1)
            variations *= CUDA_kernel.variables[i].stepCount;

    for (int i = 0; i < CUDA_kernel.PARAM_COUNT; i++)
        if (CUDA_kernel.parameters[i].TrueStepCount() > 1)
            variations *= CUDA_kernel.parameters[i].stepCount;

    CUDA_marshal.totalVariations = (int)variations;
    unsigned long long variationSize = CUDA_kernel.VAR_COUNT * (CUDA_kernel.steps + 1); // All steps for the current parameter/variable value combination
    CUDA_marshal.variationSize = (int)variationSize;

    unsigned long long variationsInBuffers = !data->isHires ? variations : data->variationsPerParallelization;
    
    if (CUDA_marshal.trajectory == nullptr) CUDA_marshal.trajectory = new numb[variationSize * variationsInBuffers];
    if (CUDA_marshal.parameterVariations == nullptr) CUDA_marshal.parameterVariations = new numb[CUDA_kernel.PARAM_COUNT * variationsInBuffers];
    if (CUDA_marshal.stepIndices == nullptr) CUDA_marshal.stepIndices = new int[CUDA_ATTR_COUNT * variationsInBuffers];

    // Vector of attribute steps (indices of values) is now outside the filling function, this way we can use it in several iterations, essential for hi-res computations
    int* attributeStepIndices = new int[CUDA_ATTR_COUNT];
    for (int i = 0; i < CUDA_ATTR_COUNT; i++) attributeStepIndices[i] = 0;
    setMapValues(data);

    bool hasFailed = false;
    cudaError_t cudaStatus;

    // Execution
    if (!data->isHires)
    {
        fillAttributeBuffers(data, attributeStepIndices, 0, variations, false);
        cudaStatus = execute(data);
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "execute failed!\n"); hasFailed = true; }
    }
    else
    {
        data->variationsFinished = 0;
        data->bufferNo = 0;
        data->otherMarshal = &(CUDA_marshal); // We trick it into thinking its own trajectory is the previous trajectory when copying the variable values (ouroboros moment)
        for (unsigned long long v = 0; v < variations; v += data->variationsPerParallelization)
        {
            unsigned long long variationsCurrent = min(variations - v, data->variationsPerParallelization);
            data->variationsInCurrentExecute = variationsCurrent;
            data->startVariationInCurrentExecute = v;

            data->isFirst = true;
            for (int b = 0; b < data->buffersPerVariation; b++)
            {
                fillAttributeBuffers(data, attributeStepIndices, v, v + variationsCurrent, !data->isFirst);
                cudaStatus = execute(data);
                if (cudaStatus != cudaSuccess) { fprintf(stderr, "execute failed!\n"); hasFailed = true; break; }
                data->isFirst = false;
            }

            data->variationsFinished = v;
        }
    }

    // Output

    after = std::chrono::steady_clock::now();
    std::chrono::steady_clock::duration elapsed = after - before;
    auto timeElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    data->timeElapsed = (float)timeElapsed;

    delete[] attributeStepIndices;

    return hasFailed;
}


void fillAttributeBuffers(Computation* data, int* attributeStepIndices, unsigned long long startVariation, unsigned long long endVariation, bool onlyTrajectory)
{
    unsigned long long varStride = CUDA_marshal.variationSize;    // Stride between variations in "trajectory"
    unsigned long long paramStride = CUDA_kernel.PARAM_COUNT;     // Stride between variations in "parameterVariations"

    for (unsigned long long i = 0; i < endVariation - startVariation; i++)
    {
        if (data->isFirst) // Meaning no previous trajectory
        {
            // Forming buffer from attribute values
            // Counting step indeces like a number, incrementing by 1 for each variation
            for (int v = 0; v < CUDA_kernel.VAR_COUNT; v++)
            {
                // i * stride - start of the variation, + v for the variable
                // setting a value from the variable's ranged values, knowing the step index is among the first of "attribute step indices"
                CUDA_marshal.trajectory[i * varStride + v] = CUDA_kernel.variables[v].values[attributeStepIndices[v]];
            }
        }
        else
        {
            // Copying from previous trajectory
            for (int v = 0; v < CUDA_kernel.VAR_COUNT; v++)
            {
                // Left side is the first step of the trajectory
                // Right side is the last step of the previous trajectory
                CUDA_marshal.trajectory[i * varStride + v] = data->otherMarshal->trajectory[i * varStride + (CUDA_kernel.steps * CUDA_kernel.VAR_COUNT) + v];
            }
        }

        if (!onlyTrajectory)
        {
            for (int p = 0; p < CUDA_kernel.PARAM_COUNT; p++)
                CUDA_marshal.parameterVariations[i * paramStride + p] = CUDA_kernel.parameters[p].values[attributeStepIndices[p + CUDA_kernel.VAR_COUNT]];

            for (int j = 0; j < CUDA_ATTR_COUNT; j++)
                CUDA_marshal.stepIndices[i * CUDA_ATTR_COUNT + j] = attributeStepIndices[j];

            // Incrementing the "attribute step indices" total number
            for (int j = CUDA_ATTR_COUNT - 1; j >= 0; j--)
            {
                attributeStepIndices[j]++;

                bool isParam = j >= CUDA_kernel.VAR_COUNT;
                int stepCountOfAttribute = isParam ?
                    CUDA_kernel.parameters[j - CUDA_kernel.VAR_COUNT].TrueStepCount() :
                    CUDA_kernel.variables[j].TrueStepCount();

                if (attributeStepIndices[j] < stepCountOfAttribute) break;
                attributeStepIndices[j] = 0;
            }
        }
    }
}

void setMapValues(Computation* data)
{
    if (CUDA_marshal.totalVariations == 1)
        for (int m = 0; m < CUDA_kernel.MAP_COUNT; m++)
        {
            CUDA_kernel.mapDatas[m].toCompute = false;
        }

    // Look through all maps and set their offsets depending on which are to be computed and which are not
    int offset = 0; // Offset is counted in maps to be computed, so it's then multiplied by totalVariations on the device
    for (int m = 0; m < CUDA_kernel.MAP_COUNT; m++)
    {
        if (CUDA_kernel.mapDatas[m].toCompute)
        {
            CUDA_kernel.mapDatas[m].offset = offset;
            offset += CUDA_kernel.mapDatas[m].valueCount;
        }
    }
    CUDA_marshal.totalMapValuesPerVariation = offset;

    // Initialize buffer
    if (CUDA_marshal.totalVariations > 1 && CUDA_kernel.MAP_COUNT > 0 && CUDA_marshal.maps == nullptr)
    {
        CUDA_marshal.maps = new numb[CUDA_marshal.totalVariations * CUDA_marshal.totalMapValuesPerVariation];
    }

    // Copy previous map values if present
    if (data->isFirst || CUDA_kernel.mapWeight == 1.0f)
    {
        memset(CUDA_marshal.maps, 0, sizeof(numb) * CUDA_marshal.totalVariations * CUDA_marshal.totalMapValuesPerVariation);
    }
    else
    {
        memcpy(CUDA_marshal.maps, data->otherMarshal->maps, sizeof(numb) * CUDA_marshal.totalVariations * CUDA_marshal.totalMapValuesPerVariation);
    }
}