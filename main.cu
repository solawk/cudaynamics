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
    unsigned long long size = CUDA_marshal.variationSize * CUDA_marshal.totalVariations;

    //std::chrono::steady_clock::time_point before = std::chrono::steady_clock::now();
    //std::chrono::steady_clock::time_point precompute, incompute, postcompute;

    cudaError_t cudaStatus;

    int blocks = (int)ceil((float)CUDA_marshal.totalVariations / KERNEL_TPB);
    int threads = KERNEL_TPB;
    data->threads_per_block = KERNEL_TPB;

    // We create a dummy CUDA Computation with a Marshal to store the pointers in
    Computation* cuda_computation = nullptr;
    numb* cuda_trajectory = nullptr;
    numb* cuda_parameters = nullptr;
    int* cuda_stepIndices = nullptr;
    numb* cuda_maps2 = nullptr;

    CUDA_SET_DEVICE;

    CUDA_MALLOC(&cuda_computation, sizeof(Computation), "cudaMalloc computation failed!");
    CUDA_MALLOC(&cuda_trajectory, size * sizeof(numb), "cudaMalloc data failed!");
    CUDA_MALLOC(&cuda_parameters, (unsigned long long)CUDA_marshal.totalVariations * CUDA_kernel.PARAM_COUNT * sizeof(numb), "cudaMalloc params failed!");
    CUDA_MALLOC(&cuda_stepIndices, (unsigned long long)CUDA_marshal.totalVariations * CUDA_ATTR_COUNT * sizeof(int), "cudaMalloc indices failed!");
    if (CUDA_marshal.totalVariations > 1) CUDA_MALLOC(&cuda_maps2, (unsigned long long)CUDA_marshal.totalVariations * CUDA_kernel.MAP_COUNT * sizeof(numb), "cudaMalloc maps2 failed!");

    CUDA_MEMCPY(cuda_computation, data, cudaMemcpyHostToDevice, sizeof(Computation), "cudaMemcpy computation failed!");
    CUDA_MEMCPY(&(cuda_computation->marshal.trajectory), &cuda_trajectory, cudaMemcpyHostToDevice, sizeof(numb*), "cudaMemcpy trajectory address failed!");
    CUDA_MEMCPY(&(cuda_computation->marshal.parameterVariations), &cuda_parameters, cudaMemcpyHostToDevice, sizeof(numb*), "cudaMemcpy parameter address failed!");
    CUDA_MEMCPY(&(cuda_computation->marshal.stepIndices), &cuda_stepIndices, cudaMemcpyHostToDevice, sizeof(int*), "cudaMemcpy indices address failed!");
    if (CUDA_marshal.totalVariations > 1) CUDA_MEMCPY(&(cuda_computation->marshal.maps2), &cuda_maps2, cudaMemcpyHostToDevice, sizeof(numb*), "cudaMemcpy maps2 address failed!");

    CUDA_MEMCPY(cuda_trajectory, CUDA_marshal.trajectory, cudaMemcpyHostToDevice, size * sizeof(numb), "cudaMemcpy data failed!");
    CUDA_MEMCPY(cuda_parameters, CUDA_marshal.parameterVariations, cudaMemcpyHostToDevice, (unsigned long long)CUDA_marshal.totalVariations * CUDA_kernel.PARAM_COUNT * sizeof(numb), "cudaMemcpy params failed!");
    CUDA_MEMCPY(cuda_stepIndices, CUDA_marshal.stepIndices, cudaMemcpyHostToDevice, (unsigned long long)CUDA_marshal.totalVariations * CUDA_ATTR_COUNT * sizeof(int), "cudaMemcpy indices failed!");
    if (CUDA_marshal.totalVariations > 1) CUDA_MEMCPY(cuda_maps2, CUDA_marshal.maps2, cudaMemcpyHostToDevice, (unsigned long long)CUDA_marshal.totalVariations * CUDA_kernel.MAP_COUNT * sizeof(numb), "cudaMemcpy maps2 failed!");

    // Kernel execution
    //precompute = std::chrono::steady_clock::now();
    KERNEL_PROG <<< blocks, threads >>> (cuda_computation);

    CUDA_LASTERROR;

    CUDA_SYNCHRONIZE;
    //incompute = std::chrono::steady_clock::now();

    CUDA_MEMCPY(CUDA_marshal.trajectory, cuda_trajectory, cudaMemcpyDeviceToHost, size * sizeof(numb), "cudaMemcpy back failed!");
    if (CUDA_marshal.totalVariations > 1) CUDA_MEMCPY(CUDA_marshal.maps2, cuda_maps2, cudaMemcpyDeviceToHost, (unsigned long long)CUDA_marshal.totalVariations * CUDA_kernel.MAP_COUNT * sizeof(numb), "cudaMemcpy maps back failed!");

Error:
    if (cuda_trajectory != nullptr) cudaFree(cuda_trajectory);
    if (cuda_parameters != nullptr) cudaFree(cuda_parameters);
    if (cuda_stepIndices != nullptr) cudaFree(cuda_stepIndices);
    if (cuda_maps2 != nullptr) cudaFree(cuda_maps2);
    if (cuda_computation != nullptr) cudaFree(cuda_computation);

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
    unsigned long int variations = 1; // Parameter/variable variations (ranging steps)

    CUDA_kernel.CopyFrom(&KERNEL);

    for (int i = 0; i < CUDA_kernel.VAR_COUNT; i++)
        if (CUDA_kernel.variables[i].TrueStepCount() > 1)
            variations *= CUDA_kernel.variables[i].stepCount;

    for (int i = 0; i < CUDA_kernel.PARAM_COUNT; i++)
        if (CUDA_kernel.parameters[i].TrueStepCount() > 1)
            variations *= CUDA_kernel.parameters[i].stepCount;

    CUDA_marshal.totalVariations = variations;
    unsigned long long trajectorySize = CUDA_kernel.VAR_COUNT * (CUDA_kernel.steps + 1); // All steps for the current parameter/variable value combination
    CUDA_marshal.variationSize = (int)trajectorySize;
    unsigned long long size = trajectorySize * variations; // Entire data array size

    if (CUDA_marshal.trajectory == nullptr) CUDA_marshal.trajectory = new numb[size];
    if (CUDA_marshal.parameterVariations == nullptr) CUDA_marshal.parameterVariations = new numb[CUDA_kernel.PARAM_COUNT * variations];
    if (CUDA_marshal.stepIndices == nullptr) CUDA_marshal.stepIndices = new int[CUDA_ATTR_COUNT * variations];

    // Vector of attribute steps (indices of values) is now outside the filling function, this way we can use it in several iterations, essential for hi-res computations
    std::vector<int> attributeStepIndices;
    for (int i = 0; i < CUDA_ATTR_COUNT; i++) attributeStepIndices.push_back(0);

    fillAttributeBuffers(data, attributeStepIndices, 0, variations);
    setMapValues(data);

    bool hasFailed = false;

    // Execution

    cudaError_t cudaStatus = execute(data);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "execute failed!\n"); hasFailed = true; }

    // Output

    after = std::chrono::steady_clock::now();
    std::chrono::steady_clock::duration elapsed = after - before;
    auto timeElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    data->timeElapsed = (float)timeElapsed;

    CUDA_RESET;

    return hasFailed;
}


void fillAttributeBuffers(Computation* data, std::vector<int>& attributeStepIndices, unsigned long long startVariation, unsigned long long endVariation)
{
    unsigned long long varStride = CUDA_marshal.variationSize;    // Stride between variations in "trajectory"
    unsigned long long paramStride = CUDA_kernel.PARAM_COUNT;     // Stride between variations in "parameterVariations"

    for (unsigned long long i = startVariation; i < endVariation; i++)
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

void setMapValues(Computation* data)
{
    // Initialize buffer
    if (CUDA_marshal.totalVariations > 1 && CUDA_kernel.MAP_COUNT > 0 && CUDA_marshal.maps2 == nullptr)
        CUDA_marshal.maps2 = new numb[CUDA_marshal.totalVariations * CUDA_kernel.MAP_COUNT];

    if (CUDA_marshal.totalVariations == 1) for (int m = 0; m < CUDA_kernel.MAP_COUNT; m++)
    {
        CUDA_kernel.mapDatas[m].toCompute = false;
        return;
    }

    // Copy previous map values if present
    if (data->isFirst || !CUDA_kernel.continuousMaps)
    {
        memset(CUDA_marshal.maps2, 0, sizeof(numb) * CUDA_marshal.totalVariations * CUDA_kernel.MAP_COUNT);
    }
    else
    {
        memcpy(CUDA_marshal.maps2, data->otherMarshal->maps2, sizeof(numb) * CUDA_marshal.totalVariations * CUDA_kernel.MAP_COUNT);
    }
}