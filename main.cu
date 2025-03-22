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
    //numb* cuda_maps;

    CUDA_SET_DEVICE;

    CUDA_MALLOC(&cuda_computation, sizeof(Computation), "cudaMalloc computation failed!");
    CUDA_MALLOC(&cuda_trajectory, size * sizeof(numb), "cudaMalloc data failed!");
    CUDA_MALLOC(&cuda_parameters, (unsigned long long)CUDA_marshal.totalVariations * CUDA_kernel.PARAM_COUNT * sizeof(numb), "cudaMalloc data failed!");
    //CUDA_MALLOC(&cuda_maps, CUDA_marshal.mapsSize * sizeof(numb), "cudaMalloc maps failed!");

    CUDA_MEMCPY(cuda_computation, data, cudaMemcpyHostToDevice, sizeof(Computation), "cudaMemcpy computation failed!");
    CUDA_MEMCPY(&(cuda_computation->marshal.trajectory), &cuda_trajectory, cudaMemcpyHostToDevice, sizeof(numb*), "cudaMemcpy trajectory address failed!");
    CUDA_MEMCPY(&(cuda_computation->marshal.parameterVariations), &cuda_parameters, cudaMemcpyHostToDevice, sizeof(numb*), "cudaMemcpy parameter address failed!");

    CUDA_MEMCPY(cuda_trajectory, CUDA_marshal.trajectory, cudaMemcpyHostToDevice, size * sizeof(numb), "cudaMemcpy data failed!");
    CUDA_MEMCPY(cuda_parameters, CUDA_marshal.parameterVariations, cudaMemcpyHostToDevice, (unsigned long long)CUDA_marshal.totalVariations * CUDA_kernel.PARAM_COUNT * sizeof(numb), "cudaMemcpy params failed!");
    //CUDA_MEMCPY(cuda_maps, maps, cudaMemcpyHostToDevice, mapsSize * sizeof(numb), "cudaMemcpy maps failed!");
    //CUDA_MEMCPY(cuda_map_data, &kernel::MAP_DATA, cudaMemcpyHostToDevice, kernel::MAP_COUNT * sizeof(MapData), "cudaMemcpy map data failed!");

    // Kernel execution
    //precompute = std::chrono::steady_clock::now();
    KERNEL_PROG <<< blocks, threads >>> (cuda_computation);

    CUDA_LASTERROR;

    CUDA_SYNCHRONIZE;
    //incompute = std::chrono::steady_clock::now();

    CUDA_MEMCPY(CUDA_marshal.trajectory, cuda_trajectory, cudaMemcpyDeviceToHost, size * sizeof(numb), "cudaMemcpy back failed!");
    //CUDA_MEMCPY(maps, cuda_maps, cudaMemcpyDeviceToHost, mapsSize * sizeof(numb), "cudaMemcpy maps back failed!");

Error:
    if (cuda_trajectory != nullptr) cudaFree(cuda_trajectory);
    if (cuda_parameters != nullptr) cudaFree(cuda_parameters);
    //cudaFree(cuda_maps);
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
    {
        CUDA_kernel.variables[i].CalcStepCount(); // Ackchyually should be calculated in the gui before computation, but let's keep it as a failsafe for now (f o r e v e r)
        CUDA_kernel.variables[i].CalcStep();
        //if (!CUDA_kernel.variables[i].DoValuesExist()) CUDA_kernel.variables[i].Generate(); TODO
        CUDA_kernel.variables[i].ClearValues(); CUDA_kernel.variables[i].Generate();

        if (CUDA_kernel.variables[i].stepCount > 1)
            variations *= CUDA_kernel.variables[i].stepCount;
    }

    for (int i = 0; i < CUDA_kernel.PARAM_COUNT; i++)
    {
        CUDA_kernel.parameters[i].CalcStepCount();
        CUDA_kernel.parameters[i].CalcStep();
        //if (!CUDA_kernel.parameters[i].DoValuesExist()) CUDA_kernel.parameters[i].Generate();
        CUDA_kernel.parameters[i].ClearValues(); CUDA_kernel.parameters[i].Generate();

        if (CUDA_kernel.parameters[i].stepCount > 1)
            variations *= CUDA_kernel.parameters[i].stepCount;
    }

    CUDA_marshal.totalVariations = variations;
    unsigned long long trajectorySize = CUDA_kernel.VAR_COUNT * (CUDA_kernel.steps + 1); // All steps for the current parameter/variable value combination
    CUDA_marshal.variationSize = trajectorySize;
    unsigned long long size = trajectorySize * variations; // Entire data array size

    if (CUDA_marshal.trajectory == nullptr)
        CUDA_marshal.trajectory = new numb[size];

    if (CUDA_marshal.parameterVariations == nullptr) CUDA_marshal.parameterVariations = new numb[CUDA_kernel.PARAM_COUNT * variations];

    fillAttributeBuffers(data);

#define WRITE 0
#if WRITE
    std::ofstream outputFile;
    outputFile.open("cudaynamics input.txt", std::ios::out);
    for (unsigned long int i = 0; i < size / 3; i++)
    {
        outputFile << CUDA_marshal.trajectory[3 * i + 0] << " " << CUDA_marshal.trajectory[3 * i + 1] << " " << CUDA_marshal.trajectory[3 * i + 2] << std::endl;
    }
    outputFile.close();
#endif

    // Maps allocate memory for a Xvariations*Yvariations matrix, where X and Y are MAP_X and MAP_Y

    CUDA_kernel.MapsSetSizes();
    CUDA_marshal.mapsSize = 0;
    for (int i = 0; i < CUDA_kernel.MAP_COUNT; i++) CUDA_marshal.mapsSize += CUDA_kernel.mapDatas[i].xSize * CUDA_kernel.mapDatas[i].ySize;
    delete[] CUDA_marshal.maps;
    CUDA_marshal.maps = nullptr;
    if (CUDA_marshal.mapsSize > 0) CUDA_marshal.maps = new numb[CUDA_marshal.mapsSize];

    bool hasFailed = false;

    // Execution

    cudaError_t cudaStatus = execute(data);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "execute failed!\n"); hasFailed = true; }

    // Output

    after = std::chrono::steady_clock::now();
    std::chrono::steady_clock::duration elapsed = after - before;
    auto timeElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    //printf("CUDA ended in %Ii ms\n", timeElapsed);
    data->timeElapsed = (float)timeElapsed;

#if WRITE
    outputFile.open("cudaynamics output.txt", std::ios::out);
    for (unsigned long int i = 0; i < size / 3; i++)
    {
        outputFile << CUDA_marshal.trajectory[3 * i + 0] << " " << CUDA_marshal.trajectory[3 * i + 1] << " " << CUDA_marshal.trajectory[3 * i + 2] << std::endl;
    }
    outputFile.close();
#endif

    CUDA_RESET;

    return hasFailed;
}


void fillAttributeBuffers(Computation* data)
{
    unsigned long long varStride = CUDA_marshal.variationSize;    // Stride between variations in "trajectory"
    unsigned long long paramStride = CUDA_kernel.PARAM_COUNT;                          // Stride between variations in "parameterVariations"
   
    std::vector<int> attributeStepIndices;
    int totalAttributes = CUDA_kernel.VAR_COUNT + CUDA_kernel.PARAM_COUNT;
    for (int i = 0; i < totalAttributes; i++) attributeStepIndices.push_back(0);

    for (unsigned long long i = 0; i < CUDA_marshal.totalVariations; i++)
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
        {
            CUDA_marshal.parameterVariations[i * paramStride + p] = CUDA_kernel.parameters[p].values[attributeStepIndices[p + CUDA_kernel.VAR_COUNT]];
        }

        // Incrementing the "attribute step indices" total number
        for (int j = totalAttributes - 1; j >= 0; j--)
        {
            attributeStepIndices[j]++;

            bool isParam = j >= CUDA_kernel.VAR_COUNT;
            int stepCountOfAttribute = isParam ? CUDA_kernel.parameters[j - CUDA_kernel.VAR_COUNT].stepCount : CUDA_kernel.variables[j].stepCount;

            if (attributeStepIndices[j] < stepCountOfAttribute) break;
            attributeStepIndices[j] = 0;
        }
    }
}