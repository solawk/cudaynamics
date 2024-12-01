#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <fstream>

#include "sjj.h"
#include <objects.h>
#include <chrono>
#include <wtypes.h>

#define V0(n) varValues[kernel::n]
#define V(n) data[stepStart + kernel::n]
#define P(n) paramValues[kernel::n]
#define NEXT kernel::VAR_COUNT

namespace kernel
{
    const char* name = "Shunted Josephson Junction";

    const char* VAR_NAMES[]{ "x1", "x2", "x3" };
    float VAR_VALUES[]{ -0.31f, 3.3f, 0.76f };
    bool VAR_RANGING[]{ true, true, true };
    float VAR_STEPS[]{ 0.1f, 0.1f, 0.1f };
    float VAR_MAX[]{ -0.01f, 4.3f, 1.76f };
    int VAR_STEP_COUNTS[]{ 0, 0, 0 };

    const char* PARAM_NAMES[]{ "betaL", "betaC", "i", "Vg/IcRs", "Rn", "Rsg" };
    float PARAM_VALUES[]{ 29.215f, 0.707f, 1.25f, 6.9f, 0.367f, 0.0478f };
    bool PARAM_RANGING[]{ false, false, false, false, false, false };
    float PARAM_STEPS[]{ 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    float PARAM_MAX[]{ 19.0f, 40.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    int PARAM_STEP_COUNTS[]{ 0, 0, 0, 0, 0, 0 };

    bool executeOnLaunch = true;
    int steps = 10000;
    float stepSize = 0.01f;
    bool onlyShowLast = false;
}

const int THREADS_PER_BLOCK = 256;

__global__ void kernelProgram(float* data, float* params, PreRanging* ranging, int steps, float h, int variationSize)
{
    int b = blockIdx.x;                                     // Current block of THREADS_PER_BLOCK threads
    int t = threadIdx.x;                                    // Current thread in the block, from 0 to THREADS_PER_BLOCK-1
    int variation = (b * THREADS_PER_BLOCK) + t;            // Variation (parameter combination) index
    if (variation >= ranging->totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    int variationStart = variation * variationSize;         // Start index to store the modelling data for the variation
    int stepStart = variationStart;                         // Start index for the current modelling step

    float varValues[kernel::VAR_COUNT];
    float paramValues[kernel::PARAM_COUNT];
    for (int i = 0; i < kernel::VAR_COUNT; i++) varValues[i] = data[i];
    for (int i = 0; i < kernel::PARAM_COUNT; i++) paramValues[i] = params[i];

    // Editing initial state and parameters from ranging
    int tVariation = variation;
    for (int i = ranging->rangingCount - 1; i >= 0; i--)
    {
        bool isVar = ranging->rangings[i].index < kernel::VAR_COUNT;
        int csteps = ranging->rangings[i].steps;
        int step = tVariation % csteps;
        tVariation = tVariation / csteps;
        float value = ranging->rangings[i].min + ranging->rangings[i].step * step;
        
        if (isVar)
        {
            varValues[ranging->rangings[i].index] = value;
        }
        else
        {
            paramValues[ranging->rangings[i].index - kernel::VAR_COUNT] = value;
        }
    }

    // Copying initial state to other variations
    V(x0) = V0(x0);
    V(x1) = V0(x1);
    V(x2) = V0(x2);

    for (int s = 0; s < steps; s++)
    {
        stepStart = variationStart + s * NEXT;

        // x[0] = x[0] + h*( x[1] );
        // x[2] = x[2] + h * ((1 / p[0]) * (x[1] - x[2]));
        // x[1] = x[1] + h * ((1 / p[1]) * (p[2] - ((x[1] > p[3]) ? p[4] : p[5]) * x[1] - sin(x[0]) - x[2]));

        V(x0 + NEXT) = V(x0) + h * V(x1);
        V(x2 + NEXT) = V(x2) + h * (1.0f / P(p0)) * (V(x1) - V(x2));
        V(x1 + NEXT) = V(x1) + h * (1.0f / P(p1)) *
            (
                P(p2)
                - ((V(x1) > P(p3)) ? P(p4) : P(p5)) * V(x1)
                - sinf(V(x0 + NEXT))
                - V(x2 + NEXT)
            );
    }

    for (int s = 0; s < steps + 1; s++)
    {
        data[variationStart + s * NEXT + 0] = sinf(data[variationStart + s * NEXT + 0]);
    }
}

cudaError_t execute(float* data, int rangingCount, int variationSize, int variations, unsigned long int size)
{
    std::chrono::steady_clock::time_point before = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point precompute, incompute, postcompute;

    cudaError_t cudaStatus;

    int blocks = (int)ceil((float)variations / THREADS_PER_BLOCK);
    int threads = THREADS_PER_BLOCK;

    PreRanging ranging(kernel::VAR_COUNT, kernel::PARAM_COUNT, rangingCount, variations);
    int rangingIndex = 0;

    float* cuda_data = 0;
    float* cuda_params = 0;
    PreRanging* cuda_ranging = 0;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess){fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");goto Error;}

    // Allocating the data array
    cudaStatus = cudaMalloc((void**)&cuda_data, size * sizeof(float));
    if (cudaStatus != cudaSuccess){fprintf(stderr, "cudaMalloc data failed!");goto Error;}

    // Allocating the parameter array
    cudaStatus = cudaMalloc((void**)&cuda_params, kernel::PARAM_COUNT * sizeof(float));
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc params failed!"); goto Error; }

    // Parameters array structure (PreRanging):
    for (int i = 0; i < kernel::VAR_COUNT; i++)
        if (kernel::VAR_RANGING[i])
            ranging.rangings[rangingIndex++].init(i, kernel::VAR_VALUES[i], kernel::VAR_STEPS[i], kernel::VAR_MAX[i], kernel::VAR_STEP_COUNTS[i]);
    for (int i = 0; i < kernel::PARAM_COUNT; i++)
        if (kernel::PARAM_RANGING[i])
            ranging.rangings[rangingIndex++].init(i + kernel::VAR_COUNT, kernel::PARAM_VALUES[i], kernel::PARAM_STEPS[i], kernel::PARAM_MAX[i], kernel::PARAM_STEP_COUNTS[i]);

    // Allocating the ranging struct
    cudaStatus = cudaMalloc((void**)&cuda_ranging, sizeof(ranging));
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc ranging failed!"); goto Error; }

    // Copying the initial variable values
    cudaStatus = cudaMemcpy(cuda_data, kernel::VAR_VALUES, kernel::VAR_COUNT * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess){fprintf(stderr, "cudaMemcpy data failed!");goto Error;}

    // Copying the parameter values
    cudaStatus = cudaMemcpy(cuda_params, kernel::PARAM_VALUES, kernel::PARAM_COUNT * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess){fprintf(stderr, "cudaMemcpy params failed!");goto Error;}

    // Copying the ranging struct
    cudaStatus = cudaMemcpy(cuda_ranging, &ranging, sizeof(ranging), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy ranging failed!"); goto Error; }

    // Kernel execution
    precompute = std::chrono::steady_clock::now();
    kernelProgram <<< blocks, threads >>> (cuda_data, cuda_params, cuda_ranging, kernel::steps, kernel::stepSize, variationSize);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess){fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));goto Error;}

    // Awaiting kernel execution end
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess){fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel!\n", cudaStatus);goto Error;}
    incompute = std::chrono::steady_clock::now();

    // Copying the computed data back
    cudaStatus = cudaMemcpy(data, cuda_data, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess){fprintf(stderr, "cudaMemcpy failed!");goto Error;}

Error:
    cudaFree(cuda_data);

    postcompute = std::chrono::steady_clock::now();
    printf("Precompute time: %i ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(precompute - before).count());
    printf("Incompute time: %i ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(incompute - precompute).count());
    printf("Postcompute time: %i ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(postcompute - incompute).count());

    return cudaStatus;
}

int compute(void** dest, PostRanging* rangingData, HANDLE* writeSemaphore)
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
            currentVariations = (int)((kernel::VAR_MAX[i] - kernel::VAR_VALUES[i]) / kernel::VAR_STEPS[i]) + 1;
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
            kernel::VAR_STEP_COUNTS[i] = 0;
    }

    for (int i = 0; i < kernel::PARAM_COUNT; i++)
    {
        if (kernel::PARAM_RANGING[i])
        {
            currentVariations = (int)((kernel::PARAM_MAX[i] - kernel::PARAM_VALUES[i]) / kernel::PARAM_STEPS[i]) + 1;
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
            kernel::PARAM_STEP_COUNTS[i] = 0;
    }

    rangingData->rangingCount = rangingCount;
    rangingData->totalVariations = variations;
    unsigned long int variationSize = kernel::VAR_COUNT * (kernel::steps + 1); // All steps for the current parameter/variable value combination
    unsigned long int size = variationSize * variations; // Entire data array size
    float* data = new float[size];
    bool hasFailed = false;

    // Execution

    cudaError_t cudaStatus = execute(data, rangingCount, variationSize, variations, size);
    if (cudaStatus != cudaSuccess){fprintf(stderr, "execute failed!\n");hasFailed = true;}

    // Output

    after = std::chrono::steady_clock::now();
    std::chrono::steady_clock::duration elapsed = after - before;
    printf("CUDA ended in %i ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count());

    /*for (int i = 0; i < kernel::steps + 1; i++)
    {
        printf("%f %f %f\n", data[i * NEXT + kernel::x], data[i * NEXT + kernel::y], data[i * NEXT + kernel::z]);
    }*/

#define WRITE 0
#if WRITE
    std::ofstream outputFile;
    outputFile.open("output.txt", std::ios::out);
    for (unsigned long int i = 0; i < size / 3; i++)
    {
        outputFile << data[3 * i + kernel::x0] << " " << data[3 * i + kernel::x1] << " " << data[3 * i + kernel::x2] << std::endl;
    }
#endif

    // Resetting

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess){fprintf(stderr, "cudaDeviceReset failed!\n");hasFailed = true;}

    *dest = data;
    return hasFailed;
}
