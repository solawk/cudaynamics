#include "cuda_runtime.h"
#include "cuda_macros.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <fstream>
#include <chrono>

#include "lorenzExample.h"
#include "cuda_macros.h"
#include <objects.h>

namespace kernel
{
    const char* name = "Lorenz system";

    const char* VAR_NAMES[]{ "x", "y", "z" };
    float VAR_VALUES[]{ 1.0f, 1.0f, 1.0f };
    bool VAR_RANGING[]{ true, true, true };
    float VAR_STEPS[]{ 3.0f, 3.0f, 3.0f };
    float VAR_MAX[]{ 29.0f, 29.0f, 29.0f };
    int VAR_STEP_COUNTS[]{ 0, 0, 0 };

    const char* PARAM_NAMES[]{ "sigma", "rho", "beta" };
    float PARAM_VALUES[]{ 10.0f, 28.0f, (8.0f / 3.0f) };
    bool PARAM_RANGING[]{ true, false, false };
    float PARAM_STEPS[]{ 1.0f, 1.0f, 0.0f };
    float PARAM_MAX[]{ 19.0f, 40.0f, 0.0f };
    int PARAM_STEP_COUNTS[]{ 0, 0, 0 };

    const char* MAP_NAMES[]{ "test" };
    int MAP_X[]{ kernel::x };
    int MAP_Y[]{ kernel::y };

    bool executeOnLaunch = true;
    int steps = 1000;
    float stepSize = 0.01f;
    bool onlyShowLast = false;
}

__global__ void kernelProgram(float* data, float* params, float* maps, PreRanging* ranging, int steps, float h, int variationSize, float* previousData)
{
    int b = blockIdx.x;                                     // Current block of THREADS_PER_BLOCK threads
    int t = threadIdx.x;                                    // Current thread in the block, from 0 to THREADS_PER_BLOCK-1
    int variation = (b * THREADS_PER_BLOCK) + t;            // Variation (parameter combination) index
    if (variation >= ranging->totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    int variationStart = variation * variationSize;         // Start index to store the modelling data for the variation
    int stepStart = variationStart;                         // Start index for the current modelling step

    float varValues[kernel::VAR_COUNT];
    float paramValues[kernel::PARAM_COUNT];

    for (int i = 0; i < kernel::VAR_COUNT; i++) varValues[i] = !ranging->continuation ? data[i] : previousData[variationStart + (steps * kernel::VAR_COUNT) + i];
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
            if (!ranging->continuation) varValues[ranging->rangings[i].index] = value;
        }
        else
        {
            paramValues[ranging->rangings[i].index - kernel::VAR_COUNT] = value;
        }
    }

    V(x) = V0(x);
    V(y) = V0(y);
    V(z) = V0(z);

    for (int i = 0; i < steps; i++)
    {
        stepStart = variationStart + i * NEXT;

        float dx = P(sigma) * (V(y) - V(x));
        float dy = V(x) * (P(rho) - V(z)) - V(y);
        float dz = V(x) * V(y) - P(beta) * V(z);

        V(x + NEXT) = V(x) + h * dx;
        V(y + NEXT) = V(y) + h * dy;
        V(z + NEXT) = V(z) + h * dz;
    }

    M(variation) = V0(x);
}