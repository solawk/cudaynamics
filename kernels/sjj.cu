#include "cuda_runtime.h"
#include "cuda_macros.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <fstream>

#include "sjj.h"
#include <objects.h>
#include <chrono>
#include <wtypes.h>

namespace kernel
{
    const char* name = "Shunted Josephson Junction";

    const char* VAR_NAMES[]{ "sin_x1", "x1", "x2", "x3" };
    float VAR_VALUES[]{ 0.0f, -0.31f, 3.3f, 0.76f };
    bool VAR_RANGING[]{ false, true, true, true };
    float VAR_STEPS[]{ 0.0f, 0.02f, 0.1f, 0.1f };
    float VAR_MAX[]{ 0.0f, -0.01f, 4.3f, 1.76f };
    int VAR_STEP_COUNTS[]{ 0, 0, 0, 0 };

    const char* PARAM_NAMES[]{ "betaL", "betaC", "i", "Vg/IcRs", "Rn", "Rsg" };
    float PARAM_VALUES[]{ 29.215f, 0.707f, 1.25f, 6.9f, 0.367f, 0.0478f };
    bool PARAM_RANGING[]{ false, false, false, false, false, false };
    float PARAM_STEPS[]{ 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    float PARAM_MAX[]{ 19.0f, 40.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    int PARAM_STEP_COUNTS[]{ 0, 0, 0, 0, 0, 0 };

    const char* MAP_NAMES[]{ "blank" };
    int MAP_X[]{ 0 };
    int MAP_Y[]{ 0 };

    bool executeOnLaunch = true;
    int steps = 50000;
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

    // Copying initial state to other variations
    V(x0) = V0(x0);
    V(sin_x0) = sinf(V(x0));
    V(x1) = V0(x1);
    V(x2) = V0(x2);

    for (int s = 0; s < steps; s++)
    {
        stepStart = variationStart + s * NEXT;

        // x[0] = x[0] + h*( x[1] );
        // x[2] = x[2] + h * ((1 / p[0]) * (x[1] - x[2]));
        // x[1] = x[1] + h * ((1 / p[1]) * (p[2] - ((x[1] > p[3]) ? p[4] : p[5]) * x[1] - sin(x[0]) - x[2]));

        V(x0 + NEXT) = fmodf(V(x0) + h * V(x1), 2.0f * 3.141592f);
        V(sin_x0 + NEXT) = sinf(V(x0 + NEXT));
        V(x2 + NEXT) = V(x2) + h * (1.0f / P(p0)) * (V(x1) - V(x2));
        V(x1 + NEXT) = V(x1) + h * (1.0f / P(p1)) *
            (
                P(p2)
                - ((V(x1) > P(p3)) ? P(p4) : P(p5)) * V(x1)
                - sinf(V(x0 + NEXT))
                - V(x2 + NEXT)
                );
    }
}