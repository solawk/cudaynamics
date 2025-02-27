#include "main.h"
#ifdef SEL_MRLCs

#include "MRLCs-JJ.h"

namespace kernel
{
    const char* name = "MRLCs-JJ";

    const char* VAR_NAMES[]{ "sin_x1", "x1", "V", "IL", "theta" };
    numb VAR_VALUES[]{ 0.0f, -0.31f, 0.0f, 3.3f, 0.76f };
    RangingType VAR_RANGING[]{ None, None, None, None, None };
    numb VAR_STEPS[]{ 0.0f, 0.02f, 0.0f, 0.1f, 0.1f };
    numb VAR_MAX[]{ 0.0f, -0.01f, 0.0f, 4.3f, 1.76f };
    int VAR_STEP_COUNTS[]{ 0, 0, 0, 0 };

    const char* PARAM_NAMES[]{ "betaL", "betaC", "betaM", "i", "eps", "delta", "s" };
    numb PARAM_VALUES[]{ 50.0f, 0.5f, 1.0f, 0.9f, -2.0f, 1.0f, 0.5f };
    RangingType PARAM_RANGING[]{ None, None, None, Linear, Linear, None, None };
    numb PARAM_STEPS[]{ 1.0f, 1.0f, 0.0f, 0.01f, 0.01f, 0.0f, 0.0f };
    numb PARAM_MAX[]{ 19.0f, 40.0f, 0.0f, 1.5f, 0.0f, 0.0f, 0.0f };
    int PARAM_STEP_COUNTS[]{ 0, 0, 0, 0, 0, 0, 0 };

    const char* MAP_NAMES[]{ "LLE" };
    MapData MAP_DATA[]{ { 0, 0, 0, kernel::p3, MapDimensionType::PARAMETER, kernel::p4, MapDimensionType::PARAMETER } };

    const char* ANALYSIS_FEATURES[]{ "blank" };
    bool ANALYSIS_ENABLED[]{ false };

    bool executeOnLaunch = true;
    int steps = 5000;
    numb stepSize = 0.1f;
    bool onlyShowLast = false;
}

__global__ void kernelProgram(numb* data, numb* params, numb* maps, MapData* mapData, PreRanging* ranging, int steps, numb h, int variationSize, numb* previousData)
{
    int b = blockIdx.x;                                     // Current block of THREADS_PER_BLOCK threads
    int t = threadIdx.x;                                    // Current thread in the block, from 0 to THREADS_PER_BLOCK-1
    int variation = (b * THREADS_PER_BLOCK) + t;            // Variation (parameter combination) index
    if (variation >= ranging->totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    int variationStart = variation * variationSize;         // Start index to store the modelling data for the variation
    int stepStart = variationStart;                         // Start index for the current modelling step

    int varStep[kernel::VAR_COUNT];
    int paramStep[kernel::PARAM_COUNT];

    for (int i = 0; i < kernel::VAR_COUNT; i++) varStep[i] = 0;
    for (int i = 0; i < kernel::PARAM_COUNT; i++) paramStep[i] = 0;

    numb varValues[kernel::VAR_COUNT];
    numb paramValues[kernel::PARAM_COUNT];

    // Copying initial values into the beginning of the 0th variation
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
        numb value = ranging->rangings[i].min + ranging->rangings[i].step * step;

        if (isVar)
        {
            if (!ranging->continuation) varValues[ranging->rangings[i].index] = value;
            varStep[ranging->rangings[i].index] = step;
        }
        else
        {
            paramValues[ranging->rangings[i].index - kernel::VAR_COUNT] = value;
            paramStep[ranging->rangings[i].index - kernel::VAR_COUNT] = step;
        }
    }

    // Custom area (usually) starts here
    
    // Copying initial state to other variations
    initV(x0);
    data[stepStart + kernel::sin_x0] = sinf(varValues[kernel::x0]);
    data[stepStart + kernel::x0true] = varValues[kernel::x0];
    initV(x1);
    initV(x2);

    for (int s = 0; s < steps; s++)
    {
        stepStart = variationStart + s * kernel::VAR_COUNT;

        finiteDifferenceScheme(&data[stepStart], &data[stepStart + kernel::VAR_COUNT], paramValues, h);
    }

    // Analysis

    LLE(steps, variationStart, data, paramValues, h, maps, mapData, varStep, paramStep, &finiteDifferenceScheme);
}

__device__ void finiteDifferenceScheme(numb* currentV, numb* nextV, numb* parameters, numb h)
{
    float hs = (1.0f - P(p6)) * h;

    Vnext(x0) = fmodf(V(x0) + hs * V(x1), 2.0f * 3.141592f);
    Vnext(sin_x0) = sinf(Vnext(x0));
    Vnext(x0true) = V(x0true) + hs * V(x1);
    Vnext(x2) = V(x2) + hs * ((1.0f / P(p0)) * (V(x1) - V(x2)));
    Vnext(x1) = V(x1) + hs * ((1.0f / P(p1)) * (P(p3) - P(p2) * (1 + P(p4) * cosf(Vnext(x0))) * V(x1) - sinf(Vnext(x0)) - P(p5) * Vnext(x2)));

    hs = P(p6) * h;
    V(x0) = Vnext(x0);
    V(sin_x0) = sinf(V(x0));
    V(x1) = Vnext(x1);
    V(x2) = Vnext(x2);

    Vnext(x0) = fmodf(V(x0) + hs * V(x1), 2.0f * 3.141592f);
    Vnext(sin_x0) = sinf(Vnext(x0));
    Vnext(x0true) = V(x0true) + hs * V(x1);
    Vnext(x2) = V(x2) + hs * (1.0f / P(p0)) * (V(x1) - V(x2)) / (1.0f + hs * (1.0f / P(p0)));
    Vnext(x1) = V(x1) + hs * ((1.0f / P(p1)) * (P(p3) - P(p2) * (1 + P(p4) * cosf(Vnext(x0))) * V(x1) - sinf(Vnext(x0)) - P(p5) * Vnext(x2)));
    Vnext(x1) = V(x1) + hs * ((1.0f / P(p1)) * (P(p3) - P(p2) * (1 + P(p4) * cosf(Vnext(x0))) * Vnext(x1) - sinf(Vnext(x0)) - P(p5) * Vnext(x2)));

}

#endif