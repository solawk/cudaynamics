#include "main.h"
#ifdef SEL_WILSON

#include "wilson.h"

namespace kernel
{
    const char* name = "Wilson neuron";

    const char* VAR_NAMES[]{ "V", "R", "I", "t" };
    numb VAR_VALUES[]{ 0.0f, 0.0f, 0.0f, 0.0f };
    RangingType VAR_RANGING[]{ None, None, None, None };
    numb VAR_STEPS[]{ 0.1f, 1.0f, 0.1f, 0.1f };
    numb VAR_MAX[]{ 29.0f, 29.0f, 2.0f, 0.1f };
    int VAR_STEP_COUNTS[]{ 0, 0, 0, 0 };

    const char* PARAM_NAMES[]{ "C", "tau", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "Imax", "Iduty" };
    numb PARAM_VALUES[]{ 0.8f, 1.9f, 0.4f, 2.17f, 32.63f, 1.25f, -0.22f, 26.0f, 1.35f, 0.0874f, 0.5f, 5.0f };
    RangingType PARAM_RANGING[]{ None, None, None, None, None, None, None, None, None, None, None, None };
    numb PARAM_STEPS[]{ 0.4f, 0.4f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    numb PARAM_MAX[]{ 100.0f, 100.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    int PARAM_STEP_COUNTS[]{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

    const char* MAP_NAMES[]{ "LLE" };
    MapData MAP_DATA[]{ { 0, 0, 0, kernel::C, MapDimensionType::PARAMETER, kernel::tau, MapDimensionType::PARAMETER } };

    const char* ANALYSIS_FEATURES[]{ "LLE" };
    bool ANALYSIS_ENABLED[]{ true };

    bool executeOnLaunch = false;
    int steps = 1000;
    numb stepSize = 0.01f;
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
    initV(v);
    initV(R);
    initV(I);
    initV(T);

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
    Vnext(I) = fmodf(V(T), P(Idc)) < (0.5f * P(Idc)) ? P(Imax) : 0.0f;
    Vnext(T) = V(T) + h;

    numb dv = (-(P(p0)+P(p1)*V(v) + P(p2)*V(v)*V(v))*(V(v) - P(p3)) - P(p5)*V(R)*(V(v)-P(p4)) + Vnext(I)) / P(C);
    numb dr = (1 / P(tau)) * (-V(R) + P(p6) * V(v) + P(p7));

    //y[0] = x[0] + h*( (-(p[3]+p[4]*x[0]+p[5]*x[0]**2)*(x[0]-p[6])-p[8]*x[1]*(x[0]-p[7]) + sl)/p[1] );
    //y[1] = x[1] + h * ((1 / p[2]) * (-x[1] + p[9] * x[0] + p[10]));

    Vnext(v) = V(v) + h * dv;
    Vnext(R) = V(R) + h * dr;
}

#endif