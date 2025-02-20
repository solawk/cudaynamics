#include "main.h"
#ifdef SEL_LORENZ

#include "cuda_runtime.h"
#include "cuda_macros.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <fstream>

#include "lorenz.h"
#include <objects.h>
#include <chrono>
#include <wtypes.h>

namespace kernel
{
    const char* name = "Lorenz system";

    const char* VAR_NAMES[]{ "x", "y", "z" };
    numb VAR_VALUES[]{ 10.0f, 10.0f, 10.0f };
    RangingType VAR_RANGING[]{ None, None, None };
    numb VAR_STEPS[]{ 0.1f, 1.0f, 1.0f };
    numb VAR_MAX[]{ 29.0f, 29.0f, 29.0f };
    int VAR_STEP_COUNTS[]{ 0, 0, 0 };

    const char* PARAM_NAMES[]{ "sigma", "rho", "beta" };
    numb PARAM_VALUES[]{ 0.0f, 20.0f, (8.0f / 3.0f) };
    RangingType PARAM_RANGING[]{ Linear, Linear, None };
    numb PARAM_STEPS[]{ 0.4f, 0.4f, 0.0f };
    numb PARAM_MAX[]{ 100.0f, 100.0f, 0.0f };
    int PARAM_STEP_COUNTS[]{ 0, 0, 0 };

    const char* MAP_NAMES[]{ "LLE" };
    MapData MAP_DATA[]{ { 0, 0, 0, kernel::sigma, MapDimensionType::PARAMETER, kernel::rho, MapDimensionType::PARAMETER } };

    const char* ANALYSIS_FEATURES[]{ "LLE" };
    bool ANALYSIS_ENABLED[]{ true };

    bool executeOnLaunch = true;
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
    initV(x);
    initV(y);
    initV(z);

    LLE_INIT(numb);
    LLE_FILL;
    numb deflection = 0.1f;
    int L = 50;
    LLE_DEFLECT(z, deflection);

    for (int s = 0; s < steps; s++)
    {
        stepStart = variationStart + s * kernel::VAR_COUNT;

        finiteDifferenceScheme(&data[stepStart], &data[stepStart + kernel::VAR_COUNT], paramValues, h);

        // LLE
        LLE_STORE_TO_TEMP;

        finiteDifferenceScheme(LLE_array_temp, LLE_array, paramValues, h);

        LLE_IF_MOD(s, L)
        {
            // Calculate local LLE
            numb norm = NORM_3D(LLE_V(x), dataV(x NEXT), LLE_V(y), dataV(y NEXT), LLE_V(z), dataV(z NEXT));
            numb growth = norm / deflection;
            if (deflection == 0.0f) growth = 0.0f;
            LLE_ADD(log(growth));

            // Reset
            LLE_RETRACT(x, growth);
            LLE_RETRACT(y, growth);
            LLE_RETRACT(z, growth);
        }

        int LLEx = mapData->typeX == STEP ? s : mapData->typeX == VARIABLE ? varStep[mapData->indexX] : paramStep[mapData->indexX];
        int LLEy = mapData->typeY == STEP ? s : mapData->typeY == VARIABLE ? varStep[mapData->indexY] : paramStep[mapData->indexY];
        M(kernel::LLE, LLEx, LLEy) = LLE_MEAN_RESULT;
    }
}

__device__ void finiteDifferenceScheme(numb* currentV, numb* nextV, numb* parameters, numb h)
{
    numb dx = P(sigma) * (V(y) - V(x));
    numb dy = V(x) * (P(rho) - V(z)) - V(y);
    numb dz = V(x) * V(y) - P(beta) * V(z);

    Vnext(x) = V(x) + h * dx;
    Vnext(y) = V(y) + h * dy;
    Vnext(z) = V(z) + h * dz;
}

#endif