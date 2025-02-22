#include "main.h"
#ifdef SEL_RLC_SJJ

#include "cuda_runtime.h"
#include "cuda_macros.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <fstream>

#include "RLC-sJJ.h"
#include <objects.h>
#include <chrono>
#include <wtypes.h>

namespace kernel
{
    const char* name = "Shunted Josephson Junction";

    const char* VAR_NAMES[]{ "sin_x1", "x1", "x2", "x3" };
    numb VAR_VALUES[]{ 0.0f, -0.31f, 3.3f, 0.76f };
    bool VAR_RANGING[]{ false, false, false, false };
    numb VAR_STEPS[]{ 0.0f, 0.02f, 0.1f, 0.1f };
    numb VAR_MAX[]{ 0.0f, -0.01f, 4.3f, 1.76f };
    int VAR_STEP_COUNTS[]{ 0, 0, 0, 0 };

    const char* PARAM_NAMES[]{ "betaL", "betaC", "i", "Vg/IcRs", "Rn", "Rsg" };
    numb PARAM_VALUES[]{ 2.0f, 0.5f, 1.25f, 6.9f, 0.367f, 0.0478f }; // 29.215f, 0.707f, 1.25f, 6.9f, 0.367f, 0.0478f
    bool PARAM_RANGING[]{ true, true, false, false, false, false };
    numb PARAM_STEPS[]{ 0.01f, 0.01f, 0.0f, 0.0f, 0.0f, 0.0f };
    numb PARAM_MAX[]{ 3.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    int PARAM_STEP_COUNTS[]{ 0, 0, 0, 0, 0, 0 };

    const char* MAP_NAMES[]{ "LLE" };
    MapData MAP_DATA[]{ { 0, 0, 0, kernel::p0, MapDimensionType::PARAMETER, kernel::p1, MapDimensionType::PARAMETER } };

    const char* ANALYSIS_FEATURES[]{ "LLE" };
    bool ANALYSIS_ENABLED[]{ true };

    bool executeOnLaunch = true;
    int steps = 400;
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
    dataV(sin_x0) = sinf(dataV(x0));
    initV(x1);
    initV(x2);

    LLE_INIT(numb);
    LLE_FILL;
    numb deflection = 0.0001f;
    int L = 10;
    LLE_DEFLECT(x2, deflection);

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
            numb norm = NORM_3D(LLE_V(sin_x0), dataV(sin_x0 NEXT), LLE_V(x1), dataV(x1 NEXT), LLE_V(x2), dataV(x2 NEXT));
            LLE_ADD(log(norm / deflection));

            // Reset
            LLE_RETRACT(x0, (norm / deflection));
            LLE_RETRACT(x1, (norm / deflection));
            LLE_RETRACT(x2, (norm / deflection));
        }

        int LLEx = mapData->typeX == STEP ? s : mapData->typeX == VARIABLE ? varStep[mapData->indexX] : paramStep[mapData->indexX];
        int LLEy = mapData->typeY == STEP ? s : mapData->typeY == VARIABLE ? varStep[mapData->indexY] : paramStep[mapData->indexY];
        M(kernel::LLE, LLEx, LLEy) = LLE_MEAN_RESULT;
    }
}

__device__ void finiteDifferenceScheme(numb* currentV, numb* nextV, numb* parameters, numb h)
{
    Vnext(x0) = fmodf(V(x0) + h * V(x1), 2.0f * 3.141592f);
    Vnext(sin_x0) = sinf(Vnext(x0));
    Vnext(x2) = V(x2) + h * (1.0f / P(p0)) * (V(x1) - V(x2));
    Vnext(x1) = V(x1) + h * (1.0f / P(p1)) *
        (
            P(p2)
            - ((V(x1) > P(p3)) ? P(p4) : P(p5)) * V(x1)
            - sinf(Vnext(x0))
            - Vnext(x2)
            );
}

#endif