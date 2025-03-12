#include "lle.h"

#ifdef SEL_LORENZ
__device__ void LLE(int steps, int variationStart, numb* data, numb* paramValues, numb h, numb* maps, MapData* mapData,
    int* varStep, int* paramStep, void(* finiteDifferenceScheme)(numb*, numb*, numb*, numb))
{
    int stepStart = variationStart;

    LLE_INIT(numb);
    LLE_FILL;
    numb r = 0.1f;
    int L = 50;
    LLE_DEFLECT(z, r);

    for (int s = 0; s < steps; s++)
    {
        stepStart = variationStart + s * kernel::VAR_COUNT;

        LLE_STORE_TO_TEMP;

        finiteDifferenceScheme(LLE_array_temp, LLE_array, paramValues, h);

        LLE_IF_MOD(s, L)
        {
            // Calculate local LLE
            numb norm = NORM_3D(LLE_V(x), dataV(x NEXT), LLE_V(y), dataV(y NEXT), LLE_V(z), dataV(z NEXT));
            numb growth = norm / r;
            if (r == 0.0f) growth = 0.0f;
            LLE_ADD(log(growth) / ((steps + 1) * h));

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
#endif

#ifdef SEL_LORENZ_VAR
__device__ void LLE(int steps, int variationStart, numb* data, numb* paramValues, numb h, numb* maps, MapData* mapData,
    int* varStep, int* paramStep, void(*finiteDifferenceScheme)(numb*, numb*, numb*, numb))
{
    int stepStart = variationStart;

    LLE_INIT(numb);
    LLE_FILL;
    numb r = 0.1f;
    int L = 50;
    LLE_DEFLECT(z, r);

    for (int s = 0; s < steps; s++)
    {
        stepStart = variationStart + s * kernel::VAR_COUNT;

        LLE_STORE_TO_TEMP;

        finiteDifferenceScheme(LLE_array_temp, LLE_array, paramValues, h);

        LLE_IF_MOD(s, L)
        {
            // Calculate local LLE
            numb norm = NORM_3D(LLE_V(x), dataV(x NEXT), LLE_V(y), dataV(y NEXT), LLE_V(z), dataV(z NEXT));
            numb growth = norm / r;
            if (r == 0.0f) growth = 0.0f;
            LLE_ADD(log(growth) / ((steps + 1) * h));

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
#endif

#ifdef SEL_WILSON
__device__ void LLE(int steps, int variationStart, numb* data, numb* paramValues, numb h, numb* maps, MapData* mapData,
    int* varStep, int* paramStep, void(*finiteDifferenceScheme)(numb*, numb*, numb*, numb))
{
    
}
#endif

#ifdef SEL_MRLCs
__device__ void LLE(int steps, int variationStart, numb* data, numb* paramValues, numb h, numb* maps, MapData* mapData,
    int* varStep, int* paramStep, void(*finiteDifferenceScheme)(numb*, numb*, numb*, numb))
{
    int stepStart = variationStart;

    LLE_INIT(numb);
    LLE_FILL;
    numb r = 0.001f;
    int L = 500;
    LLE_DEFLECT(x0, r);

    for (int s = 0; s < steps; s++)
    {
        stepStart = variationStart + s * kernel::VAR_COUNT;

        LLE_STORE_TO_TEMP;

        finiteDifferenceScheme(LLE_array_temp, LLE_array, paramValues, h);

        LLE_IF_MOD(s, L)
        {
            // Calculate local LLE
            numb norm = NORM_3D(LLE_V(x0), dataV(x0 NEXT), LLE_V(x1), dataV(x1 NEXT), LLE_V(x2), dataV(x2 NEXT));
            numb growth = norm / r;
            if (r == 0.0f) growth = 0.0f;
            LLE_ADD(log(growth) / ((steps + 1) * h));

            // Reset
            LLE_RETRACT(x0, growth);
            LLE_RETRACT(x1, growth);
            LLE_RETRACT(x2, growth);
        }

        int LLEx = mapData->typeX == STEP ? s : mapData->typeX == VARIABLE ? varStep[mapData->indexX] : paramStep[mapData->indexX];
        int LLEy = mapData->typeY == STEP ? s : mapData->typeY == VARIABLE ? varStep[mapData->indexY] : paramStep[mapData->indexY];
        M(kernel::LLE, LLEx, LLEy) = LLE_MEAN_RESULT;
    }
}
#endif