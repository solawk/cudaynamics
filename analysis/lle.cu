#include "lle.h"

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