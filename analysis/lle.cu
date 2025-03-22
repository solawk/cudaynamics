#include "lle.h"

__device__ void LLE(Computation* data, int variation, int mapX, int mapY, void(* finiteDifferenceScheme)(numb*, numb*, numb*, numb))
{
    MapData* mapData = &(CUDA_kernel.mapDatas[0]);
    int variationStart = variation * CUDA_marshal.variationSize;
    int stepStart = variationStart;

    numb LLE_array[MAX_ATTRIBUTES];
    numb LLE_array_temp[MAX_ATTRIBUTES];
    numb LLE_value = 0.0f;
    int LLE_div = 0;

    for (int i = 0; i < CUDA_kernel.VAR_COUNT; i++)
        LLE_array[i] = CUDA_marshal.trajectory[stepStart + i];

    numb r = 0.1f;
    int L = 50;
    LLE_array[2] += r; // TODO: Which var to deflect

    for (int s = 0; s < CUDA_kernel.steps; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;

        for (int i = 0; i < CUDA_kernel.VAR_COUNT; i++)
            LLE_array_temp[i] = LLE_array[i];

        finiteDifferenceScheme(LLE_array_temp, LLE_array, &(CUDA_marshal.parameterVariations[variation * CUDA_kernel.PARAM_COUNT]), CUDA_kernel.stepSize);

        if (s % L == 0)
        {
            // Calculate local LLE
            numb norm = NORM_3D(LLE_array[0], CUDA_marshal.trajectory[stepStart + 0],
                LLE_array[1], CUDA_marshal.trajectory[stepStart + 1],
                LLE_array[2], CUDA_marshal.trajectory[stepStart + 2]);
            numb growth = norm / r; // How many times the deflection has grown
            if (r == 0.0f) growth = 0.0f;
            LLE_div++;
            LLE_value += log(growth) / ((CUDA_kernel.steps + 1) * CUDA_kernel.stepSize);

            // Reset
            LLE_array[0] = CUDA_marshal.trajectory[stepStart + 0] + (LLE_array[0] - CUDA_marshal.trajectory[stepStart + 0]) / growth;
            LLE_array[1] = CUDA_marshal.trajectory[stepStart + 1] + (LLE_array[1] - CUDA_marshal.trajectory[stepStart + 1]) / growth;
            LLE_array[2] = CUDA_marshal.trajectory[stepStart + 2] + (LLE_array[2] - CUDA_marshal.trajectory[stepStart + 2]) / growth;
        }

        CUDA_marshal.maps[mapData->offset + mapY * mapData->xSize + mapX] = LLE_value / LLE_div;
        //CUDA_marshal.maps[mapX] = mapX;
        //CUDA_marshal.maps[0] = 2;
    }
}