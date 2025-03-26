#include "lle.h"

__device__ void LLE(Computation* data, LLE_Settings settings, int variation, int mapX, int mapY, void(* finiteDifferenceScheme)(numb*, numb*, numb*, numb))
{
    MapData* mapData = &(CUDA_kernel.mapDatas[0]);
    int variationStart = variation * CUDA_marshal.variationSize;
    int stepStart = variationStart;

    numb LLE_array[MAX_ATTRIBUTES];
    numb LLE_array_temp[MAX_ATTRIBUTES];
    numb LLE_value = 0.0f;

    for (int i = 0; i < CUDA_kernel.VAR_COUNT; i++)
        LLE_array[i] = CUDA_marshal.trajectory[stepStart + i];

    numb r = settings.r;
    numb epsilon = settings.epsilon;
    LLE_array[settings.variableToDeflect] += r;

    for (int s = 0; s < CUDA_kernel.steps; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;

        for (int i = 0; i < CUDA_kernel.VAR_COUNT; i++)
            LLE_array_temp[i] = LLE_array[i];

        finiteDifferenceScheme(LLE_array_temp, LLE_array, &(CUDA_marshal.parameterVariations[variation * CUDA_kernel.PARAM_COUNT]), CUDA_kernel.stepSize);

        // TODO: Rework into any dimension count up to MAX_LLE_NORM_VARIABLES
        numb norm;
        if (settings.normVariables[1] == -1)
        {
            // 1D
            norm = NORM_3D(
                LLE_array[0], CUDA_marshal.trajectory[stepStart + 0],
                LLE_array[1], CUDA_marshal.trajectory[stepStart + 1],
                LLE_array[2], CUDA_marshal.trajectory[stepStart + 2]
            );
        }
        else if (settings.normVariables[2] == -1)
        {
            // 2D
            norm = NORM_3D(
                LLE_array[0], CUDA_marshal.trajectory[stepStart + 0],
                LLE_array[1], CUDA_marshal.trajectory[stepStart + 1],
                LLE_array[2], CUDA_marshal.trajectory[stepStart + 2]
            );
        }
        else
        {
            // 3D
            norm = NORM_3D(
                LLE_array[0], CUDA_marshal.trajectory[stepStart + 0],
                LLE_array[1], CUDA_marshal.trajectory[stepStart + 1],
                LLE_array[2], CUDA_marshal.trajectory[stepStart + 2]
            );
        }

        numb growth = norm / r; // How many times the deflection has grown
        if (growth > 0.0f) LLE_value += log(growth);

        if (norm >= epsilon)
        {
            // Reset
            LLE_array[0] = CUDA_marshal.trajectory[stepStart + 0] + (LLE_array[0] - CUDA_marshal.trajectory[stepStart + 0]) / growth;
            LLE_array[1] = CUDA_marshal.trajectory[stepStart + 1] + (LLE_array[1] - CUDA_marshal.trajectory[stepStart + 1]) / growth;
            LLE_array[2] = CUDA_marshal.trajectory[stepStart + 2] + (LLE_array[2] - CUDA_marshal.trajectory[stepStart + 2]) / growth;
        }
    }

    CUDA_marshal.maps[mapData->offset + mapY * mapData->xSize + mapX] = LLE_value / CUDA_kernel.steps;
}