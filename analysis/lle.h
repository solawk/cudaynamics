#pragma once
#include "cuda_runtime.h"
#include "../objects.h"

#include "cuda_macros.h"
#include "device_launch_parameters.h"

// TEMPORARY
#define SEL_LORENZ

#ifdef SEL_LORENZ
#include "../kernels/lorenz.h"
#endif

#ifdef SEL_RLC_SJJ
#include "../kernels/RLC-sJJ.h"
#endif

__device__ void LLE(int steps, int variationStart, numb* data, numb* paramValues, numb h, numb* maps, MapData* mapData,
	int* varStep, int* paramStep, void(* finiteDifferenceScheme)(numb*, numb*, numb*, numb));