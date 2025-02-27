#pragma once
#include "../analysis.h"

// TEMPORARY
//#define SEL_LORENZ
//#define SEL_MRLCs

#ifdef SEL_LORENZ
#include "../kernels/lorenz.h"
#endif

#ifdef SEL_RLC_SJJ
#include "../kernels/RLC-sJJ.h"
#endif

#ifdef SEL_MRLCs
#include "../kernels/MRLCs-JJ.h"
#endif

__device__ void LLE(int steps, int variationStart, numb* data, numb* paramValues, numb h, numb* maps, MapData* mapData,
	int* varStep, int* paramStep, void(* finiteDifferenceScheme)(numb*, numb*, numb*, numb));