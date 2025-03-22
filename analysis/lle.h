#pragma once
#include "../analysis.h"
#include "../computation_struct.h"
#include "../mapData_struct.h"

__device__ void LLE(Computation* data, int variation, int mapX, int mapY, void(*finiteDifferenceScheme)(numb*, numb*, numb*, numb));