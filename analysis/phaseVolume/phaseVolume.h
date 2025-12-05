#pragma once
#include "../analysis.h"
#include "../computation_struct.h"
#include "phaseVolume_settings.h"

__host__ __device__ void PhaseVolume(Computation* data, uint64_t variation, void(*finiteDifferenceScheme)(numb*, numb*, numb*));