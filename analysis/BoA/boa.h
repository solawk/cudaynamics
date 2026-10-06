#pragma once

#include "../analysis.h"
#include "../computation_struct.h"
#include "boa_settings.h"

// Per-variation marker, called by AnalysisLobby after all feature analyses.
__host__ __device__ void BOA(Computation* data, uint64_t variation);

// Stage 2: exact DBSCAN in the normalized two-feature space.
void FinalizeBOAOpenMP(Computation* data);
cudaError_t FinalizeBOACUDA(Computation* data, numb* cudaMaps, uint64_t variations);

// Converts deterministic component roots to dense basin ids 1..K.
// Noise is -1 and an invalid feature pair is -2.
void CompactBOALabels(Computation* data);
