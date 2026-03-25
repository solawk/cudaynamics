#include "cuda_runtime.h"
#include "cuda_macros.h"
#include "numb.h"
#include "computation_struct.h"
#include "analysisHeaders.h"
#include "analysisSettingsHeaders.h"
#include "perthread_struct.h"

__host__ __device__ void AnalysisLobby(Computation* data, void(*finiteDifferenceScheme)(numb*, numb*, numb*, PerThread*), uint64_t variation);