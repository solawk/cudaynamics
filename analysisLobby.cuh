#include "cuda_runtime.h"
#include "cuda_macros.h"
#include "numb.h"
#include "computation_struct.h"
#include "analysisHeaders.h"
#include "analysisSettingsHeaders.h"

__device__ void AnalysisLobby(Computation* data, void(*finiteDifferenceScheme)(numb*, numb*, numb*), int variation);