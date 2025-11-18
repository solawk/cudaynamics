#pragma once
#include "main.h"
#include "cuda_macros.h"
#include <omp.h>

void AnalysisLobby_cpu(Computation* data, void(*finiteDifferenceScheme)(numb*, numb*, numb*), int variation);

void cpu_execute(Computation* data, bool openmp);