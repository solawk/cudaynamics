#pragma once
#include "main.h"
#include "cuda_macros.h"
#include <omp.h>

void cpu_execute(Computation* data, bool openmp);