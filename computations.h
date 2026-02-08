#pragma once
#include "computation_struct.h"
#include "index.h"

extern Computation computations[2];
extern Computation computationHires;
extern AnalysisIndex hiresIndex;

void computationsInit();

void terminateComputationBuffers(bool isHires);

void prepareKernel();

void hiresComputationSetup();