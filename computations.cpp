#include "computations.h"

Computation computations[2];
Computation computationHires;
AnalysisIndex hiresIndex = IND_NONE;

void computationsInit()
{
    computations[0].marshal.trajectory = computations[1].marshal.trajectory = nullptr;
    computations[0].marshal.parameterVariations = computations[1].marshal.parameterVariations = nullptr;
    computations[0].isHires = computations[1].isHires = false;
    computationHires.isHires = true;
    computationHires.isGPU = true;
    computations[0].index = 0;
    computations[1].index = 1;
    computations[0].otherMarshal = &(computations[1].marshal);
    computations[1].otherMarshal = &(computations[0].marshal);
}

void terminateComputationBuffers(bool isHires)
{
    if (!isHires)
    {
        if (computations[0].future.valid()) computations[0].future.wait();
        if (computations[1].future.valid()) computations[1].future.wait();
        computations[0].Clear();
        computations[1].Clear();
    }
    else
    {
        if (computationHires.future.valid()) computationHires.future.wait();
        computationHires.Clear();
    }
}

// prepareKernel handles switching to the selected kernel
// initializeKernel now only manages the UI side
void prepareKernel()
{
    terminateComputationBuffers(false);
    terminateComputationBuffers(true);
    hiresIndex = IND_NONE;
}

void hiresComputationSetup()
{
    computationHires.ready = false;
    computationHires.isFirst = true;
    computationHires.mapIndex = hiresIndex;
    computationHires.marshal.kernel.mapWeight = 0.0f;
    computationHires.threadsPerBlock = 32;
}