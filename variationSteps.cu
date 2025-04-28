#include "variationSteps.h"

__device__ void variation2Steps(int* variation, int* steps, Kernel* kernel)
{
    int totalAttributes = kernel->VAR_COUNT + kernel->PARAM_COUNT; for (int i = 0; i < kernel->VAR_COUNT + kernel->PARAM_COUNT; i++) steps[i] = 0;
    int localVariation = *variation;
    for (unsigned long long i = 0; i < localVariation; i++)
    {
        for (int j = totalAttributes - 1; j >= 0; j--)
        {
            steps[j]++;

            bool isParam = j >= kernel->VAR_COUNT;
            int stepCountOfAttribute = isParam ? 
                kernel->parameters[j - kernel->VAR_COUNT].TrueStepCount() :
                kernel->variables[j].TrueStepCount();

            if (steps[j] < stepCountOfAttribute) break;
            steps[j] = 0;
        }
    }
}

void steps2Variation(int* variation, int* steps, Kernel* kernel)
{
    *variation = 0;
    int attrStride = 1;
    for (int i = kernel->VAR_COUNT + kernel->PARAM_COUNT - 1; i >= 0; i--)
    {
        int stepCount = i >= kernel->VAR_COUNT ? (kernel->parameters[i - kernel->VAR_COUNT].TrueStepCount()) : (kernel->variables[i].TrueStepCount());

        if (stepCount == 0) continue;

        *variation += steps[i] * attrStride;
        attrStride *= stepCount;
    }
}