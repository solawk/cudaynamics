#include "phaseVolume.h"

__host__ __device__ void PhaseVolume(Computation* data, uint64_t variation, void(*finiteDifferenceScheme)(numb*, numb*, numb*)) {
    uint64_t stepStart, variationStart = variation * CUDA_marshal.variationSize;
    LOCAL_BUFFERS;
    LOAD_ATTRIBUTES(true);
    if (data->isHires) TRANSIENT_SKIP_NEW(finiteDifferenceScheme);
    

	PV_Settings settings =  CUDA_kernel.analyses.PV;
    int ObsSteps = settings.ObsSteps;   //The amount of trajectory points in one sample

    int ObservationCount = data->marshal.variationSize / CUDA_kernel.VAR_COUNT / ObsSteps;  //The amount of samples
    numb VolumeSum = 0; //Summ of all sample volumes in variation
    // For each sample we go through every chosen variable find their min max in that sample, max - min gives us an approximate length of the variable in that sample, we multiply whese lengths and get the volume for one sample
    //  Add the volumes of each sample and divide by sample amount and we get a mean of sample volumes
        for (int i = 0; i < ObservationCount; i++) {
            numb Volume = 1;
            for (int v = 0; v < 4; v++) {
                if (settings.normVariables[v] == -1)break;
                numb min = INFINITY;
                numb max = -INFINITY;
                for (int step = 0; step < ObsSteps; step++) {
                    stepStart = variationStart + ObsSteps * i * CUDA_kernel.VAR_COUNT + step * CUDA_kernel.VAR_COUNT;
                    uint64_t s = ObsSteps * i * CUDA_kernel.VAR_COUNT + step * CUDA_kernel.VAR_COUNT;
                    NORMAL_STEP_IN_ANALYSIS_IF_HIRES;

                    numb value = !data->isHires ? CUDA_marshal.trajectory[variationStart + i * ObsSteps * CUDA_kernel.VAR_COUNT + step * CUDA_kernel.VAR_COUNT + settings.normVariables[v]] : variables[settings.normVariables[v]];
                    if (max < value)max = value;
                    if (min > value)min = value;
                    
                }
                numb Length = abs(max - min);
                Volume *= Length;
            }
            VolumeSum += Volume;
        }
    
    if (CUDA_kernel.analyses.PV.PV.used)
    {
        numb mapValue = VolumeSum / ObservationCount;
        if (CUDA_kernel.mapWeight == 0.0f)
        {
            CUDA_marshal.maps[indexPosition(settings.PV.offset, 0)] = (CUDA_marshal.maps[indexPosition(settings.PV.offset, 0)] * data->bufferNo + mapValue) / (data->bufferNo + 1);
        }
        else if (CUDA_kernel.mapWeight == 1.0f)
        {
            CUDA_marshal.maps[indexPosition(settings.PV.offset, 0)] = mapValue;
        }
        else
        {
            CUDA_marshal.maps[indexPosition(settings.PV.offset, 0)] = CUDA_marshal.maps[indexPosition(settings.PV.offset, 0)] * (1.0f - CUDA_kernel.mapWeight) + mapValue * CUDA_kernel.mapWeight;
        }
    }
}