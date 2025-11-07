#ifndef DBSCAN_H
#define DBSCAN_H

#include <vector>
#include <cmath>

#include "../analysis.h"
#include "../computation_struct.h"

using namespace std;

struct DBSCAN_Settings
{
    numb eps;			
    int analysedVariable;	// Variable to analyse
    numb CoefIntervals;
    numb CoefPeaks;
    numb maxAllowedValue;
    numb epsFXP;
    numb timeFractionFXP;
    numb peakThreshold;
    numb stepSize;


    __device__ DBSCAN_Settings(numb _eps, int _analysedVariable, numb _CoefIntervals, numb _CoefPeaks, numb _maxAllowedValue, numb _epsFXP, numb _timeFractionFXP, numb _peakThreshold, numb _stepSize)
    {
        CoefIntervals = _CoefIntervals;
        CoefPeaks = _CoefPeaks;
        eps = _eps;
        analysedVariable = _analysedVariable;
        maxAllowedValue = _maxAllowedValue;
        epsFXP = _epsFXP;
        timeFractionFXP = _timeFractionFXP;
        peakThreshold = _peakThreshold;
        stepSize = _stepSize;
    }

};


__device__ void Period(Computation* data, DBSCAN_Settings settings, int variation, void(*finiteDifferenceScheme)(numb*, numb*, numb*), int offset_period, int offset_meanPeak, int offset_meanInterval);

#endif // DBSCAN_H
