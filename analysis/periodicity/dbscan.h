#ifndef DBSCAN_H
#define DBSCAN_H

#include <vector>
#include <cmath>

#include "../analysis.h"
#include "../computation_struct.h"
#include "../mapData_struct.h"

using namespace std;



struct DBscan_Settings
{
    numb eps;			
    int analysedVariable;	// Variable to analyse
    numb CoefIntervals;
    numb CoefPeaks;
    numb maxAllowedValue;
    numb epsFXP;
    numb timeFractionEXP;
    numb peakThreshold;
    numb stepSize;


    __device__ DBscan_Settings(numb _eps, int _analysedVariable, numb _CoefIntervals, numb _CoefPeaks, numb _maxAllowedValue, numb _epsFXP, numb _timeFractionEXP, numb _peakThreshold, numb _stepSize)
    {
        CoefIntervals = _CoefIntervals;
        CoefPeaks = _CoefPeaks;
        eps = _eps;
        analysedVariable = _analysedVariable;
        maxAllowedValue = _maxAllowedValue;
        epsFXP = _epsFXP;
        timeFractionEXP = _timeFractionEXP;
        peakThreshold = _peakThreshold;
        stepSize = _stepSize;
    }

};


__device__ void Period(Computation* data, DBscan_Settings settings, int variation, void(*finiteDifferenceScheme)(numb*, numb*, numb*), int offset);

#endif // DBSCAN_H
