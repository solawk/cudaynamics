#pragma once

#include <vector>
#include <cmath>

#include "../port.h"
#include "../numb.h"
#include "abstractSettings_struct.h"

struct DBSCAN_Settings : AbstractAnalysisSettingsStruct
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

    DBSCAN_Settings() { }

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