#pragma once
#include "cuda_runtime.h"
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

    DBSCAN_Settings()
    {
        eps = (numb)1.0;
        analysedVariable = 2;
        CoefIntervals = (numb)15.0;
        CoefPeaks = (numb)5.0;
        maxAllowedValue = (numb)1e4;
        epsFXP = (numb)1e-3;
        timeFractionFXP = (numb)0.05;
        peakThreshold = (numb)-INFINITY;
        stepSize = (numb)0.01;
    }

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

    void DisplaySettings()
    {
        DisplayNumbSetting("Epsilon", eps);
        DisplayVarSetting("Analyzed variable", analysedVariable);
        DisplayNumbSetting("CoefIntervals", CoefIntervals);
        DisplayNumbSetting("CoefPeaks", CoefPeaks);
        DisplayNumbSetting("maxAllowedValue", maxAllowedValue);
        DisplayNumbSetting("epsFXP", epsFXP);
        DisplayNumbSetting("timeFractionFXP", timeFractionFXP);
        DisplayNumbSetting("peakThreshold", peakThreshold);
        DisplayNumbSetting("stepSize", stepSize);
    }

};