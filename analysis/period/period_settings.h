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

    Port periodicity, meanPeak, meanInterval;

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

        periodicity = meanPeak = meanInterval = Port();
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