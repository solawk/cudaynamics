#include "period.h"

__device__  void Period(Computation* data, DBscan_Settings settings, int variation, void(*finiteDifferenceScheme)(numb*, numb*, numb*), int offset_period, int offset_meanPeak, int offset_meanInterval) {
    int variationStart = variation * CUDA_marshal.variationSize;
    int varCount = CUDA_kernel.VAR_COUNT;
    int variationSize = CUDA_marshal.variationSize;

    LOCAL_BUFFERS;
    LOAD_ATTRIBUTES(true);
    if (data->isHires) TRANSIENT_SKIP_NEW(finiteDifferenceScheme);

    int stepStart, s = -1;
    numb variablesPrev[MAX_ATTRIBUTES]{ 0 }, variablesCurr[MAX_ATTRIBUTES]{ 0 }; // variables will store "next" values, variablesPrev – "prev", variablesCurr – "curr"
    // When using hi-res (with no trajectory buffer available), one trajectory steps is precomputed, making first "prev" and "curr" values
    // Each next computed step will be a "next" one
    for (int v = 0; v < varCount; v++) variablesPrev[v] = variables[v];
    NORMAL_STEP_IN_ANALYSIS_IF_HIRES;
    for (int v = 0; v < varCount; v++) variablesCurr[v] = variables[v];

    // Map Settings for peak finder and DBSCAN analysis
    numb epsDBSCAN = settings.eps;  //eps value for dbscan
    int analysedVariable = settings.analysedVariable; //variable of which peak finder analyses trajectory
    numb coefPeaks = settings.CoefPeaks;    //coefficient for peaks found in peakfinder
    numb coefIntervals = settings.CoefIntervals; //coefficient for intervals found in peakfinder
    numb maxAllowedValue = settings.maxAllowedValue;    //the maximum value allowed before peak finder deems system dispersive
    numb epsFXP = settings.epsFXP;  //eps area used in checking if system is a fixed point
    numb peakThreshold = settings.peakThreshold;    //minimum value of peak that can be found in peak finder, -inf by default
    numb stepSize = settings.stepSize == -1 ? 1 : settings.stepSize;    //stepsize of system
    numb timeFractionFXP = settings.timeFractionFXP;    //fraction of the trajectory that the system need to b e fixed point for peak finder to deem it a fixed point
    int fixedPointMaxCount = round((variationSize / varCount) * timeFractionFXP);   //amount of steps in trajectory that the system need to be fixed point for peak finder to deem it a fixed point

    // Buffer to hold peak data (amplitudes and indices)
    constexpr int MAX_PEAKS = 128;
    numb peakAmplitudes[MAX_PEAKS];
    numb peakIntervals[MAX_PEAKS];

    bool returnNan = false, returnZero = false, WritingData = true; //flags for if the system is dispersive, is a fixed point or if peakfinder has filled buffer and wont write any new peaks
   
    //temp data used in analysis
    int peakCount = 0; 
    numb tempPeakAmp = 0, tempPeakTime = 0; bool tempPeakFound = false; // used in case if peak finder finds a horizontal line of equal values and doesnt know if there is a peak there until the line ends, while the line is being analysed the first value of the line is save into tempPeakAmp and tempPeakTime
    bool firstpeakreached = false; // flag for the first peak in trajectory which we dont save into data and use just for its interval with the next peak
    numb temppeakindex;     // used to save the time of last peak in trajectory
    int fixedPointCount = 0;    // used to count how many continuous values in trajectory fulfiil the epsFXP requirement
    numb* computedVariation = CUDA_marshal.trajectory + variationStart;

    numb sumPeak = 0, sumInterval = 0; // the sums of counted peaks and intervals for meanPeak and meanInterval analysis
    
    //  Peak finder
    for (int i = 1; i < (variationSize / varCount) - 1 /* && peakCount < MAX_PEAKS*/; i++)
    {
        NORMAL_STEP_IN_ANALYSIS_IF_HIRES;

        numb prev, curr, next;

        if (!data->isHires)
        {
            prev = computedVariation[analysedVariable + varCount * i - varCount];
            curr = computedVariation[analysedVariable + varCount * i];
            next = computedVariation[analysedVariable + varCount * i + varCount];
        }
        else
        {
            prev = variablesPrev[analysedVariable];
            curr = variablesCurr[analysedVariable];
            next = variablesNext[analysedVariable];

            for (int v = 0; v < varCount; v++)
            {
                variablesPrev[v] = variablesCurr[v];
                variablesCurr[v] = variables[v];
            }
        }
        
        if (abs(next - curr) / stepSize < epsFXP) // check the derivative for fixed point requirement
        { 
            fixedPointCount++; 
            if (fixedPointCount > fixedPointMaxCount) 
            { 
                returnZero = true;  
            } 
        }
        else 
        { 
            fixedPointCount = 0;  
        }

        
        if (abs(curr) > maxAllowedValue) { //check if value is too big to be a dispesive system
            returnNan = true;
        }
        else if(curr>peakThreshold) { 
            if (curr > prev && curr > next)     //peak found
            {

                tempPeakFound = false;
                if (firstpeakreached == false)
                {
                    firstpeakreached = true;
                    temppeakindex = (float)i;
                }
                else
                {
                    if (WritingData) {
                        peakAmplitudes[peakCount] = curr;
                        peakIntervals[peakCount] = (i - temppeakindex) * stepSize;
                        sumPeak += peakAmplitudes[peakCount];
                        sumInterval += peakIntervals[peakCount];
                        peakCount++;
                    }
                   
                    temppeakindex = (float)i;
                }
            }
            else if (curr == next && curr > prev) { // found a possible peak solved as a line by finiteDifferenceScheme
                tempPeakFound = true; tempPeakAmp = curr; tempPeakTime = i;
            }
            else if (curr < next) {    // case in which the line was likely not a peak
                tempPeakFound = false;
            }
            else if (curr > next && tempPeakFound) {    // the line value is larger than values before and after the line, which means first value of line is taken as peak
                if (firstpeakreached) {
                    if (WritingData) 
                    { 
                        peakAmplitudes[peakCount] = tempPeakAmp;  peakIntervals[peakCount] = (tempPeakTime - temppeakindex) * stepSize; 
                        sumPeak += peakAmplitudes[peakCount]; sumInterval += peakIntervals[peakCount];
                        peakCount++;
                    }
                    
                    temppeakindex = (float)tempPeakTime;
                    tempPeakFound = false;
                }
                else {
                    firstpeakreached = true;
                    temppeakindex = (float)tempPeakTime;
                    tempPeakFound = false;
                }
            }
            if (peakCount >= MAX_PEAKS - 1) WritingData = false; // case for filled buffer
        }
    }
    if (peakCount == 0)returnZero = true;
    //
    numb mapValue;
    
    //if (data->marshal.kernel.mapDatas[offset_meanPeak/data->marshal.totalVariations].toCompute) {
    if (data->marshal.kernel.mapDatas[3].toCompute) {
        mapValue = sumPeak / peakCount;
        if (CUDA_kernel.mapWeight == 0.0f)
        {
            numb existingValue = CUDA_marshal.maps[variation + offset_meanPeak] * data->bufferNo;
            CUDA_marshal.maps[variation + offset_meanPeak] = (existingValue + mapValue) / (data->bufferNo + 1);
        }
        else if (CUDA_kernel.mapWeight == 1.0f)
        {
            //CUDA_marshal.maps[(variation + offset_meanPeak) + ( 0 *CUDA_marshal.totalVariations)] = mapValue;
            CUDA_marshal.maps[(variation + offset_meanPeak) + (data->isHires ? 3 : 0 * CUDA_marshal.totalVariations)] = mapValue;
        }
        else
        {
            CUDA_marshal.maps[variation + offset_meanPeak] = CUDA_marshal.maps[variation + offset_meanPeak] * (1.0f - CUDA_kernel.mapWeight) + mapValue * CUDA_kernel.mapWeight;
        }
    }
    //if (data->marshal.kernel.mapDatas[offset_meanInterval/data->marshal.totalVariations].toCompute) {
    if (data->marshal.kernel.mapDatas[2].toCompute) {
        mapValue = sumInterval / peakCount;
        if (CUDA_kernel.mapWeight == 0.0f)
        {
            numb existingValue = CUDA_marshal.maps[variation + offset_meanInterval] * data->bufferNo;
            CUDA_marshal.maps[variation + offset_meanInterval] = (existingValue + mapValue) / (data->bufferNo + 1);
        }
        else if (CUDA_kernel.mapWeight == 1.0f)
        {
            //CUDA_marshal.maps[(variation + offset_meanInterval) + ( 0 *CUDA_marshal.totalVariations)] = mapValue;
            CUDA_marshal.maps[(variation + offset_meanInterval) + (data->isHires ? 2 : 0 *CUDA_marshal.totalVariations)] = mapValue;
        }
        else
        {
            CUDA_marshal.maps[variation + offset_meanInterval] = CUDA_marshal.maps[variation + offset_meanInterval] * (1.0f - CUDA_kernel.mapWeight) + mapValue * CUDA_kernel.mapWeight;
        }
    }
    
    
    
    //if (data->marshal.kernel.mapDatas[offset_meanPeak/data->marshal.totalVariations].toCompute) {
    if (data->marshal.kernel.mapDatas[4].toCompute) {

        if (returnZero) { mapValue = 0; }   // result if FXP system
        else if (returnNan) { mapValue = NAN; } // result if dispersive system
        else {
            for (int i = 0; i < peakCount - 1; i++) {   // normalization of intervals and peak values for dbscan
                peakIntervals[i] *= coefIntervals; peakAmplitudes[i] *= coefPeaks;
            }

            int cluster = 0;
            int NumNeibor = 0;
            int helpfulArraySize = 2 * MAX_PEAKS;
            int helpfulArray[2 * MAX_PEAKS];
            for (int i = 0; i < helpfulArraySize; ++i) {
                helpfulArray[i] = 0;
            }

            //      DBSCAN
            for (int i = 0; i < peakCount; i++)
                if (NumNeibor >= 1)
                {
                    i = helpfulArray[peakCount + NumNeibor - 1];
                    helpfulArray[peakCount + NumNeibor - 1] = 0;
                    NumNeibor = NumNeibor - 1;
                    for (int k = 0; k < peakCount - 1; k++) {
                        if (i != k && helpfulArray[k] == 0) {
                            if (sqrt(pow(peakAmplitudes[i] - peakAmplitudes[k], 2) + pow(peakIntervals[i] - peakIntervals[k], 2)) <= epsDBSCAN) {
                                helpfulArray[k] = cluster;
                                helpfulArray[peakCount + k] = k;
                                ++NumNeibor;
                            }
                        }

                    }
                }
                else if (helpfulArray[i] == 0) {
                    NumNeibor = 0;
                    ++cluster;
                    helpfulArray[i] = cluster;
                    for (int k = 0; k < peakCount - 1; k++) {
                        if (i != k && helpfulArray[peakCount + k] == 0) {
                            if (sqrt(pow(peakAmplitudes[i] - peakAmplitudes[k], 2) + pow(peakIntervals[i] - peakIntervals[k], 2)) <= epsDBSCAN) {
                                helpfulArray[k] = cluster;
                                helpfulArray[peakCount + k] = k;
                                ++NumNeibor;
                            }
                        }

                    }
                }
            cluster--;

            //

            mapValue = cluster;
        }

        if (CUDA_kernel.mapWeight == 0.0f)
        {
            numb existingValue = CUDA_marshal.maps[variation + offset_period] * data->bufferNo;
            CUDA_marshal.maps[variation + offset_period] = (existingValue + mapValue) / (data->bufferNo + 1);
        }
        else if (CUDA_kernel.mapWeight == 1.0f)
        {
            //CUDA_marshal.maps[(variation + offset_period) + ( 0 * CUDA_marshal.totalVariations)] = mapValue;
            CUDA_marshal.maps[(variation + offset_period) + (data->isHires ? 4 : 0 * CUDA_marshal.totalVariations)] = mapValue;
        }
        else
        {
            CUDA_marshal.maps[variation + offset_period] = CUDA_marshal.maps[variation + offset_period] * (1.0f - CUDA_kernel.mapWeight) + mapValue * CUDA_kernel.mapWeight;
        }
    }
    else  CUDA_marshal.maps[(variation + offset_period) + (0 * CUDA_marshal.totalVariations)] = data->marshal.totalVariations;
}




