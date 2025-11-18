#include "cpu_bm.h"

namespace attributes
{
    enum variables { x, y, z };
    enum parameters { sigma, rho, beta, symmetry, method, stepsize };
    enum methods { ExplicitEuler, ExplicitMidpoint, ExplicitRungeKutta4, VariableSymmetryCD };
}

void finiteDifferenceScheme_lorenz_cpu(numb* currentV, numb* nextV, numb* parameters)
{
    ifMETHOD(P(method), ExplicitEuler)
    {
        Vnext(x) = V(x) + P(stepsize) * (P(sigma) * (V(y) - V(x)));
        Vnext(y) = V(y) + P(stepsize) * (V(x) * (P(rho) - V(z)) - V(y));
        Vnext(z) = V(z) + P(stepsize) * (V(x) * V(y) - P(beta) * V(z));
    }

    ifMETHOD(P(method), ExplicitMidpoint)
    {
        numb xmp = V(x) + P(stepsize) * 0.5f * (P(sigma) * (V(y) - V(x)));
        numb ymp = V(y) + P(stepsize) * 0.5f * (V(x) * (P(rho) - V(z)) - V(y));
        numb zmp = V(z) + P(stepsize) * 0.5f * (V(x) * V(y) - P(beta) * V(z));

        Vnext(x) = V(x) + P(stepsize) * (P(sigma) * (ymp - xmp));
        Vnext(y) = V(y) + P(stepsize) * (xmp * (P(rho) - zmp) - ymp);
        Vnext(z) = V(z) + P(stepsize) * (xmp * ymp - P(beta) * zmp);
    }

    ifMETHOD(P(method), ExplicitRungeKutta4)
    {
        numb kx1 = P(sigma) * (V(y) - V(x));
        numb ky1 = V(x) * (P(rho) - V(z)) - V(y);
        numb kz1 = V(x) * V(y) - P(beta) * V(z);

        numb xmp = V(x) + 0.5f * P(stepsize) * kx1;
        numb ymp = V(y) + 0.5f * P(stepsize) * ky1;
        numb zmp = V(z) + 0.5f * P(stepsize) * kz1;

        numb kx2 = P(sigma) * (ymp - xmp);
        numb ky2 = xmp * (P(rho) - zmp) - ymp;
        numb kz2 = xmp * ymp - P(beta) * zmp;

        xmp = V(x) + 0.5f * P(stepsize) * kx2;
        ymp = V(y) + 0.5f * P(stepsize) * ky2;
        zmp = V(z) + 0.5f * P(stepsize) * kz2;

        numb kx3 = P(sigma) * (ymp - xmp);
        numb ky3 = xmp * (P(rho) - zmp) - ymp;
        numb kz3 = xmp * ymp - P(beta) * zmp;

        xmp = V(x) + P(stepsize) * kx3;
        ymp = V(y) + P(stepsize) * ky3;
        zmp = V(z) + P(stepsize) * kz3;

        numb kx4 = P(sigma) * (ymp - xmp);
        numb ky4 = xmp * (P(rho) - zmp) - ymp;
        numb kz4 = xmp * ymp - P(beta) * zmp;

        Vnext(x) = V(x) + P(stepsize) * (kx1 + 2.0f * kx2 + 2.0f * kx3 + kx4) / 6.0f;
        Vnext(y) = V(y) + P(stepsize) * (ky1 + 2.0f * ky2 + 2.0f * ky3 + ky4) / 6.0f;
        Vnext(z) = V(z) + P(stepsize) * (kz1 + 2.0f * kz2 + 2.0f * kz3 + kz4) / 6.0f;
    }

    ifMETHOD(P(method), VariableSymmetryCD)
    {
        numb h1 = 0.5f * P(stepsize) - P(symmetry);
        numb h2 = 0.5f * P(stepsize) + P(symmetry);

        numb xmp = V(x) + h1 * (P(sigma) * (V(y) - V(x)));
        numb ymp = V(y) + h1 * (V(x) * (P(rho) - V(z)) - V(y));
        numb zmp = V(z) + h1 * (V(x) * V(y) - P(beta) * V(z));

        Vnext(z) = (zmp + xmp * ymp * h2) / (1.0f + P(beta) * h2);
        Vnext(y) = (ymp + xmp * (P(rho) - Vnext(z)) * h2) / (1.0f + h2);
        Vnext(x) = (xmp + P(sigma) * Vnext(y) * h2) / (1.0f + P(sigma) * h2);
    }
}

void LLE_cpu(Computation* data, int variation, void(*finiteDifferenceScheme)(numb*, numb*, numb*))
{
    int stepStart, variationStart = variation * CUDA_marshal.variationSize;
    LOCAL_BUFFERS;
    LOAD_ATTRIBUTES(true);
    if (data->isHires) TRANSIENT_SKIP_NEW(finiteDifferenceScheme);

    numb LLE_array[MAX_ATTRIBUTES]{ 0 }; // The deflected trajectory
    numb LLE_array_next[MAX_ATTRIBUTES]; // Buffer for the next step of the deflected trajectory
    numb LLE_value = 0.0f;

    for (int i = 0; i < CUDA_kernel.VAR_COUNT; i++)
        if (!data->isHires)     LLE_array[i] = CUDA_marshal.trajectory[variationStart + i];
        else                    LLE_array[i] = variables[i];

    LLE_Settings settings = CUDA_kernel.analyses.LLE;
    numb r = settings.r;
    int L = settings.L;
    LLE_array[settings.variableToDeflect] += r;
    int stepParamIndex = CUDA_kernel.PARAM_COUNT - 1;

    for (int s = 0; s < CUDA_kernel.steps; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;

        // Normal step, if hi-res
        NORMAL_STEP_IN_ANALYSIS_IF_HIRES;

        // Deflected step
        finiteDifferenceScheme(LLE_array, LLE_array_next, &(parameters[0]));

        for (int i = 0; i < CUDA_kernel.VAR_COUNT; i++)
            LLE_array[i] = LLE_array_next[i];

        // LLE calculations
        if ((s + 1) % L == 0)
        {
            numb norm = 0.0;
            for (int i = 0; i < 4; i++)
            {
                if (settings.normVariables[i] == -1) break;

                numb x1 = LLE_array[settings.normVariables[i]];
                numb x2 = !data->isHires ? CUDA_marshal.trajectory[stepStart + CUDA_kernel.VAR_COUNT + settings.normVariables[i]] : variables[settings.normVariables[i]];
                norm += (x2 - x1) * (x2 - x1);
            }

            norm = sqrt(norm);

            numb growth = norm / r; // How many times the deflection has grown
            if (growth > 0.0f)
                LLE_value += log(growth) / H_BRANCH(
                    parameters[stepParamIndex],
                    !data->isHires ? CUDA_marshal.trajectory[stepStart + CUDA_kernel.VAR_COUNT - 1] : variables[CUDA_kernel.VAR_COUNT - 1]
                );

            // Reset
            for (int i = 0; i < CUDA_kernel.VAR_COUNT; i++)
                if (!data->isHires) LLE_array[i] = CUDA_marshal.trajectory[stepStart + CUDA_kernel.VAR_COUNT + i] + (LLE_array[i] - CUDA_marshal.trajectory[stepStart + CUDA_kernel.VAR_COUNT + i]) / growth;
                else LLE_array[i] = variables[i] + (LLE_array[i] - variables[i]) / growth;
        }
    }

    if (CUDA_kernel.analyses.LLE.LLE.used)
    {
        numb mapValue = LLE_value / (CUDA_kernel.steps + 1);
        if (CUDA_kernel.mapWeight == 0.0f)
        {
            CUDA_marshal.maps[indexPosition(settings.LLE.offset, 0)] = (CUDA_marshal.maps[indexPosition(settings.LLE.offset, 0)] * data->bufferNo + mapValue) / (data->bufferNo + 1);
        }
        else if (CUDA_kernel.mapWeight == 1.0f)
        {
            CUDA_marshal.maps[indexPosition(settings.LLE.offset, 0)] = mapValue;
        }
        else
        {
            CUDA_marshal.maps[indexPosition(settings.LLE.offset, 0)] = CUDA_marshal.maps[indexPosition(settings.LLE.offset, 0)] * (1.0f - CUDA_kernel.mapWeight) + mapValue * CUDA_kernel.mapWeight;
        }
    }
}

void MAX_cpu(Computation* data, int variation, void(*finiteDifferenceScheme)(numb*, numb*, numb*))
{
    int stepStart, variationStart = variation * CUDA_marshal.variationSize;
    LOCAL_BUFFERS;
    LOAD_ATTRIBUTES(true);
    if (data->isHires) TRANSIENT_SKIP_NEW(finiteDifferenceScheme);

    MINMAX_Settings settings = CUDA_kernel.analyses.MINMAX;
    int minvar = settings.minVariableIndex;
    int maxvar = settings.maxVariableIndex;
    numb minValue = 0.0, maxValue = 0.0;
    numb prevMin, prevMax;

    for (int s = 0; s < CUDA_kernel.steps; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;

        NORMAL_STEP_IN_ANALYSIS_IF_HIRES;

        if (CUDA_kernel.analyses.MINMAX.maximum.used)
        {
            prevMax = !data->isHires ? CUDA_marshal.trajectory[stepStart + maxvar] : variables[maxvar];
            if (s == 0 || maxValue < prevMax) maxValue = prevMax;
        }

        if (CUDA_kernel.analyses.MINMAX.minimum.used)
        {
            prevMin = !data->isHires ? CUDA_marshal.trajectory[stepStart + minvar] : variables[minvar];
            if (s == 0 || minValue > prevMin) minValue = prevMin;
        }
    }

    if (CUDA_kernel.mapWeight == 1.0f)
    {
        if (CUDA_kernel.analyses.MINMAX.maximum.used) CUDA_marshal.maps[indexPosition(settings.maximum.offset, 0)] = maxValue;
        if (CUDA_kernel.analyses.MINMAX.minimum.used) CUDA_marshal.maps[indexPosition(settings.minimum.offset, 0)] = minValue;
    }
    else
    {
        if (CUDA_kernel.analyses.MINMAX.maximum.used)
        {
            prevMax = CUDA_marshal.maps[indexPosition(settings.maximum.offset, 0)];
            CUDA_marshal.maps[indexPosition(settings.maximum.offset, 0)] = maxValue > prevMax ? maxValue : prevMax;
        }

        if (CUDA_kernel.analyses.MINMAX.minimum.used)
        {
            prevMin = CUDA_marshal.maps[indexPosition(settings.minimum.offset, 0)];
            CUDA_marshal.maps[indexPosition(settings.minimum.offset, 0)] = minValue < prevMin ? minValue : prevMin;
        }
    }
}

void Period_cpu(Computation* data, int variation, void(*finiteDifferenceScheme)(numb*, numb*, numb*))
{
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
    DBSCAN_Settings settings = CUDA_kernel.analyses.PERIOD;
    numb epsDBSCAN = settings.eps;  //eps value for dbscan
    int analysedVariable = settings.analysedVariable; //variable of which peak finder analyses trajectory
    numb coefPeaks = settings.CoefPeaks;    //coefficient for peaks found in peakfinder
    numb coefIntervals = settings.CoefIntervals; //coefficient for intervals found in peakfinder
    numb maxAllowedValue = settings.maxAllowedValue;    //the maximum value allowed before peak finder deems system dispersive
    numb epsFXP = settings.epsFXP;  //eps area used in checking if system is a fixed point
    numb peakThreshold = settings.peakThreshold;    //minimum value of peak that can be found in peak finder, -inf by default
    numb stepSize = CUDA_kernel.stepType == ST_Parameter ? parameters[CUDA_kernel.PARAM_COUNT - 1] : 1;    //stepsize of system
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
        else if (curr > peakThreshold) {
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

    if (CUDA_kernel.analyses.PERIOD.meanPeak.used)
    {
        mapValue = sumPeak / peakCount;
        if (CUDA_kernel.mapWeight == 0.0f)
        {
            numb existingValue = CUDA_marshal.maps[indexPosition(settings.meanPeak.offset, 0)] * data->bufferNo;
            CUDA_marshal.maps[indexPosition(settings.meanPeak.offset, 0)] = (existingValue + mapValue) / (data->bufferNo + 1);
        }
        else if (CUDA_kernel.mapWeight == 1.0f)
        {
            CUDA_marshal.maps[indexPosition(settings.meanPeak.offset, 0)] = mapValue;
        }
        else
        {
            CUDA_marshal.maps[indexPosition(settings.meanPeak.offset, 0)]
                = CUDA_marshal.maps[indexPosition(settings.meanPeak.offset, 0)] * (1.0f - CUDA_kernel.mapWeight) + mapValue * CUDA_kernel.mapWeight;
        }
    }

    if (CUDA_kernel.analyses.PERIOD.meanInterval.used)
    {
        mapValue = sumInterval / peakCount;
        if (CUDA_kernel.mapWeight == 0.0f)
        {
            numb existingValue = CUDA_marshal.maps[indexPosition(settings.meanInterval.offset, 0)] * data->bufferNo;
            CUDA_marshal.maps[indexPosition(settings.meanInterval.offset, 0)] = (existingValue + mapValue) / (data->bufferNo + 1);
        }
        else if (CUDA_kernel.mapWeight == 1.0f)
        {
            CUDA_marshal.maps[indexPosition(settings.meanInterval.offset, 0)] = mapValue;
        }
        else
        {
            CUDA_marshal.maps[indexPosition(settings.meanInterval.offset, 0)]
                = CUDA_marshal.maps[indexPosition(settings.meanInterval.offset, 0)] * (1.0f - CUDA_kernel.mapWeight) + mapValue * CUDA_kernel.mapWeight;
        }
    }

    if (CUDA_kernel.analyses.PERIOD.periodicity.used)
    {
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
            numb existingValue = CUDA_marshal.maps[indexPosition(settings.periodicity.offset, 0)] * data->bufferNo;
            CUDA_marshal.maps[indexPosition(settings.periodicity.offset, 0)] = (existingValue + mapValue) / (data->bufferNo + 1);
        }
        else if (CUDA_kernel.mapWeight == 1.0f)
        {
            CUDA_marshal.maps[indexPosition(settings.periodicity.offset, 0)] = mapValue;
        }
        else
        {
            CUDA_marshal.maps[indexPosition(settings.periodicity.offset, 0)]
                = CUDA_marshal.maps[indexPosition(settings.periodicity.offset, 0)] * (1.0f - CUDA_kernel.mapWeight) + mapValue * CUDA_kernel.mapWeight;
        }
    }
}

void lorenz_cpu(Computation* data, int variation)
{
    int variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
    int stepStart;                         // Start index for the current modelling step
    if (variation >= CUDA_marshal.totalVariations) return;
    LOCAL_BUFFERS;
    LOAD_ATTRIBUTES(false);

    // Custom area (usually) starts here

    //TRANSIENT_SKIP_NEW(finiteDifferenceScheme_lorenz_cpu);

    for (int s = 0; s < CUDA_kernel.steps; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;
        finiteDifferenceScheme_lorenz_cpu(FDS_ARGUMENTS);
        RECORD_STEP;
    }

    AnalysisLobby_cpu(data, &finiteDifferenceScheme_lorenz_cpu, variation);
}

void AnalysisLobby_cpu(Computation* data, void(*finiteDifferenceScheme)(numb*, numb*, numb*), int variation)
{
    if (CUDA_kernel.analyses.LLE.toCompute)
    {
        LLE_cpu(data, variation, finiteDifferenceScheme);
    }

    if (CUDA_kernel.analyses.MINMAX.toCompute)
    {
        MAX_cpu(data, variation, finiteDifferenceScheme);
    }

    if (CUDA_kernel.analyses.PERIOD.toCompute)
    {
        Period_cpu(data, variation, finiteDifferenceScheme);
    }
}

void cpu_execute(Computation* data, bool openmp)
{
	int variations = CUDA_marshal.totalVariations;

    if (!openmp)
    {
        for (int v = 0; v < variations; v++)
        {
            lorenz_cpu(data, v);
        }
    }
    else
    {
        if (!data->isHires)
        {
#pragma omp parallel for
            for (int v = 0; v < variations; v++)
            {
                lorenz_cpu(data, v);
            }
        }
        else
        {
#pragma omp parallel for
            for (int v = 0; v < data->variationsInCurrentExecute; v++)
            {
                lorenz_cpu(data, v);
            }
        }
    }
}