#include "recurrenceCalculation.h"

void CalculateRecurrence(numb* t1, numb* t2, int windows, int windowIndex, std::vector<int>& vars, int size, int steps, int decimation, double* output, double epsilon, int varCount, uint64_t variation, bool isGlobal)
{
	// variation * steps * varCount  – finds variation
    // step1 * varCount              – finds step
    // vars[var]                     – finds variable

    int windowOffsetPerIndex = (int)((double)size / windows);
    int windowOffset = windowIndex * windowOffsetPerIndex;

	int varsCount = (int)vars.size();
	uint64_t s1, s2, var, step1, step2, transstep1, transstep2;
    // Transsteps are actual steps that can go over the size to reach the next buffer
    double norm, val1, val2;

	for (s1 = 0; s1 < size; s1++)
	{
		step1 = s1 * decimation;

		for (s2 = s1; s2 < size; s2++)
		{
			step2 = s2 * decimation;
			norm = 0.0;

			for (var = 0; var < varsCount; var++)
			{
                transstep1 = step1 + windowOffset;
                if (transstep1 < size)
                    val1 = t1[variation * steps * varCount + transstep1 * varCount + vars[var]];
                else
                    val1 = t2[variation * steps * varCount + (transstep1 - size) * varCount + vars[var]];


                transstep2 = step2 + windowOffset;
                if (transstep2 < size)
                    val2 = t1[variation * steps * varCount + transstep2 * varCount + vars[var]];
                else
                    val2 = t2[variation * steps * varCount + (transstep2 - size) * varCount + vars[var]];

				norm += (val2 - val1) * (val2 - val1);
			}

			norm = sqrt(norm);

            if (!isGlobal)
            {
                if (norm < epsilon)
                    output[s1 * size + s2] = output[s2 * size + s1] = 1.0;
                else
                    output[s1 * size + s2] = output[s2 * size + s1] = 0.0;
            }
            else
                output[s1 * size + s2] = output[s2 * size + s1] = norm;
		}
	}
}

void CalculateRecurrenceSpecific(numb* t1, numb* t2, int windows, int windowIndex, std::vector<int>& vars, std::vector<int>& stepsToUse1, std::vector<int>& stepsToUse2, int steps, int decimation, double* output, double epsilon, int varCount, uint64_t variation, bool isGlobal)
{
    int varsCount = (int)vars.size();
    int stepsToUseCount1 = (int)stepsToUse1.size();
    int stepsToUseCount2 = (int)stepsToUse2.size();

    int windowOffsetPerIndex = (int)((double)stepsToUseCount1 / windows);
    int windowOffset = windowIndex * windowOffsetPerIndex;
    // Window size is interpolated between peak counts in two buffers
    int windowSize = SpecificWindowSize(stepsToUseCount1, stepsToUseCount2, (double)windowIndex / windows);

    uint64_t s1, s2, var, step1, step2, transstep1, transstep2;
    // Transsteps are indeces of steps from stepsToUse. They go up to ...Count1+...Count2
    double norm, val1, val2;

    for (s1 = 0; s1 < windowSize; s1++)
    {
        transstep1 = s1 + windowOffset;
        if (transstep1 < stepsToUseCount1)
            step1 = stepsToUse1[transstep1];
        else
            step1 = stepsToUse2[transstep1 - stepsToUseCount1];

        for (s2 = s1; s2 < windowSize; s2++)
        {
            transstep2 = s2 + windowOffset;
            if (transstep2 < stepsToUseCount1)
                step2 = stepsToUse1[transstep2];
            else
                step2 = stepsToUse2[transstep2 - stepsToUseCount1];

            norm = 0.0;

            for (var = 0; var < varsCount; var++)
            {
                if (transstep1 < windowSize)
                    val1 = t1[variation * steps * varCount + step1 * varCount + vars[var]];
                else
                    val1 = t2[variation * steps * varCount + step1 * varCount + vars[var]];

                if (transstep2 < windowSize)
                    val2 = t1[variation * steps * varCount + step2 * varCount + vars[var]];
                else
                    val2 = t2[variation * steps * varCount + step2 * varCount + vars[var]];

                norm += (val2 - val1) * (val2 - val1);
            }

            norm = sqrt(norm);

            if (!isGlobal)
            {
                if (norm < epsilon)
                    output[s1 * windowSize + s2] = output[s2 * windowSize + s1] = 1.0;
                else
                    output[s1 * windowSize + s2] = output[s2 * windowSize + s1] = 0.0;
            }
            else
                output[s1 * windowSize + s2] = output[s2 * windowSize + s1] = norm;
        }
    }
}

RQA RecurrenceRQA(double* rec, int size, int lmin, int vmin)
{
    RQA result;

    uint64_t RecurrencePoints = 0;
    uint64_t DiagonalPoints = 0;
    uint64_t VerticalPoints = 0;

    uint64_t DiagonalLines = 0;
    uint64_t VerticalLines = 0;

    std::unordered_map<int, uint64_t> diagonalHistogram;

    uint64_t diagonalLengthSum = 0;
    uint64_t diagonalLengthSqSum = 0;

#define at(r,c) rec[(r) * size + (c)]

    //--------------------------------------------------
    // Diagonal scan:
    // DET, DIV, ENTR, MeanDiagonalLength, Variance
    //--------------------------------------------------

    for (int64_t offset = -(int64_t)size + 1;
        offset <= (int64_t)size - 1;
        ++offset)
    {
        if (offset == 0)
            continue;

        int row = (offset < 0) ? (int)(-offset) : 0;
        int col = (offset > 0) ? (int)offset : 0;

        int runLength = 0;

        while (row < size && col < size)
        {
            if (at(row, col) > 0.0)
            {
                ++runLength;
                ++RecurrencePoints;
            }
            else
            {
                if (runLength > 0)
                {
                    if (runLength > result.Lmax)
                        result.Lmax = runLength;

                    if (runLength >= lmin)
                    {
                        DiagonalPoints += runLength;

                        ++DiagonalLines;

                        diagonalLengthSum += runLength;
                        diagonalLengthSqSum += (uint64_t)runLength * (uint64_t)runLength;

                        ++diagonalHistogram[runLength];
                    }

                    runLength = 0;
                }
            }

            ++row;
            ++col;
        }

        if (runLength > 0)
        {
            if (runLength > result.Lmax)
                result.Lmax = runLength;

            if (runLength >= lmin)
            {
                DiagonalPoints += runLength;

                ++DiagonalLines;

                diagonalLengthSum += runLength;
                diagonalLengthSqSum += (uint64_t)runLength * (uint64_t)runLength;

                ++diagonalHistogram[runLength];
            }
        }
    }

    //--------------------------------------------------
    // Vertical scan:
    // LAM, TT
    //--------------------------------------------------

    uint64_t verticalLengthSum = 0;

    for (int col = 0; col < size; ++col)
    {
        int runLength = 0;

        for (int row = 0; row < size; ++row)
        {
            if (at(row, col) > 0.0)
            {
                ++runLength;
            }
            else
            {
                if (runLength > 0)
                {
                    if (runLength >= vmin)
                    {
                        VerticalPoints += runLength;

                        verticalLengthSum += runLength;

                        ++VerticalLines;
                    }

                    runLength = 0;
                }
            }
        }

        if (runLength >= vmin)
        {
            VerticalPoints += runLength;

            verticalLengthSum += runLength;

            ++VerticalLines;
        }
    }

#undef at

    const uint64_t totalCells =
        (uint64_t)size * (uint64_t)size - size;


    // Ratio'd
    result.Lmax /= (double)size;

    // Ratio by default
    result.RR = totalCells ? (double)RecurrencePoints / (double)totalCells : 0.0;

    if (RecurrencePoints)
    {
        // Ratio by default
        result.DET =
            (double)DiagonalPoints /
            (double)RecurrencePoints;
    }

    if (result.Lmax)
    {
        // Ratio' by default'd by ratioing Lmax
        result.DIV =
            1.0 / (double)result.Lmax;
    }

    if (DiagonalLines)
    {
        double entropy = 0.0;

        for (const auto& kv : diagonalHistogram)
        {
            double p =
                (double)kv.second /
                (double)DiagonalLines;

            entropy -= p * std::log(p);
        }

        // Ratio by default
        result.ENTR = entropy;
    }

    if (DiagonalLines)
    {
        // Ratio'd
        result.MeanDiagonalLength = ((double)diagonalLengthSum / (double)DiagonalLines) / (double)size;
    }

    if (DiagonalLines)
    {
        double mean =
            (double)diagonalLengthSum /
            (double)DiagonalLines;

        double meanSq =
            (double)diagonalLengthSqSum /
            (double)DiagonalLines;

        // Ratio by default
        result.DiagonalVariance =
            meanSq - mean * mean;
    }

    if (RecurrencePoints)
    {
        // Ratio by default
        result.LAM =
            (double)VerticalPoints /
            (double)RecurrencePoints;
    }

    if (VerticalLines)
    {
        // Ratio'd
        result.TT = ((double)verticalLengthSum / (double)VerticalLines) / (double)size;
    }

    return result;
}

int SpecificWindowSize(int size1, int size2, float t)
{
    return (int)roundf(size1 + t * (size2 - size1));
}