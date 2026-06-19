#include "recurrenceCalculation.h"

void CalculateRecurrence(numb* t, std::vector<int> vars, int size, int steps, int decimation, bool* output, numb epsilon, int varCount, uint64_t variation)
{
	// Index of variable x value at step y
	// (vars[x] * steps) + y

	int varsCount = vars.size();
	uint64_t s1, s2, var, step1, step2;
	numb norm, val1, val2;
	for (s1 = 0; s1 < size; s1++)
	{
		step1 = s1 * decimation;

		for (s2 = s1; s2 < size; s2++)
		{
			step2 = s2 * decimation;
			norm = 0.0;

			for (var = 0; var < varsCount; var++)
			{
				val1 = t[variation * steps * varCount + step1 * varCount + vars[var]];
				val2 = t[variation * steps * varCount + step2 * varCount + vars[var]];
				norm += (val2 - val1) * (val2 - val1);
			}

			norm = sqrt(norm);
			if (norm < epsilon)
				output[s1 * size + s2] = output[s2 * size + s1] = true;
			else
				output[s1 * size + s2] = output[s2 * size + s1] = false;
		}
	}
}

void CalculateRecurrenceAnalog(numb* t, std::vector<int> vars, int size, int steps, int decimation, numb* output, int varCount, uint64_t variation)
{
	int varsCount = vars.size();
	uint64_t s1, s2, var, step1, step2;
	numb norm, val1, val2;
	for (s1 = 0; s1 < size; s1++)
	{
		step1 = s1 * decimation;

		for (s2 = s1; s2 < size; s2++)
		{
			step2 = s2 * decimation;
			norm = 0.0;

			for (var = 0; var < varsCount; var++)
			{
				val1 = t[variation * steps * varCount + step1 * varCount + vars[var]];
				val2 = t[variation * steps * varCount + step2 * varCount + vars[var]];
				norm += (val2 - val1) * (val2 - val1);
			}

			norm = sqrt(norm);
			output[s1 * size + s2] = output[s2 * size + s1] = norm;
		}
	}
}

double RecurrenceDET(bool* rec, int size, int lmin)
{
	uint64_t recurrencePoints = 0;   // denominator
	uint64_t diagonalPoints = 0;   // numerator

#define at(row, col)	rec[row * size + col]

	// Scan all diagonals except the main diagonal
	for (int64_t offset = -(int64_t)size + 1; offset <= (int64_t)size - 1; offset++)
	{
		if (offset == 0) continue;

		int row = (offset < 0) ? -offset : 0;
		int col = (offset > 0) ? offset : 0;

		int runLength = 0;

		while (row < size && col < size)
		{
			if (at(row, col))
			{
				++runLength;
				++recurrencePoints;
			}
			else
			{
				if (runLength >= lmin) diagonalPoints += runLength;
				runLength = 0;
			}

			++row;
			++col;
		}

		// Handle run reaching end of diagonal
		if (runLength >= lmin) diagonalPoints += runLength;
	}

	return recurrencePoints ? ((double)diagonalPoints / (double)recurrencePoints) : 0.0;

#undef at
}

RQA RecurrenceRQA(bool* rec, int size, int lmin, int vmin)
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
            if (at(row, col))
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
            if (at(row, col))
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

    if (RecurrencePoints)
    {
        result.DET =
            (double)DiagonalPoints /
            (double)RecurrencePoints;
    }

    if (result.Lmax)
    {
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

        result.ENTR = entropy;
    }

    if (DiagonalLines)
    {
        result.MeanDiagonalLength =
            (double)diagonalLengthSum /
            (double)DiagonalLines;
    }

    if (DiagonalLines)
    {
        double mean =
            (double)diagonalLengthSum /
            (double)DiagonalLines;

        double meanSq =
            (double)diagonalLengthSqSum /
            (double)DiagonalLines;

        result.DiagonalVariance =
            meanSq - mean * mean;
    }

    if (RecurrencePoints)
    {
        result.LAM =
            (double)VerticalPoints /
            (double)RecurrencePoints;
    }

    if (VerticalLines)
    {
        result.TT =
            (double)verticalLengthSum /
            (double)VerticalLines;
    }

    return result;
}