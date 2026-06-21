#pragma once

#include "numb.h"
#include <vector>
#include <cstdint>
#include <unordered_map>
#include <cmath>

void CalculateRecurrence(double* t, std::vector<int>& vars, int size, int steps, int decimation, double* output, double epsilon, int varCount, uint64_t variation);

void CalculateRecurrenceGlobal(double* t, std::vector<int>& vars, int size, int steps, int decimation, double* output, int varCount, uint64_t variation);

struct RQA
{
	double RR = 0.0;

	double DET = 0.0;
	double DIV = 0.0;
	double ENTR = 0.0;

	double LAM = 0.0;
	double TT = 0.0;

	double MeanDiagonalLength = 0.0;
	double DiagonalVariance = 0.0;

	int Lmax = 0;
};

RQA RecurrenceRQA(double* rec, int size, int lmin, int vmin);