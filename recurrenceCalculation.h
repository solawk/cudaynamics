#pragma once

#include "numb.h"
#include <vector>
#include <cstdint>
#include <unordered_map>
#include <cmath>

void CalculateRecurrence(numb* t, std::vector<int> vars, int size, int steps, int decimation, bool* output, numb epsilon, int varCount, uint64_t variation);

void CalculateRecurrenceAnalog(numb* t, std::vector<int> vars, int size, int steps, int decimation, numb* output, int varCount, uint64_t variation);

double RecurrenceDET(bool* rec, int size, int lmin = 2);

struct RQA
{
	double DET = 0.0;
	double DIV = 0.0;
	double ENTR = 0.0;

	double LAM = 0.0;
	double TT = 0.0;

	double MeanDiagonalLength = 0.0;
	double DiagonalVariance = 0.0;

	int Lmax = 0;
};

RQA RecurrenceRQA(bool* rec, int size, int lmin, int vmin);