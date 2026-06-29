#pragma once

#include "numb.h"
#include <vector>
#include <cstdint>
#include <unordered_map>
#include <cmath>

void CalculateRecurrence(numb* t1, numb* t2, int windows, int windowIndex, std::vector<int>& vars, int size, 
	int steps, int decimation, double* output, double epsilon, int varCount, uint64_t variation, bool isGlobal);
void CalculateRecurrenceSpecific(numb* t1, numb* t2, int windows, int windowIndex, std::vector<int>& vars, std::vector<int>& stepsToUse1, std::vector<int>& stepsToUse2,
	int steps, int decimation, double* output, double epsilon, int varCount, uint64_t variation, bool isGlobal);

int SpecificWindowSize(int size1, int size2, float t);

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

	double Lmax = 0;

	void Add(RQA& from)
	{
		RR += from.RR;
		DET += from.DET;
		DIV += from.DIV;
		ENTR += from.ENTR;
		LAM += from.LAM;
		TT += from.TT;
		MeanDiagonalLength += from.MeanDiagonalLength;
		DiagonalVariance += from.DiagonalVariance;
		Lmax += from.Lmax;
	}

	void Div(double divider)
	{
		RR /= divider;
		DET /= divider;
		DIV /= divider;
		ENTR /= divider;
		LAM /= divider;
		TT /= divider;
		MeanDiagonalLength /= divider;
		DiagonalVariance /= divider;
		Lmax /= divider;
	}
};

RQA RecurrenceRQA(double* rec, int size, int lmin, int vmin);