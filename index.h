#pragma once
#include <string>
#include "anfuncs.h"

enum AnalysisIndex
{
	IND_MIN, IND_MAX,
	IND_LLE,
	IND_PERIOD, IND_MNPEAK, IND_MNINT
};

struct Index
{
	bool enabled; // userEnabled
	std::string name;
	AnalysisFunction function;
	unsigned int size; // previously valueCount
	float decayDeltaThreshold;

	Index()
	{
		enabled = false;
		name = "ERROR";
		function = ANF_MINMAX;
		size = 1;
		decayDeltaThreshold = 1.0f;
	}

	Index(std::string _name, AnalysisFunction _function, unsigned int _size, float _decay)
	{
		enabled = true;
		name = _name;
		function = _function;
		size = _size;
		decayDeltaThreshold = _decay;
	}
};