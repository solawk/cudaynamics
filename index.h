#pragma once
#include <string>
#include "anfuncs.h"
#include "decaySettings_struct.h"

enum AnalysisIndex
{
	IND_MIN, IND_MAX,
	IND_LLE,
	IND_PERIOD, IND_MNMPEAK, IND_MNMINT, IND_MNPEAK, IND_MNINT, IND_MXMPEAK, IND_MXMINT,
	IND_PV,
	IND_COUNT, IND_NONE
};

struct Index
{
	bool enabled; // userEnabled
	std::string name;
	AnalysisFunction function;
	unsigned int size; // previously valueCount
	DecaySettings decay;

	Index()
	{
		enabled = false;
		name = "ERROR";
		function = ANF_MINMAX;
		size = 1;
	}

	Index(std::string _name, AnalysisFunction _function, unsigned int _size)
	{
		enabled = true;
		name = _name;
		function = _function;
		size = _size;
	}
};