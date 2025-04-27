#pragma once

#define numb float
#define MAX_ATTRIBUTES 128
#define MAX_MAPS 16

#include <string>
#include <vector>
#include <set>

#include "imgui/backends/imgui_impl_win32.h"
#include "imgui/backends/imgui_impl_dx11.h"
#include "implot/implot.h"
#include "quaternion.h"
#include "plotWindow.h"

#define calculateStepCount(_min, _max, _step) (_step != 0 ? (int)((_max - _min) / _step) + 1 : 0)
#define stepFromStepCount(_min, _max, _stepCount) ((_max - _min) / (numb)(_stepCount - 1))
#define calculateValue(_min, _step, _currentStep) ((_min) + (_step) * (_currentStep))
// Calculate step from value, floored int
#define stepFromValue(_min, _step, _value) (int)((_value - _min) / _step)

// None - no ranging, only the min value
// Linear - fixed step values from min (inclusive) to max (not necessarily inclusive)
// UniformRandom - random values from min to max with uniform distribution, step = quantity
// NormalRandom - random values from min to max with normal distribution around midpoint, step = quantity
enum RangingType { None, Step, Linear, UniformRandom, NormalRandom };

struct SinglePreRangingInfo
{
public:
	int index;
	numb min;
	numb step;
	numb max;
	int steps;

	void init(int _i, numb _min, numb _step, numb _max, int _steps)
	{
		index = _i;
		min = _min;
		step = _step;
		max = _max;
		steps = _steps;
	}
};

struct PreRanging
{
public:
	int varCount;
	int paramCount;
	int rangingCount;
	SinglePreRangingInfo rangings[MAX_VARS_PARAMS]{ 0 };
	int totalVariations;
	bool continuation; // For first batch – false (forming initial values from ranging data), for next batches – true (initial values are pre-formed from previous final values)

	PreRanging(int _v, int _p)
	{
		varCount = _v;
		paramCount = _p;
	}

	void setRangingAndVariations(int _r, int _t)
	{
		rangingCount = _r;
		totalVariations = _t;
	}
};

struct PostRanging
{
public:
	int rangingCount;		// Amount of ranging variables/parameters
	int totalVariations;	// Ranging var/param combinations count

	std::vector<std::string> names;
	std::vector<numb> min;
	std::vector<numb> step;
	std::vector<numb> max;

	std::vector<int> stepCount;
	std::vector<int> currentStep;

	std::vector<numb> currentValue;

	float timeElapsed;

	PostRanging()
	{
		clear();
	}

	void clear()
	{
		rangingCount = 0;
		totalVariations = 0;
		timeElapsed = 0.0f;

		names.clear();
		min.clear();
		step.clear();
		max.clear();
		stepCount.clear();
		//currentStep.clear();
		currentValue.clear();
	}

	int indexOfRangingEntity(std::string name)
	{
		for (int i = 0; i < rangingCount; i++)
		{
			if (names[i] == name) return i;
		}

		return -1;
	}

	void getIndexOfVarOrParam(bool* isParam, int* entityIndex, int varCount, int paramCount, const char** varNames, const char** paramNames, int rangingIndex)
	{
		for (int i = 0; i < varCount; i++)
		{
			if (std::string(varNames[i]) == names[rangingIndex])
			{
				*isParam = false;
				*entityIndex = i;
				return;
			}
		}

		for (int i = 0; i < paramCount; i++)
		{
			if (std::string(paramNames[i]) == names[rangingIndex])
			{
				*isParam = true;
				*entityIndex = i;
				return;
			}
		}
	}
};

struct PlotGraphSettings
{
public:
	bool isEnabled;
	float markerSize;
	float markerOutlineSize;
	ImVec4 markerColor;
	ImPlotMarker markerShape;
	float gridAlpha;

	PlotGraphSettings()
	{
		isEnabled = true;
		markerSize = 1.0f;
		markerOutlineSize = 0.0f;
		markerColor = ImVec4(1.0f, 1.0f, 1.0f, 0.5f);
		markerShape = ImPlotMarker_Circle;
		gridAlpha = 0.15f;
	}
};

template<typename T> struct InputValuesBuffer
{
	T MIN[MAX_VARS_PARAMS];
	T MAX[MAX_VARS_PARAMS];
	T STEP[MAX_VARS_PARAMS];
	RangingType RANGING[MAX_VARS_PARAMS];
	int stepCount[MAX_VARS_PARAMS];

	void load(T min, T max, T step, RangingType isRanging, int index)
	{
		MIN[index] = min;
		MAX[index] = max;
		STEP[index] = step;
		RANGING[index] = isRanging;

		stepCount[index] = calculateStepCount(min, max, step);
	}

	// Import into struct
	void load(T* min, T* max, T* step, RangingType* isRanging, int size)
	{
		for (int i = 0; i < size; i++)
		{
			load(min[i], max[i], step[i], isRanging[i], i);
		}
	}

	// Export from struct
	void unload(T* min, T* max, T* step, RangingType* isRanging, int size)
	{
		for (int i = 0; i < size; i++)
		{
			min[i] = MIN[i];
			max[i] = MAX[i];
			step[i] = STEP[i];
			isRanging[i] = RANGING[i];
		}
	}

	void recountSteps(int i)
	{
		stepCount[i] = calculateStepCount(MIN[i], MAX[i], STEP[i]);
	}

	int stepsOf(int index)
	{
		return stepCount[index];
	}
};

