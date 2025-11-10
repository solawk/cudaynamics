#pragma once

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
#include "numb.h"

#define calculateStepCount(_min, _max, _step) (_step != 0 ? (int)((_max - _min) / _step) + 1 : 0)
#define stepFromStepCount(_min, _max, _stepCount) ((_max - _min) / (numb)(_stepCount - 1))
#define calculateValue(_min, _step, _currentStep) ((_min) + (_step) * (_currentStep))
// Calculate step from value, floored int
#define stepFromValue(_min, _step, _value) (int)((_value - _min) / _step)
#define valueFromStep(_min, _step, _index) (_min + _step * _index)

// This enum lists all present analysis functions in the project
enum AnalysisFunction { ANF_MINMAX, ANF_LLE, ANF_PERIOD };

// None - no ranging, only the min value
// Linear - fixed step values from min (inclusive) to max (not necessarily inclusive)
// UniformRandom - random values from min to max with uniform distribution, step = quantity
// NormalRandom - random values from min to max with normal distribution around midpoint, step = quantity
enum RangingType { RT_None, RT_Step, RT_Linear, RT_UniformRandom, RT_NormalRandom, RT_Enum };

// Parameter step – doesn't change throughout the simulation
// Variable step - can change throughout the simulation and be plotted like a variable
// Discrete step - equals 1
enum StepType { ST_Parameter, ST_Variable, ST_Discrete };

struct Index
{
	bool enabled;
	std::string name;
	AnalysisFunction function;

	Index()
	{
		enabled = false;
		name = "ERROR";
		function = ANF_MINMAX;
	}

	Index(std::string _name, AnalysisFunction _function)
	{
		enabled = true;
		name = _name;
		function = _function;
	}
};