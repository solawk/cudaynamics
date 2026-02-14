#pragma once
#include "numb.h"
#include <string>

#define MAX_ATTRIBUTES 128
#define MAX_MAPS 16

#define calculateStepCount(_min, _max, _step) (_step != 0 ? (int)((_max - _min) / _step) + 1 : 0)
#define stepFromStepCount(_min, _max, _stepCount) ((_max - _min) / (numb)(_stepCount - 1))
#define calculateValue(_min, _step, _currentStep) ((_min) + (_step) * (_currentStep))
// Calculate step from value, floored int
#define stepFromValue(_min, _step, _value) (int)((_value - _min) / _step)
#define valueFromStep(_min, _step, _index) (_min + _step * _index)

// None - no ranging, only the min value
// Linear - fixed step values from min (inclusive) to max (not necessarily inclusive)
// UniformRandom - random values from min to max with uniform distribution, step = quantity
// NormalRandom - random values from min to max with normal distribution around midpoint, step = quantity
enum RangingType { RT_None, RT_Step, RT_Linear, RT_UniformRandom, RT_NormalRandom, RT_Enum };

// Parameter step – doesn't change throughout the simulation
// Variable step - can change throughout the simulation and be plotted like a variable
// Discrete step - equals 1
enum StepType { ST_Parameter, ST_Variable, ST_Discrete };