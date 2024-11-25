#pragma once
#include <string>
#include <vector>

using namespace std;

enum PlotType { Series, Phase, Orbit };

struct PlotWindow
{
public:
	bool active; // Deactivated window is removed
	int id; // Unique id
	string name; // Name of the window
	string plotName; // Name of the plot, generated from variable names
	PlotType type;
	int variables; // Bit mask of variable indeces (up to 32 variables per graph)

	// Plot rotation
	bool is3d;
	float yaw;
	float pitch;
	float xOffset, yOffset, zOffset;

	PlotWindow(int _id, string _name, bool _is3d)
	{
		active = true;
		id = _id;
		name = _name;

		is3d = _is3d;
		yaw = 0.0f;
		pitch = 0.0f;

		xOffset = 0.0f;
		yOffset = 0.0f;
		zOffset = 0.0f;
	}
};

struct SinglePreRangingInfo
{
public:
	int index;
	float min;
	float step;
	float max;
	int steps;

	void init(int _i, float _min, float _step, float _max, int _steps)
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
	SinglePreRangingInfo rangings[32]{ 0 };
	int totalVariations;

	PreRanging(int _v, int _p, int _r, int _t)
	{
		varCount = _v;
		paramCount = _p;
		rangingCount = _r;
		totalVariations = _t;
	}
};

struct PostRanging
{
public:
	int rangingCount;		// Amount of ranging variables/parameters
	int totalVariations;	// Ranging var/param combinations count

	vector<string> names;
	vector<float> min;
	vector<float> step;
	vector<float> max;

	vector<int> stepCount;
	vector<int> currentStep;

	vector<float> currentValue;

	PostRanging()
	{
		rangingCount = 0;
		totalVariations = 0;
	}

	void clear()
	{
		rangingCount = 0;
		totalVariations = 0;

		names.clear();
		min.clear();
		step.clear();
		max.clear();
		stepCount.clear();
		currentStep.clear();
		currentValue.clear();
	}
};