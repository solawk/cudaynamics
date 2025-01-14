#pragma once
#include <string>
#include <vector>
#include <set>

#include "imgui/backends/imgui_impl_win32.h"
#include "imgui/backends/imgui_impl_dx11.h"
#include "implot/implot.h"
#include "quaternion.h"

#define MAX_VARS_PARAMS 32

using namespace std;

#define calculateStepCount(_min, _max, _step) (_step != 0 ? (int)((_max - _min) / _step) + 1 : 0)

enum PlotType { Series, Phase, Orbit, Heatmap, PlotType_COUNT };
enum RangingType { None, Linear, UniformRandom, NormalRandom };

struct PlotWindow
{
public:
	bool active; // Deactivated window is removed
	int id; // Unique id
	string name; // Name of the window
	PlotType type;
	int variableCount;
	vector<int> variables; // or map

	// Plot rotation
	bool is3d;
	ImVec2 rotation;
	ImVec2 deltarotation;
	ImVec4 offset;
	ImVec4 scale;
	ImVec4 quatRot;

	// Plot settings
	bool settingsListEnabled;
	float markerSize;
	float markerOutlineSize;
	ImVec4 markerColor;
	ImPlotMarker markerShape;
	float rulerAlpha;
	float gridAlpha;

	PlotWindow(int _id)
	{
		active = true;
		id = _id;

		quatRot = ImVec4(1.0f, 0.0f, 0.0f, 0.0f);

		settingsListEnabled = true;
		markerSize = 1.0f;
		markerOutlineSize = 0.0f;
		markerColor = ImVec4(1.0f, 1.0f, 1.0f, 0.5f);
		markerShape = ImPlotMarker_Circle;
		rulerAlpha = 0.5f;
		gridAlpha = 0.15f;
	}

	PlotWindow(int _id, string _name, bool _is3d)
	{
		active = true;
		id = _id;
		name = _name;

		is3d = _is3d;
		rotation = ImVec2(0.0f, 0.0f);
		quatRot = ImVec4(1.0f, 0.0f, 0.0f, 0.0f);

		offset = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
		scale = ImVec4(1.0f, 1.0f, 1.0f, 0.0f);

		settingsListEnabled = true;
		markerSize = 1.0f;
		markerOutlineSize = 0.0f;
		markerColor = ImVec4(1.0f, 1.0f, 1.0f, 0.5f);
		markerShape = ImPlotMarker_Circle;
		rulerAlpha = 0.5f;
		gridAlpha = 0.15f;
	}

	void AssignVariables(int* variablesArray)
	{
		variableCount = 0;
		for (int i = 0; i < MAX_VARS_PARAMS; i++)
		{
			if (variablesArray[i] > -1)
			{
				variableCount++;
				variables.push_back(variablesArray[i]);
			}
			else break;
		}
	}

	void AssignVariables(set<int>& variablesSet)
	{
		variableCount = 0;
		for (const int& v : variablesSet)
		{
			variableCount++;
			variables.push_back(v);
		}
	}

	void AssignVariables(int singleVariable)
	{
		variableCount = 1;
		variables.push_back(singleVariable);
	}

	string ExportAsString()
	{
		string exportString = name;

		exportString += " " + to_string((int)type);

		exportString += " " + to_string((int)is3d);
		exportString += " " + to_string(rotation.x) + " " + to_string(rotation.y);
		exportString += " " + to_string(offset.x) + " " + to_string(offset.y) + " " + to_string(offset.z);
		exportString += " " + to_string(scale.x) + " " + to_string(scale.y) + " " + to_string(scale.z);

		exportString += " " + to_string(variableCount);
		for (int v : variables)
			exportString += " " + to_string(v);

		exportString += "\n";

		return exportString;
	}

	void ImportAsString(string input)
	{
		// string split by Arafat Hasan
		// https://stackoverflow.com/questions/14265581/parse-split-a-string-in-c-using-string-delimiter-standard-c
		size_t pos_start = 0, pos_end, delim_len = 1;
		std::string token;
		std::vector<std::string> data;
		while ((pos_end = input.find(" ", pos_start)) != std::string::npos)
		{
			token = input.substr(pos_start, pos_end - pos_start);
			pos_start = pos_end + delim_len;
			data.push_back(token);
		}
		data.push_back(input.substr(pos_start));

		//vector<string> data = split(input, " ");

		name = data[0];
		type = (PlotType)atoi(data[1].c_str());

		is3d = (bool)atoi(data[2].c_str());
		rotation.x = (float)atof(data[3].c_str());
		rotation.y = (float)atof(data[4].c_str());

		quatRot = ImVec4(1.0f, 0.0f, 0.0f, 0.0f);

		offset.x = (float)atof(data[5].c_str());
		offset.y = (float)atof(data[6].c_str());
		offset.z = (float)atof(data[7].c_str());

		scale.x = (float)atof(data[8].c_str());
		scale.y = (float)atof(data[9].c_str());
		scale.z = (float)atof(data[10].c_str());

		variableCount = atoi(data[11].c_str());
		variables.clear();
		for (int i = 0; i < variableCount; i++)
			variables.push_back(atoi(data[12 + i].c_str()));
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

	vector<string> names;
	vector<float> min;
	vector<float> step;
	vector<float> max;

	vector<int> stepCount;
	vector<int> currentStep;

	vector<float> currentValue;

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
		currentStep.clear();
		currentValue.clear();
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
	bool IS_RANGING[MAX_VARS_PARAMS];
	int stepCount[MAX_VARS_PARAMS];

	void load(T min, T max, T step, T isRanging, int index)
	{
		MIN[index] = min;
		MAX[index] = max;
		STEP[index] = step;
		IS_RANGING[index] = isRanging;

		stepCount[index] = calculateStepCount(min, max, step);
	}

	void load(T* min, T* max, T* step, bool* isRanging, int size)
	{
		for (int i = 0; i < size; i++)
		{
			load(min[i], max[i], step[i], isRanging[i], i);
		}
	}

	void unload(T* min, T* max, T* step, bool* isRanging, int size)
	{
		for (int i = 0; i < size; i++)
		{
			min[i] = MIN[i];
			max[i] = MAX[i];
			step[i] = STEP[i];
			isRanging[i] = IS_RANGING[i];
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