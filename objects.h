#pragma once
#include <string>
#include <vector>
#include <set>

using namespace std;

#define calculateStepCount(_min, _max, _step) (int)((_max - _min) / _step) + 1

enum PlotType { Series, Phase, Orbit, PlotType_COUNT };

struct PlotWindow
{
public:
	bool active; // Deactivated window is removed
	int id; // Unique id
	string name; // Name of the window
	PlotType type;
	int variableCount;
	vector<int> variables;

	// Plot rotation
	bool is3d;
	float yaw;
	float pitch;
	float xOffset, yOffset, zOffset;
	float xScale, yScale, zScale;

	PlotWindow(int _id)
	{
		active = true;
		id = _id;
	}

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

		xScale = 0.0f;
		yScale = 0.0f;
		zScale = 0.0f;
	}

	void AssignVariables(int* variablesArray)
	{
		variableCount = 0;
		for (int i = 0; i < 32; i++)
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

	string ExportAsString()
	{
		string exportString = name;

		exportString += " " + to_string((int)type);

		exportString += " " + to_string((int)is3d);
		exportString += " " + to_string(yaw) + " " + to_string(pitch);
		exportString += " " + to_string(xOffset) + " " + to_string(yOffset) + " " + to_string(zOffset);
		exportString += " " + to_string(xScale) + " " + to_string(yScale) + " " + to_string(zScale);

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
		yaw = atof(data[3].c_str());
		pitch = atof(data[4].c_str());

		xOffset = atof(data[5].c_str());
		yOffset = atof(data[6].c_str());
		zOffset = atof(data[7].c_str());

		xScale = atof(data[8].c_str());
		yScale = atof(data[9].c_str());
		zScale = atof(data[10].c_str());

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
	SinglePreRangingInfo rangings[32]{ 0 };
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