#pragma once
#include <string>

using namespace std;

// Maximum amount of variables and parameters in the plot
#define MAX_VARS_PARAMS 32

enum PlotType { Series, Phase, Orbit, Heatmap, PlotType_COUNT };

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
	ImVec4 offset;
	ImVec4 scale;
	ImVec4 quatRot;
	ImVec4 autorotate; // euler angles
	ImVec2 deltarotation; // euler angles

	// Plot settings
	bool settingsListEnabled;
	float markerSize;
	float markerOutlineSize;
	ImVec4 markerColor;
	ImPlotMarker markerShape;
	float rulerAlpha;
	float gridAlpha;
	bool whiteBg;
	bool grayscaleHeatmap;
	int stride;
	bool isHeatmapSelectionModeOn;

	bool showAxis;
	bool showAxisNames;
	bool showRuler;
	bool showGrid;

	bool showHeatmapValues;
	bool showActualDiapasons;

	PlotWindow(int _id, string _name = "plot", bool _is3d = false)
	{
		active = true;
		id = _id;
		name = _name;

		is3d = _is3d;
		quatRot = ImVec4(1.0f, 0.0f, 0.0f, 0.0f);
		autorotate = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);

		offset = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
		scale = ImVec4(1.0f, 1.0f, 1.0f, 0.0f);

		settingsListEnabled = true;
		markerSize = 1.0f;
		markerOutlineSize = 0.0f;
		markerColor = ImVec4(1.0f, 1.0f, 1.0f, 0.5f);
		markerShape = ImPlotMarker_Circle;
		rulerAlpha = 0.5f;
		gridAlpha = 0.15f;
		whiteBg = false;
		grayscaleHeatmap = false;
		stride = 1;
		isHeatmapSelectionModeOn = false;

		showAxis = true;
		showAxisNames = true;
		showRuler = true;
		showGrid = true;

		showHeatmapValues = false;
		showActualDiapasons = true;
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
		exportString += " " + to_string(quatRot.x) + " " + to_string(quatRot.y) + " " + to_string(quatRot.z) + " " + to_string(quatRot.w);
		exportString += " " + to_string(autorotate.x) + " " + to_string(autorotate.y) + " " + to_string(autorotate.z);
		exportString += " " + to_string(offset.x) + " " + to_string(offset.y) + " " + to_string(offset.z);
		exportString += " " + to_string(scale.x) + " " + to_string(scale.y) + " " + to_string(scale.z);
		exportString += " " + to_string((int)showAxis) + " " + to_string((int)showAxisNames) + " " + to_string((int)showRuler) + " " + to_string((int)showGrid);

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

		int d = 0;

		name = data[d++];
		type = (PlotType)atoi(data[d++].c_str());

		is3d = (bool)atoi(data[d++].c_str());

		quatRot.x = (float)atof(data[d++].c_str());
		quatRot.y = (float)atof(data[d++].c_str());
		quatRot.z = (float)atof(data[d++].c_str());
		quatRot.w = (float)atof(data[d++].c_str());

		autorotate.x = (float)atof(data[d++].c_str());
		autorotate.y = (float)atof(data[d++].c_str());
		autorotate.z = (float)atof(data[d++].c_str());

		offset.x = (float)atof(data[d++].c_str());
		offset.y = (float)atof(data[d++].c_str());
		offset.z = (float)atof(data[d++].c_str());

		scale.x = (float)atof(data[d++].c_str());
		scale.y = (float)atof(data[d++].c_str());
		scale.z = (float)atof(data[d++].c_str());

		showAxis = (bool)atoi(data[d++].c_str());
		showAxisNames = (bool)atoi(data[d++].c_str());
		showRuler = (bool)atoi(data[d++].c_str());
		showGrid = (bool)atoi(data[d++].c_str());

		variableCount = atoi(data[d++].c_str());
		variables.clear();
		for (int i = 0; i < variableCount; i++)
			variables.push_back(atoi(data[d++].c_str()));
	}
};