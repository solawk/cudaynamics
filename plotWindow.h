#pragma once

#include <string>
#include "heatmapProperties.hpp"

// Maximum amount of variables and parameters in the plot
#define MAX_VARS_PARAMS 32

enum PlotType { Series, Phase, Phase2D, Orbit, Heatmap, MCHeatmap, Metric, PlotType_COUNT };
enum OrbitPlotType {  Peak_Bifurcation, Interval_Bifurcation, Bifurcation_3D, Selected_Var_Section, OrbitPlotType_COUNT};
struct PlotWindow
{
public:
	bool active; // Deactivated window is removed
	int id; // Unique id
	std::string name; // Name of the window
	PlotType type;
	int variableCount;
	std::vector<int> variables; // or map

	// Plot rotation
	ImVec4 offset;
	ImVec4 scale;
	ImVec4 quatRot;
	ImVec4 autorotate; // euler angles
	ImVec2 deltarotation; // euler angles

	// Plot settings
	bool settingsListEnabled;

	float markerWidth;
	float markerOutlineWidth;
	ImVec4 markerColor;

	ImVec4 plotColor;

	ImPlotMarker markerShape;

	float rulerAlpha;
	float gridAlpha;
	bool whiteBg;
	bool isImplot3d;
	bool drawAllTrajectories;

	HeatmapProperties hmp;
	HeatmapProperties hireshmp;
	
	ImVec2 dragLineHiresPos;

	bool showAxis;
	bool showAxisNames;
	bool showRuler;
	bool showGrid;

	bool isFullscreen, isFullscreenEnd;
	ImVec2 originalPos, originalSize;

	int OrbitXIndex;
	bool ShowOrbitParLines;
	OrbitPlotType OrbitType;
	float OrbitPointSize;
	float OrbitMarkerWidth;
	ImVec4 OrbitMarkerColor;
	bool OrbitInvertedAxes;

	bool areOrbitValuesDirty;
	numb* bifAmps;
	numb* bifParamIndices;
	numb* bifIntervals;
	std::vector<int> lastAttributeValueIndices;
	int BifDotAmount;
	int prevTotalVariation;

	int indexX;
	MapDimensionType typeX;

	PlotWindow(int _id, std::string _name = "plot", bool _is3d = false)
	{
		active = true;
		id = _id;
		name = _name;

		quatRot = ImVec4(1.0f, 0.0f, 0.0f, 0.0f);
		autorotate = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);

		offset = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
		scale = ImVec4(1.0f, 1.0f, 1.0f, 0.0f);

		settingsListEnabled = true;

		markerWidth = 1.0f;
		markerOutlineWidth = 0.0f;
		markerColor = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);

		plotColor = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);

		markerShape = ImPlotMarker_Circle;

		rulerAlpha = 0.5f;
		gridAlpha = 0.15f;
		whiteBg = false;
		isImplot3d = false;
		drawAllTrajectories = false;

		dragLineHiresPos = ImVec2(0.0f, 0.0f);

		showAxis = true;
		showAxisNames = true;
		showRuler = true;
		showGrid = true;

		isFullscreen = isFullscreenEnd = false;
		originalPos = originalSize = ImVec2(0.0f, 0.0f);

		OrbitXIndex = 0;
		ShowOrbitParLines = true;
		OrbitType = Peak_Bifurcation;
		OrbitPointSize = 0.5f;
		OrbitMarkerColor = ImVec4(1.0f, 0.0f, 0.0f, 0.5f);
		OrbitMarkerWidth = 1;
		OrbitInvertedAxes = false;
		areOrbitValuesDirty = true;
		bifAmps = NULL;
		bifParamIndices = NULL;
		bifIntervals = NULL;

		indexX = 0;
		typeX = MDT_Variable;
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

	void AssignVariables(std::set<int>& variablesSet)
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

	std::string ExportAsString()
	{
		std::string exportString = name;

		exportString += " " + std::to_string((int)type);
		exportString += " " + std::to_string((int)isImplot3d);

		exportString += " " + std::to_string(quatRot.x) + " " + std::to_string(quatRot.y) + " " + std::to_string(quatRot.z) + " " + std::to_string(quatRot.w);
		exportString += " " + std::to_string(autorotate.x) + " " + std::to_string(autorotate.y) + " " + std::to_string(autorotate.z);
		exportString += " " + std::to_string(offset.x) + " " + std::to_string(offset.y) + " " + std::to_string(offset.z);
		exportString += " " + std::to_string(scale.x) + " " + std::to_string(scale.y) + " " + std::to_string(scale.z);
		exportString += " " + std::to_string((int)showAxis) + " " + std::to_string((int)showAxisNames) + " " + std::to_string((int)showRuler) + " " + std::to_string((int)showGrid);

		exportString += " " + std::to_string(variableCount);
		for (int v : variables)
			exportString += " " + std::to_string(v);

		exportString += "\n";

		return exportString;
	}

	void ImportAsString(std::string input)
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
		isImplot3d = (bool)atoi(data[d++].c_str());

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