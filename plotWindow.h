#pragma once
#include <string>
#include <vector>
#include <set>
#include "heatmapProperties.hpp"
#include "fontSettings_struct.h"
#include "plots/decay.h"
#include "plots/trs.h"
#include "plots/orbit.h"
#include "index.h"

// Maximum amount of variables and parameters in the plot
#define MAX_VARS_PARAMS 32

enum PlotType { VarSeries, Phase, Phase2D, Orbit, Heatmap, MCHeatmap, Metric, IndSeries, Decay, PlotType_COUNT };
enum DeltaState { DS_No, DS_Delta, DS_Decay };

struct PlotWindow
{
public:
	bool active; // Deactivated window is removed
	int id; // Unique id
	bool newWindow;
	std::string name; // Name of the window
	PlotType type;
	int variableCount;
	std::vector<int> variables; // or map

	// Plot rotation
	TRS trs;

	// Plot settings
	bool settingsListEnabled;

	bool overrideFontOnNextFrame;
	bool overrideFontSettings;
	FontSettings localFontSettings;

	float markerWidth;
	float markerOutlineWidth;
	ImVec4 markerColor;

	ImVec4 plotColor;

	ImPlotMarker markerShape;

	bool isYLog;
	float rulerAlpha;
	float gridAlpha;
	bool whiteBg;
	bool isImplot3d;
	bool drawAllTrajectories;

	bool isFrozen;
	bool isFrozenAsHires;

	HeatmapProperties hmp;
	HeatmapProperties hireshmp;

	DeltaState deltaState;

	DecayProperties decay;
	
	ImVec2 dragLineHiresPos;

	bool showAxis;
	bool showAxisNames;
	bool showRuler;
	bool showGrid;

	bool isFullscreen, isFullscreenEnd;
	ImVec2 originalPos, originalSize;

	///// Orbit settings
	OrbitProperties orbit;
	///// 

	bool ShowMultAxes;
	bool LineColorMaps;
	ImPlotColormap colormap;

	int indexX;
	MapDimensionType typeX;

	std::vector<numb> indSeries;
	int firstBufferNo;
	int prevbufferNo;

	PlotWindow(int _id, std::string _name = "plot", bool _is3d = false)
	{
		active = true;
		id = _id;
		newWindow = false;
		name = _name;

		settingsListEnabled = false;
		isFrozen = isFrozenAsHires = false;

		overrideFontOnNextFrame = false;
		overrideFontSettings = false;
		localFontSettings = FontSettings(0, false, false, 24);

		markerWidth = 1.0f;
		markerOutlineWidth = 0.0f;
		markerColor = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);

		plotColor = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);

		markerShape = ImPlotMarker_Circle;

		isYLog = false;
		rulerAlpha = 0.5f;
		gridAlpha = 0.15f;
		whiteBg = false;
		isImplot3d = false;
		drawAllTrajectories = false;
		
		deltaState = DS_No;
		dragLineHiresPos = ImVec2(0.0f, 0.0f);

		showAxis = true;
		showAxisNames = true;
		showRuler = true;
		showGrid = true;

		isFullscreen = isFullscreenEnd = false;
		originalPos = originalSize = ImVec2(0.0f, 0.0f);

		ShowMultAxes = false;
		colormap = ImPlotColormap_Deep;

		indexX = 0;
		typeX = MDT_Variable;

		
		firstBufferNo = 0;
		prevbufferNo = 0;
	}

	bool isTheHiresWindow(AnalysisIndex _hiresIndex)
	{
		return variables[0] == _hiresIndex;
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

		exportString += " " + std::to_string(trs.quatRot.x) + " " + std::to_string(trs.quatRot.y) + " " + std::to_string(trs.quatRot.z) + " " + std::to_string(trs.quatRot.w);
		exportString += " " + std::to_string(trs.autorotate.x) + " " + std::to_string(trs.autorotate.y) + " " + std::to_string(trs.autorotate.z);
		exportString += " " + std::to_string(trs.offset.x) + " " + std::to_string(trs.offset.y) + " " + std::to_string(trs.offset.z);
		exportString += " " + std::to_string(trs.scale.x) + " " + std::to_string(trs.scale.y) + " " + std::to_string(trs.scale.z);
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

		trs.quatRot.x = (float)atof(data[d++].c_str());
		trs.quatRot.y = (float)atof(data[d++].c_str());
		trs.quatRot.z = (float)atof(data[d++].c_str());
		trs.quatRot.w = (float)atof(data[d++].c_str());

		trs.autorotate.x = (float)atof(data[d++].c_str());
		trs.autorotate.y = (float)atof(data[d++].c_str());
		trs.autorotate.z = (float)atof(data[d++].c_str());

		trs.offset.x = (float)atof(data[d++].c_str());
		trs.offset.y = (float)atof(data[d++].c_str());
		trs.offset.z = (float)atof(data[d++].c_str());

		trs.scale.x = (float)atof(data[d++].c_str());
		trs.scale.y = (float)atof(data[d++].c_str());
		trs.scale.z = (float)atof(data[d++].c_str());

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