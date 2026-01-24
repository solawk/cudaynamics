#pragma once
#include <vector>
#include "../decaySettings_struct.h"
#include <string>
#include "imgui/backends/imgui_impl_win32.h"
#include "imgui/backends/imgui_impl_dx11.h"

struct DecayProperties
{
	// Outer layer is per threshold, inner layer is per index
	std::vector<std::vector<float>> buffer;
	std::vector<std::vector<float>> total;
	std::vector<std::vector<float>> alive;
	int thresholdCount;
	std::vector<DecaySettings> settings; // Per index
	// Index 0 is the OG, all shared settings are stored in that one
	std::vector<std::string> thresholdNames;

	bool indicesAreAND; // true if AND, false if OR
	ImVec4 plotFillColor;
	float fillAlpha;
	double markerPosition;
	bool calcLifetime;
	float lifetime;

	DecayProperties()
	{
		thresholdCount = 1;
		indicesAreAND = true;
		fillAlpha = 0.75f;
		plotFillColor = ImVec4(1.0f, 1.0f, 1.0f, 0.0f);
		markerPosition = 0.0f;
		calcLifetime = true;
		lifetime = 0.0f;
	}

	void ForceDecayThresholdCount()
	{
		for (int i = 0; i < settings.size(); i++)
		{
			settings[i].thresholds.resize(thresholdCount, 0.0f);
			thresholdNames.resize(thresholdCount, "Threshold");
		}
	}

	void DeleteBuffers()
	{
		for (int t = 0; t < alive.size(); t++)
		{
			alive[t].clear();
			buffer[t].clear();
			total[t].clear();
		}
		alive.clear();
		buffer.clear();
		total.clear();
	}
};