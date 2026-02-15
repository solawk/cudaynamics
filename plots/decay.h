#pragma once
#include <vector>
#include "../decaySettings_struct.h"
#include <string>
#include "imgui/backends/imgui_impl_win32.h"
#include "imgui/backends/imgui_impl_dx11.h"
#include "../kernel_map.h"
#include "../computation_struct.h"
#include "../index.h"
#include "../index2port.h"

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

	void RecalculatePlot(Computation* cmp, std::vector<int>& variables, int mapValueIndex)
	{
		if (buffer.size() == 0)
		{
			for (int t = 0; t < thresholdCount; t++)
			{
				buffer.push_back(std::vector<float>{});
				total.push_back(std::vector<float>{});
				alive.push_back(std::vector<float>{});
			}
		}

		for (int t = 0; t < thresholdCount; t++)
		{
			int decayAlive = 0, decayTotal = cmp->marshal.totalVariations;
			buffer[t].push_back(!KERNEL.usingTime ? (cmp->bufferNo * KERNEL.steps + KERNEL.transientSteps) : (cmp->bufferNo * KERNEL.time + KERNEL.transientTime));

			for (int i = 0; i < cmp->marshal.totalVariations; i++)
			{
				bool anyAlive = false;
				bool anyDead = false;
				for (int index : variables)
				{
					numb* decay = cmp->marshal.indecesDecay
						+ (index2port(cmp->marshal.kernel.analyses, (AnalysisIndex)index)->offset + mapValueIndex) * cmp->marshal.totalVariations;
					if (decay[i] < (numb)(t + 1))
						anyAlive = true;
					else
						anyDead = true;
				}
				if (indicesAreAND && !anyDead) decayAlive++;
				else if (!indicesAreAND && anyAlive) decayAlive++;
			}

			total[t].push_back(decayTotal);
			alive[t].push_back(decayAlive);
		}
	}

	void RecalculateLifetime(Computation* cmp)
	{
		int endDecayTimepointIndex = 0;
		for (int tp = 1; tp < buffer[0].size(); tp++)
		{
			if (markerPosition > buffer[0][tp]) endDecayTimepointIndex = tp;
			else break;
		}

		float totalArea = 0.0f;

		for (int tp = 0; tp < endDecayTimepointIndex; tp++)
		{
			float min1 = alive[0][tp];
			float min2 = alive[0][tp + 1];

			for (int t = 1; t < thresholdCount; t++)
			{
				if (min1 > alive[t][tp]) min1 = alive[t][tp];
				if (min2 > alive[t][tp + 1]) min2 = alive[t][tp + 1];
			}

			float time1 = buffer[0][tp];
			float time2 = buffer[0][tp + 1];

			float area = (time2 - time1) * (min1 + min2) / 2.0f;
			totalArea += area;
		}

		lifetime = totalArea / cmp->marshal.totalVariations;
	}
};