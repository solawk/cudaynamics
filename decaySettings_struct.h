#pragma once
#include <vector>

enum DecayThresholdSource { DTS_Index, DTS_Delta };
enum DecayThresholdMode { DTM_Disabled, DTM_Less, DTM_More, DTM_Abs_More };

struct DecaySettings
{
	DecayThresholdSource source;
	DecayThresholdMode mode;
	std::vector<float> thresholds;

	DecaySettings()
	{
		source = DTS_Index;
		mode = DTM_Disabled;
		thresholds.push_back(0.0f);
	}
};