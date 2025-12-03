#pragma once

enum DecayThresholdSource { DTS_Index, DTS_Delta };
enum DecayThresholdMode { DTM_Less, DTM_More, DTM_Abs_More };

struct DecaySettings
{
	DecayThresholdSource source;
	DecayThresholdMode mode;
	float threshold;

	DecaySettings()
	{
		source = DTS_Index;
		mode = DTM_Less;
		threshold = 0.0f;
	}
};