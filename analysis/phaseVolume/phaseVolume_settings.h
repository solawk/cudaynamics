#pragma once
#include "cuda_runtime.h"
#include "../port.h"
#include "../numb.h"
#include "abstractSettings_struct.h"

struct PV_Settings : AbstractAnalysisSettingsStruct
{
	int ObsSteps;
	int normVariables[4];

	Port PV;

	PV_Settings()
	{
		ObsSteps = 1000;
		normVariables[0] = 0;
		normVariables[1] = 1;
		normVariables[2] = 2;
		normVariables[3] = -1;
		PV = Port();
	}

	void DisplaySettings(std::vector<Attribute>& variables)
	{
		DisplayIntSetting("Observation steps", ObsSteps);
		for (int i = 0; i < 4; i++) DisplayVarSetting("Norm variable " + std::to_string(i + 1), normVariables[i], variables, true);
	}

	bool setup(std::vector<std::string> s)
	{
		if (!isMapSetupOfCorrectLength(s, 5)) return false;

		ObsSteps = s2i(s[0]);
		normVariables[0] = s2i(s[1]);
		normVariables[1] = s2i(s[2]);
		normVariables[2] = s2i(s[3]);
		normVariables[3] = s2i(s[4]);

		return true;
	}

	json::jobject ExportSettings()
	{
		json::jobject j;
		j["name"] = std::string(AnFuncNames[(int)ANF_PV]);

		std::vector<std::string> s;
		s.push_back(std::to_string(ObsSteps));
		s.push_back(std::to_string(normVariables[0]));
		s.push_back(std::to_string(normVariables[1]));
		s.push_back(std::to_string(normVariables[2]));
		s.push_back(std::to_string(normVariables[3]));
		j["settings"] = s;

		return j;
	}
};