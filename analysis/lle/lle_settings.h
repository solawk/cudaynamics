#pragma once
#include "cuda_runtime.h"
#include "../port.h"
#include "../numb.h"
#include "abstractSettings_struct.h"

struct LLE_Settings : AbstractAnalysisSettingsStruct
{
	numb r;			// Initial deflection
	int L;			// Deflection observation duration (steps)
	int variableToDeflect;	// Variable to initially deflect
	int normVariables[4];	// Variables to count the norm with (-1 if not counting)

	Port LLE;

	LLE_Settings() 
	{
		r = (numb)0.01;
		L = 50;
		variableToDeflect = 0;
		normVariables[0] = 0;
		normVariables[1] = 1;
		normVariables[2] = 2;
		normVariables[3] = -1;

		LLE = Port();
	}

	void DisplaySettings(std::vector<Attribute>& variables)
	{
		DisplayNumbSetting("Initial deflection", r);
		DisplayIntSetting("Observation steps", L);
		DisplayVarSetting("Deflected variable", variableToDeflect, variables);
		for (int i = 0; i < 4; i++) DisplayVarSetting("Norm variable " + std::to_string(i + 1), normVariables[i], variables, true);
	}

	bool setup(std::vector<std::string> s)
	{
		if (!isMapSetupOfCorrectLength(s, 7)) return false;

		r = s2n(s[0]);
		L = s2i(s[1]);
		variableToDeflect = s2i(s[2]);
		normVariables[0] = s2i(s[3]);
		normVariables[1] = s2i(s[4]);
		normVariables[2] = s2i(s[5]);
		normVariables[3] = s2i(s[6]);

		return true;
	}

	json::jobject ExportSettings()
	{
		json::jobject j;
		j["name"] = std::string(AnFuncNames[(int)ANF_LLE]);

		std::vector<std::string> s;
		s.push_back(std::to_string(r));
		s.push_back(std::to_string(L));
		s.push_back(std::to_string(variableToDeflect));
		s.push_back(std::to_string(normVariables[0]));
		s.push_back(std::to_string(normVariables[1]));
		s.push_back(std::to_string(normVariables[2]));
		s.push_back(std::to_string(normVariables[3]));
		j["settings"] = s;

		return j;
	}
};