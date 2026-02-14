#include "main_utils.h"

Kernel readKernelText(std::string name)
{
	std::ifstream fileStream(("systems/" + name + "/" + name + ".txt").c_str(), std::ios::out);
	std::vector<std::string> str;

	Kernel kernel;
	Attribute tempAttribute;
	MapData tempMapData;
	Constraint tempConstraint;
	bool constraintExpected = false;

	int mapSettingsCount = 0;

	bool stepLineFound = false;
	std::string stepLine = "";
	kernel.stepType = ST_Parameter;

	for (std::string line; std::getline(fileStream, line); )
	{
		str = splitString(line);

		if (str[0] == "//") continue;

		if (str[0] == "Name:")
		{
			kernel.name = "";
			for (int i = 1; i < str.size(); i++)
				kernel.name += (i > 1 ? " " : "") + str[i];
			continue;
		}
		if (str[0] == "Steps:") { kernel.steps = atoi(str[1].c_str()); continue; }
		if (str[0] == "Transient:") { kernel.transientSteps = atoi(str[1].c_str()); continue; }
		if (str[0] == "Execute") { kernel.executeOnLaunch = str[3] == "yes"; continue; }

		if (str[0] == "Step" && str[1] == "type:")
		{ 
			if (str[2] == "variable") kernel.stepType = ST_Variable;
			if (str[2] == "discrete") kernel.stepType = ST_Discrete;
			stepLineFound = true;
			stepLine = line;
			continue; 
		}

		if (constraintExpected && str[0] != "constraint")
		{
			tempConstraint.Clear();
			kernel.constraints.push_back(tempConstraint);
			constraintExpected = false;
		}

		if (constraintExpected && str[0] == "constraint")
		{
			tempConstraint.Clear();
			int constraintCount = ((int)str.size() - 1) / 2;
			for (int i = 0; i < constraintCount; i++)
			{
				tempConstraint.lhs.push_back(str[1 + i * 2 + 0]);
				tempConstraint.rhs.push_back(str[1 + i * 2 + 1]);
			}
			tempConstraint.hasConstraints = true;
			tempConstraint.count = constraintCount;
			kernel.constraints.push_back(tempConstraint);
			constraintExpected = false;
		}

		if (str[0] == "var" || str[0] == "param")
		{
			tempAttribute.name = str[1];
			tempAttribute.rangingType = rangingTypeFromString(str[2]);
			tempAttribute.min = (numb)atof(str[3].c_str());
			tempAttribute.max = (numb)atof(str[4].c_str());
			tempAttribute.step = (numb)atof(str[5].c_str());
			tempAttribute.stepCount = atoi(str[6].c_str());
			tempAttribute.mean = (numb)atof(str[7].c_str());
			tempAttribute.deviation = (numb)atof(str[8].c_str());
			tempAttribute.values = nullptr;

			tempAttribute.CalcStep();
			tempAttribute.CalcStepCount();
			
			if (tempAttribute.stepCount < 2) tempAttribute.stepCount = 2;

			if (str[0] == "var")
				kernel.variables.push_back(tempAttribute);
			else
			{
				kernel.parameters.push_back(tempAttribute);
				constraintExpected = true;
			}
		}

		if (str[0] == "enum")
		{
			tempAttribute.name = str[1];
			tempAttribute.values = nullptr;
			tempAttribute.rangingType = RT_Enum;

			tempAttribute.enumCount = (int)str.size() - 2;
			if (tempAttribute.enumCount > MAX_ENUMS) tempAttribute.enumCount = MAX_ENUMS;
			for (int i = 0; i < tempAttribute.enumCount; i++)
			{
				tempAttribute.enumEnabled[i] = str[2 + i][0] == '1';
				str[2 + i] = str[2 + i].substr(1);
				tempAttribute.enumNames[i] = str[2 + i];
			}

			tempAttribute.CalcStep();
			tempAttribute.CalcStepCount();

			if (tempAttribute.stepCount < 2) tempAttribute.stepCount = 2;

			kernel.parameters.push_back(tempAttribute);

			constraintExpected = true;
		}

		if (str[0] == "analysis")
		{
			bool correct = false;
			int nameEnd = 0, settingsStart = 0;
			for (int i = 2; i < str.size(); i++)
			{
				if (str[i] == "settings")
				{
					nameEnd = i - 1;
					settingsStart = i + 1;
					correct = true;
					break;
				}
			}

			if (correct)
			{
				std::vector<std::string> settingsVector = std::vector<std::string>(str.begin() + settingsStart, str.end());
				std::string anSetName = "";
				for (int i = 1; i <= nameEnd; i++) anSetName += str[i] + (i != nameEnd ? " " : "");

				for (int anfunc = 0; anfunc < (int)AnalysisFunction::ANF_COUNT; anfunc++)
					if (anSetName == AnFuncNames[anfunc])
					{
						switch ((AnalysisFunction)anfunc)
						{
						case ANF_MINMAX:
							if (!kernel.analyses.MINMAX.setup(settingsVector)) printf("Wrong settings count for MINMAX in %s\n", kernel.name.c_str());
							break;
						case ANF_LLE:
							if (!kernel.analyses.LLE.setup(settingsVector)) printf("Wrong settings count for LLE in %s\n", kernel.name.c_str());
							break;
						case ANF_PERIOD:
							if (!kernel.analyses.PERIOD.setup(settingsVector)) printf("Wrong settings count for PERIOD in %s\n", kernel.name.c_str());
							break;
						case ANF_PV:
							if (!kernel.analyses.PV.setup(settingsVector)) printf("Wrong settings count for PV in %s\n", kernel.name.c_str());
							break;
						}
					}
			}
		}
	}

	if (constraintExpected)
	{
		tempConstraint.Clear();
		kernel.constraints.push_back(tempConstraint);
		constraintExpected = false;
	}

	// Adding step
	if (!stepLineFound)
	{
		// Default step values
		tempAttribute.name = "h";
		tempAttribute.rangingType = RT_None;
		tempAttribute.min = (numb)0.01;
		tempAttribute.max = (numb)0.1;
		tempAttribute.step = (numb)0.01;
		tempAttribute.stepCount = 10;
		tempAttribute.mean = (numb)0.0;
		tempAttribute.deviation = (numb)0.0;
		tempAttribute.values = nullptr;

		tempAttribute.CalcStep();
		tempAttribute.CalcStepCount();

		if (tempAttribute.stepCount < 2) tempAttribute.stepCount = 2;

		kernel.parameters.push_back(tempAttribute);
	}
	else if (kernel.stepType != ST_Discrete)
	{
		// Import step values
		str = splitString(stepLine);
		tempAttribute.name = str[3];
		tempAttribute.rangingType = rangingTypeFromString(str[4]);
		tempAttribute.min = (numb)atof(str[5].c_str());
		tempAttribute.max = (numb)atof(str[6].c_str());
		tempAttribute.step = (numb)atof(str[7].c_str());
		tempAttribute.stepCount = atoi(str[8].c_str());
		tempAttribute.mean = (numb)atof(str[9].c_str());
		tempAttribute.deviation = (numb)atof(str[10].c_str());
		tempAttribute.values = nullptr;

		tempAttribute.CalcStep();
		tempAttribute.CalcStepCount();

		if (tempAttribute.stepCount < 2) tempAttribute.stepCount = 2;

		if (kernel.stepType == ST_Parameter)
			kernel.parameters.push_back(tempAttribute);
		else if (kernel.stepType == ST_Variable)
			kernel.variables.push_back(tempAttribute);
	}

	fileStream.close();

	kernel.calcAttributeCounts();
	kernel.mapWeight = 1.0f;
	kernel.usingTime = false;
	return kernel;
}

std::string timeAsString()
{
	time_t timestamp = time(NULL);
	struct tm timeStruct;
	localtime_s(&timeStruct, &timestamp);

	std::string mday = std::to_string(timeStruct.tm_mday);	if (mday.size() < 2) mday = "0" + mday;
	std::string mon = std::to_string(timeStruct.tm_mon);	if (mon.size() < 2) mon = "0" + mon;
	std::string hour = std::to_string(timeStruct.tm_hour);	if (hour.size() < 2) hour = "0" + hour;
	std::string min = std::to_string(timeStruct.tm_min);	if (min.size() < 2) min = "0" + min;
	std::string sec = std::to_string(timeStruct.tm_sec);	if (sec.size() < 2) sec = "0" + sec;

	std::string time = mday + mon + "_" + hour + min + sec;
	return time;
}