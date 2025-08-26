#include "main_utils.h"

std::vector<std::string> splitString(std::string str)
{
	// string split by Arafat Hasan
	// https://stackoverflow.com/questions/14265581/parse-split-a-string-in-c-using-string-delimiter-standard-c
	size_t pos_start = 0, pos_end, delim_len = 1;
	std::string token;
	std::vector<std::string> data;
	while ((pos_end = str.find(" ", pos_start)) != std::string::npos)
	{
		token = str.substr(pos_start, pos_end - pos_start);
		pos_start = pos_end + delim_len;
		data.push_back(token);
	}
	data.push_back(str.substr(pos_start));

	return data;
}

RangingType rangingTypeFromString(std::string str)
{
	if (str == "Fixed") return None;
	if (str == "Linear") return Linear;
	if (str == "Step") return Step;
	if (str == "Random") return UniformRandom;
	if (str == "Normal") return NormalRandom;

	throw std::runtime_error("Invalid ranging type");
}

Kernel readKernelText(std::string name)
{
	std::ifstream fileStream(("systems/" + name + "/" + name + ".txt").c_str(), std::ios::out);
	std::vector<std::string> str;

	Kernel kernel;
	Attribute tempAttribute;
	MapData tempMapData;

	int mapSettingsCount = 0;

	for (std::string line; std::getline(fileStream, line); )
	{
		str = splitString(line);

		if (str[0] == "//") continue;

		if (str[0] == "Name:") { kernel.name = str[1]; continue; }
		if (str[0] == "Steps:") { kernel.steps = atoi(str[1].c_str()); continue; }
		if (str[0] == "Transient:") { kernel.transientSteps = atoi(str[1].c_str()); continue; }
		if (str[0] == "Step") { kernel.stepSize = (numb)atof(str[2].c_str()); continue; }
		if (str[0] == "Execute") { kernel.executeOnLaunch = str[3] == "yes"; continue; }

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
				kernel.parameters.push_back(tempAttribute);
		}

		if (str[0] == "enum")
		{
			tempAttribute.name = str[1];
			tempAttribute.values = nullptr;
			tempAttribute.rangingType = Enum;

			tempAttribute.CalcStep();
			tempAttribute.CalcStepCount();

			tempAttribute.enumCount = (int)str.size() - 2;
			for (int i = 0; i < tempAttribute.enumCount && i < MAX_ENUMS; i++)
			{
				tempAttribute.enumEnabled[i] = str[2 + i][0] == '1';
				str[2 + i] = str[2 + i].substr(1);
				tempAttribute.enumNames[i] = str[2 + i];
			}

			if (tempAttribute.stepCount < 2) tempAttribute.stepCount = 2;

			kernel.parameters.push_back(tempAttribute);
		}

		if (str[0] == "map")
		{
			tempMapData.name = str[1];
			int structSettingsCount = atoi(str[2].c_str()); // Settings by the struct
			int currentSettingsCount = mapSettingsCount;
			tempMapData.settingsOffset = mapSettingsCount;
			mapSettingsCount += structSettingsCount;
			tempMapData.settingsCount = structSettingsCount;
			int writtenSettingsCount = ((int)str.size() - 3) / 2; // Settings in the config file

			for (int i = 0; i < writtenSettingsCount; i++)
			{
				tempMapData.settingName[i] = str[3 + i * 2 + 0];

				tempMapData.isSettingNumb[i] = str[3 + i * 2 + 1][0] == 'n';
				str[3 + i * 2 + 1] = str[3 + i * 2 + 1].substr(1);
				kernel.mapSettings[currentSettingsCount++] = (numb)atof(str[3 + i * 2 + 1].c_str());
			}

			tempMapData.typeX = tempMapData.typeY = VARIABLE;
			tempMapData.indexX = tempMapData.indexY = 0;

			kernel.mapDatas.push_back(tempMapData);
		}
	}

	fileStream.close();

	kernel.calcAttributeCounts();
	kernel.mapWeight = 1.0f;
	return kernel;
}