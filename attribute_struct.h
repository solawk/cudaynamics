#pragma once
#include "objects.h"
#include <string>
#include "json/json.h"
#include "rangingTypeFromString.h"
#include "gui/ui_strings.h"

#define MAX_ENUMS 16

struct Attribute
{
public:
	std::string name;
	RangingType rangingType;
	numb min;
	numb max;
	numb step;
	int stepCount;
	numb mean;
	numb deviation;

	std::string enumNames[MAX_ENUMS];
	bool enumEnabled[MAX_ENUMS];
	int enumCount;

	// Actual generated values of the attribute, if it's ranged
	numb* values = nullptr;

	int TrueStepCount()
	{
		if (rangingType == RT_None) return 1;
		if (rangingType == RT_Enum)
		{
			stepCount = 0;
			for (int e = 0; e < enumCount; e++) if (enumEnabled[e]) stepCount++;
		}
		return stepCount;
	}

	// When ranging by step value, calculate step count
	void CalcStepCount()
	{
		//if (rangingType == None) stepCount = 1;
		if (rangingType == RT_Enum)
		{
			TrueStepCount();
			return;
		}

		if (rangingType != RT_Step) return;
		stepCount = (int)((max - min) / step + 1);
		if (stepCount < 1) stepCount = 1;
	}

	void CalcStep()
	{
		if (rangingType != RT_Linear) return;
		step = (max - min) / (stepCount - 1);
	}
	
	bool DoValuesExist()
	{
		return values != nullptr;
	}

	void ClearValues()
	{
		if (values != nullptr)
		{
			delete[] values;
			values = nullptr;
		}
	}

	void Generate(bool preserveValues)
	{
		//numb* oldValues = values; // if the attribute has been copied, "values" points to the source array of values

		int trueStepCount = TrueStepCount();
		values = new numb[trueStepCount];

		switch (rangingType)
		{
		case RT_None:
			values[0] = min;
			break;
		case RT_Step:
		case RT_Linear:
			for (int i = 0; i < trueStepCount; i++)
			{
				values[i] = min + step * i;
				if (values[i] > max) values[i] = max;
			}
			break;
		case RT_Enum:
			int i = 0;
			for (int e = 0; e < enumCount; e++)
			{
				if (enumEnabled[e]) values[i++] = (numb)e;
			}
			break;
		}
	}

	bool IsDifferentFrom(Attribute* attrib)
	{
		if (min != attrib->min)					return true;
		if (max != attrib->max)					return true;
		if (step != attrib->step)				return true;
		if (stepCount != attrib->stepCount)		return true;
		if (rangingType != attrib->rangingType) return true;
		if (mean != attrib->mean)				return true;
		if (deviation != attrib->deviation)		return true;

		return false;
	}

	json::jobject ExportToJSON()
	{
		json::jobject a;
		a["name"] = name;
		a["ranging"] = rangingTypes[(int)rangingType];
		a["min"] = min;
		a["max"] = max;
		a["step"] = step;
		a["stepCount"] = stepCount;
		return a;
	}

	void ImportFromJSON(json::jobject& j)
	{
		if (j.has_key("ranging"))	rangingType = rangingTypeFromString((std::string)j["ranging"]);
		if (j.has_key("min"))		min = (numb)j["min"];
		if (j.has_key("max"))		max = (numb)j["max"];
		if (j.has_key("step"))		step = (numb)j["step"];
		if (j.has_key("stepCount"))	stepCount = (int)j["stepCount"];
	}
};