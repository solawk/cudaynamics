#pragma once
#include "objects.h"

#define MAX_ENUMS 8

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
	bool selectedForMaps = false;

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
		numb* oldValues = values; // if the attribute has been copied, "values" points to the source array of values

		int trueStepCount = TrueStepCount();
		values = new numb[trueStepCount];

		switch (rangingType)
		{
		case RT_None:
			values[0] = min;
			break;
		case RT_Step:
		case RT_Linear:
			if (!preserveValues || oldValues == nullptr)
			{
				for (int i = 0; i < trueStepCount; i++)
				{
					values[i] = min + step * i;
					if (values[i] > max) values[i] = max;
				}
			}
			else // i.e. if (preserveValues && oldValues != nullptr)
			{
				for (int i = 0; i < trueStepCount; i++)
				{
					values[i] = oldValues[i];
				}
			}
			break;
		case RT_Enum:
			if (!preserveValues || oldValues == nullptr)
			{
				int i = 0;
				for (int e = 0; e < enumCount; e++)
				{
					if (enumEnabled[e]) values[i++] = (numb)e;
				}
			}
			else
			{
				for (int i = 0; i < trueStepCount; i++)
				{
					values[i] = oldValues[i];
				}
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
};