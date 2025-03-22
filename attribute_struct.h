#pragma once
#include "objects.h"

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

	// Actual generated values of the attribute, if it's ranged
	numb* values;

	// When ranging by step value, calculate step count
	void CalcStepCount()
	{
		if (rangingType == None) stepCount = 1;
		if (rangingType != Step) return;
		stepCount = (int)((max - min) / step + 1);
		if (stepCount < 1) stepCount = 1;
	}

	void CalcStep()
	{
		if (rangingType != Linear) return;
		step = (max - min) / (stepCount - 1);
	}
	
	bool DoValuesExist()
	{
		return values != nullptr;
	}

	void ClearValues()
	{
		if (!DoValuesExist()) return;

		delete[] values;
		values = nullptr;
	}

	void Generate()
	{
		values = new numb[stepCount];

		switch (rangingType)
		{
		case None:
			values[0] = min;
			break;
		case Step:
		case Linear:
			for (int i = 0; i < stepCount; i++)
			{
				values[i] = min + step * i;
				if (values[i] > max) values[i] = max;
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