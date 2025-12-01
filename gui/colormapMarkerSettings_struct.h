#pragma once
#include "numb.h"

struct ColormapMarkerSettings
{
	numb* values;
	int valueCount;

	int color;
	float width;

	ColormapMarkerSettings(numb* _values, int _valueCount, int _color, float _width)
	{
		values = _values;
		valueCount = _valueCount;
		color = _color;
		width = _width;
	}
};