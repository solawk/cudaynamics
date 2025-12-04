#pragma once
#include "numb.h"

struct ColormapMarkerSettings
{
	numb* values;
	int valueCount;
	bool enabled;
	int color;
	float width;

	ColormapMarkerSettings(bool _enabled, numb* _values, int _valueCount, int _color, float _width)
	{
		enabled = _enabled;
		values = _values;
		valueCount = _valueCount;
		color = _color;
		width = _width;
	}
};