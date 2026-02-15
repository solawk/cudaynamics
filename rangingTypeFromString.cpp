#include "rangingTypeFromString.h"

RangingType rangingTypeFromString(std::string str)
{
	if (str == rangingTypes[RT_None]) return RT_None;
	if (str == rangingTypes[RT_Step]) return RT_Step;
	if (str == rangingTypes[RT_Linear]) return RT_Linear;
	if (str == rangingTypes[RT_UniformRandom]) return RT_UniformRandom;
	if (str == rangingTypes[RT_NormalRandom]) return RT_NormalRandom;
	if (str == rangingTypes[RT_Enum]) return RT_Enum;

	printf("Invalid ranging type\n");
	throw std::runtime_error("Invalid ranging type");
}