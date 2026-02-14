#include "rangingTypeFromString.h"

RangingType rangingTypeFromString(std::string str)
{
	if (str == "Fixed") return RT_None;
	if (str == "Linear") return RT_Linear;
	if (str == "Step") return RT_Step;
	if (str == "Random") return RT_UniformRandom;
	if (str == "Normal") return RT_NormalRandom;
	if (str == "Enum") return RT_Enum;

	printf("Invalid ranging type\n");
	throw std::runtime_error("Invalid ranging type");
}