#include "anfunc2indices.h"

std::vector<AnalysisIndex> anfunc2indices(AnalysisFunction anfunc)
{
	std::vector<AnalysisIndex> result;

	switch (anfunc)
	{
	case ANF_MINMAX:
		result.push_back(IND_MIN);
		result.push_back(IND_MAX);
		break;

	case ANF_LLE:
		result.push_back(IND_LLE);
		break;

	case ANF_PERIOD:
		result.push_back(IND_PERIOD);
		result.push_back(IND_MNPEAK);
		result.push_back(IND_MNINT);
		break;
	}

	return result;
}