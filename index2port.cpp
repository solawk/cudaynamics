#include "index2port.h"

Port* index2port(AnalysesSettings& analyses, AnalysisIndex index)
{
	switch (index)
	{
	case IND_MIN:
		return &(analyses.MINMAX.minimum);
	case IND_MAX:
		return &(analyses.MINMAX.maximum);
	case IND_LLE:
		return &(analyses.LLE.LLE);
	case IND_PERIOD:
		return &(analyses.PERIOD.periodicity);
	case IND_MNPEAK:
		return &(analyses.PERIOD.meanPeak);
	case IND_MNINT:
		return &(analyses.PERIOD.meanInterval);
	}

	return nullptr;
}