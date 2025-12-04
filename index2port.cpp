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
	case IND_MNMPEAK:
		return &(analyses.PERIOD.minimumPeak);
	case IND_MNMINT:
		return &(analyses.PERIOD.minimumInterval);
	case IND_MNPEAK:
		return &(analyses.PERIOD.meanPeak);
	case IND_MNINT:
		return &(analyses.PERIOD.meanInterval);
	case IND_MXMPEAK:
		return &(analyses.PERIOD.maximumPeak);
	case IND_MXMINT:
		return &(analyses.PERIOD.maximumInterval);
	case IND_PV:
		return &(analyses.PV.PV);
	}

	return nullptr;
}