#pragma once
#include "analysisSettingsHeaders.h"

struct AnalysesSettings
{
	MINMAX_Settings MINMAX;
	LLE_Settings LLE;
	DBSCAN_Settings PERIOD;
	PV_Settings PV;

	AnalysesSettings() {}

	void Used2ToCompute()
	{
		MINMAX.toCompute = MINMAX.minimum.used || MINMAX.maximum.used;
		LLE.toCompute = LLE.LLE.used;
		PERIOD.toCompute = PERIOD.periodicity.used || PERIOD.minimumPeak.used || PERIOD.minimumInterval.used || PERIOD.meanInterval.used || PERIOD.meanPeak.used || PERIOD.maximumPeak.used || PERIOD.maximumInterval.used ;
		PV.toCompute = PV.PV.used;
	}
};