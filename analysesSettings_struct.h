#pragma once
#include "analysisSettingsHeaders.h"

struct AnalysesSettings
{
	MINMAX_Settings MINMAX;
	LLE_Settings LLE;
	DBSCAN_Settings PERIOD;

	AnalysesSettings() {}
};