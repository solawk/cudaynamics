#pragma once

namespace kernel
{
	extern const char* name;

	enum VARIABLES { x0, x1, x2, VAR_COUNT };
	extern const char* VAR_NAMES[];
	extern float VAR_VALUES[];
	extern bool VAR_RANGING[];
	extern float VAR_STEPS[];
	extern float VAR_MAX[];
	extern int VAR_STEP_COUNTS[];

	enum PARAMETERS { p0, p1, p2, p3, p4, p5, PARAM_COUNT };
	extern const char* PARAM_NAMES[];
	extern float PARAM_VALUES[];
	extern bool PARAM_RANGING[];
	extern float PARAM_STEPS[];
	extern float PARAM_MAX[];
	extern int PARAM_STEP_COUNTS[];

	extern int steps;
	extern bool executeOnLaunch;
	extern float stepSize;
	extern bool onlyShowLast;
}