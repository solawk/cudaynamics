#pragma once

// Put 0 for float, 1 for double
#define USE_DOUBLE_PRECISION 0

#if !USE_DOUBLE_PRECISION
	#define numb float
#else
	#define numb double
#endif

#undef USE_DOUBLE_PRECISION

#pragma warning(disable:4305)
#pragma warning(disable:4244)
#pragma warning(disable:4996)