#pragma once
#include "imgui_main.h"
#include <objects.h>

// Choosing the kernel to compile
#define SEL_LORENZ

#ifdef SEL_LORENZ
#include "kernels/lorenz.h"
#endif

#ifdef SEL_RLC_SJJ
#include "kernels/RLC-sJJ.h"
#endif

int compute(void**, void**, float*, PostRanging*);