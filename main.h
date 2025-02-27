#pragma once
#include "imgui_main.h"
#include <objects.h>
#include "analysis.h"

// Choosing the kernel to compile
#define SEL_RLC_SJJ

#ifdef SEL_LORENZ
#include "kernels/lorenz.h"
#endif

#ifdef SEL_RLC_SJJ
#include "kernels/RLC-sJJ.h"
#endif

#ifdef SEL_MRLCs
#include "kernels/MRLCs-JJ.h"
#endif

int compute(void**, void**, numb*, PostRanging*);