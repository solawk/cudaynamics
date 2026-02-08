#pragma once
#include <vector>
#include <map>
#include <string>
#include <fstream>
#include <ctime>
#include "objects.h"
#include "kernel_struct.h"
#include "splitString.h"
#include "anfuncs.h"
#include "anfunc_names.h"

RangingType rangingTypeFromString(std::string str);

Kernel readKernelText(std::string name);

std::string timeAsString();