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

Kernel readKernelText(std::string name);

std::string timeAsString();

std::map<std::string, std::string> launchOptions(int argc, char** argv);