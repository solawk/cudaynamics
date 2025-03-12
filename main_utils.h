#pragma once
#include <vector>
#include <string>
#include <fstream>
#include "objects.h"
#include "kernel_struct.h"

std::vector<std::string> splitString(std::string str);

Kernel readKernelText(std::string name);