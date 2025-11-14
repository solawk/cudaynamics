#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <ctime>
#include "objects.h"
#include "kernel_struct.h"
#include "splitString.h"

Kernel readKernelText(std::string name);

std::string timeAsString();