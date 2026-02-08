#pragma once
#include <map>
#include <string>
#include "../jsonRW.h"
#include "../kernel_map.h"
#include "../computations.h"
#include "../main_utils.h"
#include "../indices_map.h"
#include <fstream>

extern json::jobject launchConfig;
extern std::string exportPath;
extern bool launchedAsOneShot;

std::map<std::string, std::string> launchOptions(int argc, char** argv);

bool readLaunchOptions(int argc, char** argv);

bool launchOneShotComputation();

void exportHires();