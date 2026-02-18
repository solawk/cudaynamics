#pragma once
#include <map>
#include <string>
#include "../jsonRW.h"
#include "../kernel_map.h"
#include "../computations.h"
#include "../main_utils.h"
#include "../indices_map.h"
#include <fstream>
#include "gui/applicationSettings_struct.h"

json::jobject saveCfg(bool saveHires, bool saveNew);

bool loadCfg(json::jobject cfg, bool cleanStart, bool needPrints);