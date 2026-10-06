#pragma once

#include <algorithm>
#include <vector>
#include <string>
#include <cmath>

#include "../../port.h"
#include "../../index.h"
#include "../../numb.h"
#include "../../abstractSettings_struct.h"

// Basins of Attraction are always classified in exactly two feature dimensions.
// Keeping this fixed makes the configuration unambiguous and lets the CUDA stage
// use a compact, branch-free 2D distance calculation.
struct BOA_Settings : AbstractAnalysisSettingsStruct
{
    AnalysisIndex features[2];
    numb epsilon;
    int minimumPoints;
    Port basinId;

    BOA_Settings()
    {
        features[0] = IND_MNPEAK;
        features[1] = IND_MNINT;
        epsilon = (numb)0.05;
        minimumPoints = 4;
        basinId = Port();
    }

    static const char* FeatureName(AnalysisIndex index)
    {
        switch (index)
        {
        case IND_MIN: return "Minimum variable value";
        case IND_MAX: return "Maximum variable value";
        case IND_LLE: return "Largest Lyapunov exponent";
        case IND_PERIOD: return "Period";
        case IND_MNMPEAK: return "Minimum peak";
        case IND_MNMINT: return "Minimum interval";
        case IND_MNPEAK: return "Mean peak";
        case IND_MNINT: return "Mean interval";
        case IND_MXMPEAK: return "Maximum peak";
        case IND_MXMINT: return "Maximum interval";
        case IND_PV: return "Phase volume";
        default: return "Invalid feature";
        }
    }

    static bool IsFeature(AnalysisIndex index)
    {
        return index >= IND_MIN && index < IND_BOA;
    }

    void DisplayFeatureSetting(const char* label, int slot)
    {
        ImGui::Text("%s", label);
        ImGui::SameLine();
        ImGui::PushItemWidth(210.0f);
        const std::string id = std::string("##BOA_") + label;
        if (ImGui::BeginCombo(id.c_str(), FeatureName(features[slot])))
        {
            for (int raw = (int)IND_MIN; raw < (int)IND_BOA; ++raw)
            {
                const AnalysisIndex candidate = (AnalysisIndex)raw;
                const bool selected = candidate == features[slot];
                const bool alreadyUsed = candidate == features[1 - slot];
                if (ImGui::Selectable(FeatureName(candidate), selected,
                    alreadyUsed ? ImGuiSelectableFlags_Disabled : 0))
                {
                    features[slot] = candidate;
                }
            }
            ImGui::EndCombo();
        }
        ImGui::PopItemWidth();
    }

    void DisplaySettings(std::vector<Attribute>&)
    {
        DisplayFeatureSetting("Feature 1", 0);
        DisplayFeatureSetting("Feature 2", 1);
        DisplayNumbSetting("Normalized epsilon", epsilon);
        DisplayIntSetting("Minimum points", minimumPoints);
        if (epsilon < (numb)0.000001) epsilon = (numb)0.000001;
        if (minimumPoints < 1) minimumPoints = 1;
    }

    bool setup(std::vector<std::string> s)
    {
        if (!isMapSetupOfCorrectLength(s, 4)) return false;
        const AnalysisIndex f0 = (AnalysisIndex)s2i(s[0]);
        const AnalysisIndex f1 = (AnalysisIndex)s2i(s[1]);
        const numb parsedEpsilon = s2n(s[2]);
        const int parsedMinimumPoints = s2i(s[3]);
        if (!IsFeature(f0) || !IsFeature(f1) || f0 == f1 ||
            !std::isfinite((double)parsedEpsilon) || parsedEpsilon <= 0 || parsedMinimumPoints < 1)
            return false;

        features[0] = f0;
        features[1] = f1;
        epsilon = parsedEpsilon;
        minimumPoints = parsedMinimumPoints;
        return true;
    }

    json::jobject ExportSettings()
    {
        json::jobject j;
        j["name"] = std::string(AnFuncNames[(int)ANF_BOA]);
        std::vector<std::string> s;
        s.push_back(std::to_string((int)features[0]));
        s.push_back(std::to_string((int)features[1]));
        s.push_back(std::to_string(epsilon));
        s.push_back(std::to_string(minimumPoints));
        j["settings"] = s;
        return j;
    }
};
