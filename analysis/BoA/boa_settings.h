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
    bool parameterSweep;
    int sweepParameter;
    numb sweepTrackingTolerance;
    Port basinId;

    BOA_Settings()
    {
        features[0] = IND_MNPEAK;
        features[1] = IND_MNINT;
        epsilon = (numb)0.05;
        minimumPoints = 4;
        parameterSweep = false;
        sweepParameter = -1;
        sweepTrackingTolerance = (numb)0.35;
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

    static bool IsSelectableSweepParameter(const std::vector<Attribute>& parameters,
        int index, bool lastParameterIsStep)
    {
        if (index < 0 || index >= (int)parameters.size()) return false;
        if (lastParameterIsStep && index == (int)parameters.size() - 1) return false;
        return const_cast<Attribute&>(parameters[index]).TrueStepCount() > 1;
    }

    void DisplaySweepParameterSetting(std::vector<Attribute>& parameters, bool lastParameterIsStep,
        const char* id)
    {
        if (!parameterSweep) return;
        const char* preview = IsSelectableSweepParameter(parameters, sweepParameter, lastParameterIsStep)
            ? parameters[sweepParameter].name.c_str() : "Select a ranged parameter";
        ImGui::Text("Sweep parameter");
        ImGui::SameLine();
        ImGui::PushItemWidth(210.0f);
        if (ImGui::BeginCombo(id, preview))
        {
            for (int p = 0; p < (int)parameters.size(); ++p)
            {
                if (!IsSelectableSweepParameter(parameters, p, lastParameterIsStep)) continue;
                const bool selected = p == sweepParameter;
                if (ImGui::Selectable(parameters[p].name.c_str(), selected)) sweepParameter = p;
            }
            ImGui::EndCombo();
        }
        ImGui::PopItemWidth();
        if (!IsSelectableSweepParameter(parameters, sweepParameter, lastParameterIsStep))
            ImGui::TextWrapped("Set a non-step parameter to a range of at least two values.");
    }

    void DisplaySettings(std::vector<Attribute>&, std::vector<Attribute>& parameters, bool lastParameterIsStep)
    {
        DisplayFeatureSetting("Feature 1", 0);
        DisplayFeatureSetting("Feature 2", 1);
        DisplayNumbSetting("Normalized epsilon", epsilon);
        DisplayIntSetting("Minimum points", minimumPoints);
        if (ImGui::Checkbox("Parameter sweep", &parameterSweep) && parameterSweep &&
            !IsSelectableSweepParameter(parameters, sweepParameter, lastParameterIsStep))
        {
            for (int p = 0; p < (int)parameters.size(); ++p)
                if (IsSelectableSweepParameter(parameters, p, lastParameterIsStep))
                {
                    sweepParameter = p;
                    break;
                }
        }
        DisplaySweepParameterSetting(parameters, lastParameterIsStep, "##BOA_sweep_parameter_settings");
        if (parameterSweep)
        {
            DisplayNumbSetting("Sweep tracking tolerance", sweepTrackingTolerance);
            ImGui::TextWrapped("Higher values preserve attractor IDs through larger changes; lower values detect births and deaths more strictly.");
        }
        if (epsilon < (numb)0.000001) epsilon = (numb)0.000001;
        if (minimumPoints < 1) minimumPoints = 1;
        if (sweepTrackingTolerance < (numb)0.000001) sweepTrackingTolerance = (numb)0.000001;
    }

    bool setup(std::vector<std::string> s)
    {
        // Four and six fields are legacy BoA formats. The seventh field stores
        // the user-controlled sweep tracking tolerance.
        if (s.size() != 4 && s.size() != 6 && s.size() != 7) return false;
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
        parameterSweep = s.size() >= 6 ? s2i(s[4]) != 0 : false;
        sweepParameter = s.size() >= 6 ? s2i(s[5]) : -1;
        sweepTrackingTolerance = s.size() == 7 ? s2n(s[6]) : (numb)0.35;
        if (sweepParameter < -1) return false;
        if (!std::isfinite((double)sweepTrackingTolerance) || sweepTrackingTolerance <= 0) return false;
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
        s.push_back(parameterSweep ? "1" : "0");
        s.push_back(std::to_string(sweepParameter));
        s.push_back(std::to_string(sweepTrackingTolerance));
        j["settings"] = s;
        return j;
    }
};
