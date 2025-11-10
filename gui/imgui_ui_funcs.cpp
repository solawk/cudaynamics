#include "imgui_ui_funcs.h"

extern std::string rangingTypes[];
extern std::string rangingDescriptions[];
extern bool popStyle;
extern bool playingParticles;
extern PlotWindow* hiresHeatmapWindow;
extern bool autoLoadNewParams;
extern float dragChangeSpeed;
extern ImGuiSliderFlags dragFlag;
extern bool thisChanged;
extern bool anyChanged;
extern int maxNameLength;
extern Kernel kernelNew;
extern Kernel kernelHiresNew;

#define ATTR_BEGIN  ImGui::SameLine(); popStyle = false; \
    if (hiresHeatmapWindow == nullptr) { if (isChanged && !autoLoadNewParams) { PUSH_UNSAVED_FRAME; popStyle = true; } } \
    else { PUSH_HIRES_FRAME; popStyle = true; }
#define ATTR_END    ImGui::PopItemWidth(); if (popStyle) POP_FRAME(3);

void listAttrRanging(Attribute* attr, bool isChanged)
{
    ATTR_BEGIN;
    ImGui::PushItemWidth(120.0f);
    if (ImGui::BeginCombo(("##RANGING_" + attr->name).c_str(), (rangingTypes[attr->rangingType]).c_str()))
    {
        for (int r = 0; r < /*5*/3; r++) // 3 without randoms, 5 with randoms
        {
            bool isSelected = attr->rangingType == r;
            ImGuiSelectableFlags selectableFlags = 0;
            if (ImGui::Selectable(rangingTypes[r].c_str(), isSelected, selectableFlags) && !playingParticles)
            {
                attr->rangingType = (RangingType)r;
            }
            TOOLTIP(rangingDescriptions[r].c_str());
        }

        ImGui::EndCombo();
    }
    ATTR_END;
}

void listAttrNumb(Attribute* attr, numb* field, std::string name, std::string inner, bool isChanged)
{
    ATTR_BEGIN;
    ImGui::PushItemWidth(200.0f);
    float var = (float)(*field);
    ImGui::DragFloat(("##" + name + attr->name).c_str(), &var, dragChangeSpeed, 0.0f, 0.0f, (inner + "%f").c_str(), dragFlag);
    (*field) = (numb)var;
    ATTR_END;
}

void listAttrInt(Attribute* attr, int* field, std::string name, std::string inner, bool isChanged, int minimum)
{
    ATTR_BEGIN;
    ImGui::PushItemWidth(200.0f);
    ImGui::DragInt(("##" + name + attr->name).c_str(), field, dragChangeSpeed, 0, 0, (inner + "%i").c_str(), dragFlag);
    if (*field < minimum) *field = minimum;
    ATTR_END;
}

void listVariable(int i)
{
    thisChanged = false;
    if (KERNELNEWCURRENT.variables[i].IsDifferentFrom(&(KERNEL.variables[i]))) { anyChanged = true; thisChanged = true; }
    //if (thisChanged) varNew.recountSteps(i); // TODO

    ImGui::Text(padString(KERNEL.variables[i].name, maxNameLength).c_str());

    if (playingParticles)
    {
        ImGui::PushStyleColor(ImGuiCol_Text, CUSTOM_COLOR(DisabledText));
        PUSH_DISABLED_FRAME;
    }

    // Ranging
    listAttrRanging(&(KERNELNEWCURRENT.variables[i]), KERNELNEWCURRENT.variables[i].rangingType != KERNEL.variables[i].rangingType);

    dragFlag = !playingParticles ? 0 : ImGuiSliderFlags_ReadOnly;

    switch (KERNELNEWCURRENT.variables[i].rangingType)
    {
    case RT_None:
        listAttrNumb(&(KERNELNEWCURRENT.variables[i]), &(KERNELNEWCURRENT.variables[i].min), "", "", KERNELNEWCURRENT.variables[i].min != KERNEL.variables[i].min);
        break;
    case RT_Step:
        listAttrNumb(&(KERNELNEWCURRENT.variables[i]), &(KERNELNEWCURRENT.variables[i].min), "", "Min: ", KERNELNEWCURRENT.variables[i].min != KERNEL.variables[i].min);
        listAttrNumb(&(KERNELNEWCURRENT.variables[i]), &(KERNELNEWCURRENT.variables[i].max), "MAX", "Max: ", KERNELNEWCURRENT.variables[i].max != KERNEL.variables[i].max);
        listAttrNumb(&(KERNELNEWCURRENT.variables[i]), &(KERNELNEWCURRENT.variables[i].step), "STEP", "Step: ", KERNELNEWCURRENT.variables[i].step != KERNEL.variables[i].step);
        KERNELNEWCURRENT.variables[i].CalcStepCount();
        ImGui::SameLine(); ImGui::Text((std::to_string(KERNELNEWCURRENT.variables[i].stepCount) + " steps").c_str());
        break;
    case RT_Linear:
        listAttrNumb(&(KERNELNEWCURRENT.variables[i]), &(KERNELNEWCURRENT.variables[i].min), "", "Min: ", KERNELNEWCURRENT.variables[i].min != KERNEL.variables[i].min);
        listAttrNumb(&(KERNELNEWCURRENT.variables[i]), &(KERNELNEWCURRENT.variables[i].max), "MAX", "Max: ", KERNELNEWCURRENT.variables[i].max != KERNEL.variables[i].max);
        listAttrInt(&(KERNELNEWCURRENT.variables[i]), &(KERNELNEWCURRENT.variables[i].stepCount), "STEPCOUNT", "Count: ", KERNELNEWCURRENT.variables[i].stepCount != KERNEL.variables[i].stepCount, 2);
        KERNELNEWCURRENT.variables[i].CalcStep();
        ImGui::SameLine(); ImGui::Text(("Step: " + (std::to_string(KERNELNEWCURRENT.variables[i].step))).c_str());
        break;
    case RT_UniformRandom:
        listAttrNumb(&(KERNELNEWCURRENT.variables[i]), &(KERNELNEWCURRENT.variables[i].mean), "MEAN", "Mean: ", KERNELNEWCURRENT.variables[i].mean != KERNEL.variables[i].mean);
        listAttrNumb(&(KERNELNEWCURRENT.variables[i]), &(KERNELNEWCURRENT.variables[i].deviation), "DEV", "Dev: ", KERNELNEWCURRENT.variables[i].deviation != KERNEL.variables[i].deviation);
        listAttrInt(&(KERNELNEWCURRENT.variables[i]), &(KERNELNEWCURRENT.variables[i].stepCount), "STEPCOUNT", "Count: ", KERNELNEWCURRENT.variables[i].stepCount != KERNEL.variables[i].stepCount, 2);
        break;
    case RT_NormalRandom:
        listAttrNumb(&(KERNELNEWCURRENT.variables[i]), &(KERNELNEWCURRENT.variables[i].mean), "MU", "Mu: ", KERNELNEWCURRENT.variables[i].mean != KERNEL.variables[i].mean);
        listAttrNumb(&(KERNELNEWCURRENT.variables[i]), &(KERNELNEWCURRENT.variables[i].deviation), "SIGMA", "Sigma: ", KERNELNEWCURRENT.variables[i].deviation != KERNEL.variables[i].deviation);
        listAttrInt(&(KERNELNEWCURRENT.variables[i]), &(KERNELNEWCURRENT.variables[i].stepCount), "STEPCOUNT", "Count: ", KERNELNEWCURRENT.variables[i].stepCount != KERNEL.variables[i].stepCount, 2);
        break;
    }

    if (playingParticles)
    {
        ImGui::PopStyleColor();
        POP_FRAME(3);
    }
}

void listParameter(int i)
{
    if (!isParameterUnconstrainted(i)) return;

    bool changeAllowed = KERNELNEWCURRENT.parameters[i].rangingType == RT_None || !playingParticles || !autoLoadNewParams;

    thisChanged = false;
    if (KERNELNEWCURRENT.parameters[i].IsDifferentFrom(&(KERNEL.parameters[i]))) { anyChanged = true; thisChanged = true; }
    //if (thisChanged) paramNew.recountSteps(i); // TODO

    ImGui::Text(padString(KERNEL.parameters[i].name, maxNameLength).c_str());

    if (!changeAllowed)
    {
        ImGui::PushStyleColor(ImGuiCol_Text, CUSTOM_COLOR(DisabledText)); // disabledText push
        PUSH_DISABLED_FRAME;
    }

    // Ranging
    if (playingParticles)
    {
        ImGui::PushStyleColor(ImGuiCol_Text, CUSTOM_COLOR(DisabledText));
        PUSH_DISABLED_FRAME;
    }

    listAttrRanging(&(KERNELNEWCURRENT.parameters[i]), KERNELNEWCURRENT.parameters[i].rangingType != KERNEL.parameters[i].rangingType);

    dragFlag = (!playingParticles || KERNELNEWCURRENT.parameters[i].rangingType == RT_None) ? 0 : ImGuiSliderFlags_ReadOnly;

    switch (KERNELNEWCURRENT.parameters[i].rangingType)
    {
    case RT_Step:
        listAttrNumb(&(KERNELNEWCURRENT.parameters[i]), &(KERNELNEWCURRENT.parameters[i].min), "", "Min: ", KERNELNEWCURRENT.parameters[i].min != KERNEL.parameters[i].min);
        listAttrNumb(&(KERNELNEWCURRENT.parameters[i]), &(KERNELNEWCURRENT.parameters[i].max), "MAX", "Max: ", KERNELNEWCURRENT.parameters[i].max != KERNEL.parameters[i].max);
        listAttrNumb(&(KERNELNEWCURRENT.parameters[i]), &(KERNELNEWCURRENT.parameters[i].step), "STEP", "Step: ", KERNELNEWCURRENT.parameters[i].step != KERNEL.parameters[i].step);
        KERNELNEWCURRENT.parameters[i].CalcStepCount();
        ImGui::SameLine(); ImGui::Text((std::to_string(KERNELNEWCURRENT.parameters[i].stepCount) + " steps").c_str());
        break;
    case RT_Linear:
        listAttrNumb(&(KERNELNEWCURRENT.parameters[i]), &(KERNELNEWCURRENT.parameters[i].min), "", "Min: ", KERNELNEWCURRENT.parameters[i].min != KERNEL.parameters[i].min);
        listAttrNumb(&(KERNELNEWCURRENT.parameters[i]), &(KERNELNEWCURRENT.parameters[i].max), "MAX", "Max: ", KERNELNEWCURRENT.parameters[i].max != KERNEL.parameters[i].max);
        listAttrInt(&(KERNELNEWCURRENT.parameters[i]), &(KERNELNEWCURRENT.parameters[i].stepCount), "STEPCOUNT", "Count: ", KERNELNEWCURRENT.parameters[i].stepCount != KERNEL.parameters[i].stepCount, 2);
        KERNELNEWCURRENT.parameters[i].CalcStep();
        ImGui::SameLine(); ImGui::Text(("Step: " + (std::to_string(KERNELNEWCURRENT.parameters[i].step))).c_str());
        break;
    case RT_UniformRandom:
        listAttrNumb(&(KERNELNEWCURRENT.parameters[i]), &(KERNELNEWCURRENT.parameters[i].mean), "MEAN", "Mean: ", KERNELNEWCURRENT.parameters[i].mean != KERNEL.parameters[i].mean);
        listAttrNumb(&(KERNELNEWCURRENT.parameters[i]), &(KERNELNEWCURRENT.parameters[i].deviation), "DEV", "Dev: ", KERNELNEWCURRENT.parameters[i].deviation != KERNEL.parameters[i].deviation);
        listAttrInt(&(KERNELNEWCURRENT.parameters[i]), &(KERNELNEWCURRENT.parameters[i].stepCount), "STEPCOUNT", "Count: ", KERNELNEWCURRENT.parameters[i].stepCount != KERNEL.parameters[i].stepCount, 2);
        break;
    case RT_NormalRandom:
        listAttrNumb(&(KERNELNEWCURRENT.parameters[i]), &(KERNELNEWCURRENT.parameters[i].mean), "MU", "Mu: ", KERNELNEWCURRENT.parameters[i].mean != KERNEL.parameters[i].mean);
        listAttrNumb(&(KERNELNEWCURRENT.parameters[i]), &(KERNELNEWCURRENT.parameters[i].deviation), "SIGMA", "Sigma: ", KERNELNEWCURRENT.parameters[i].deviation != KERNEL.parameters[i].deviation);
        listAttrInt(&(KERNELNEWCURRENT.parameters[i]), &(KERNELNEWCURRENT.parameters[i].stepCount), "STEPCOUNT", "Count: ", KERNELNEWCURRENT.parameters[i].stepCount != KERNEL.parameters[i].stepCount, 2);
        break;
    }

    if (playingParticles)
    {
        ImGui::PopStyleColor();
        POP_FRAME(3);
    }

    if (KERNELNEWCURRENT.parameters[i].rangingType == RT_None)
    {
        listAttrNumb(&(KERNELNEWCURRENT.parameters[i]), &(KERNELNEWCURRENT.parameters[i].min), "", "", KERNELNEWCURRENT.parameters[i].min != KERNEL.parameters[i].min);
    }

    if (!changeAllowed) POP_FRAME(4); // disabledText popped as well
}

// TODO: changing enabled enums should force recomputation
void listEnum(int i)
{
    if (!isParameterUnconstrainted(i)) return;

    bool changeAllowed = KERNELNEWCURRENT.parameters[i].rangingType == RT_None || !playingParticles || !autoLoadNewParams;

    thisChanged = false;
    if (KERNELNEWCURRENT.parameters[i].IsDifferentFrom(&(KERNEL.parameters[i]))) { anyChanged = true; thisChanged = true; }

    ImGui::Text(padString(KERNEL.parameters[i].name, maxNameLength).c_str());

    if (!changeAllowed)
    {
        ImGui::PushStyleColor(ImGuiCol_Text, CUSTOM_COLOR(DisabledText)); // disabledText push
        PUSH_DISABLED_FRAME;
    }

    // Ranging
    if (playingParticles)
    {
        ImGui::PushStyleColor(ImGuiCol_Text, CUSTOM_COLOR(DisabledText));
        PUSH_DISABLED_FRAME;
    }

    dragFlag = (!playingParticles || KERNELNEWCURRENT.parameters[i].rangingType == RT_None) ? 0 : ImGuiSliderFlags_ReadOnly;

    int selectedCount = 0;
    std::string selectedKernelsString;
    for (int e = 0; e < KERNELNEWCURRENT.parameters[i].enumCount; e++) if (KERNELNEWCURRENT.parameters[i].enumEnabled[e])
    {
        selectedKernelsString += (selectedCount == 0 ? "" : ", ") + KERNELNEWCURRENT.parameters[i].enumNames[e];
        selectedCount++;
    }

    if (selectedKernelsString.length() > 52)
        selectedKernelsString = selectedKernelsString.substr(0, 52) + "...";

    ImGui::SameLine();
    ImGui::PushItemWidth(740.0f);
    bool isChanged = false;
    ATTR_BEGIN;
    if (ImGui::BeginCombo(("##ENUMSELECT_" + KERNELNEWCURRENT.parameters[i].name).c_str(), selectedCount == 0 ? "None" : selectedKernelsString.c_str()))
    {
        for (int e = 0; e < KERNELNEWCURRENT.parameters[i].enumCount; e++)
        {
            bool isSelected = KERNELNEWCURRENT.parameters[i].enumEnabled[e];
            if (ImGui::Checkbox(KERNELNEWCURRENT.parameters[i].enumNames[e].c_str(), &(isSelected)) && dragFlag == 0)
            {
                KERNELNEWCURRENT.parameters[i].enumEnabled[e] = !KERNELNEWCURRENT.parameters[i].enumEnabled[e];
            }
            TOOLTIP("Enable this item in computations");
        }

        ImGui::EndCombo();
    }
    ATTR_END;
    //ImGui::PopItemWidth();

    ImGui::SameLine(); ImGui::Text((std::to_string(selectedCount) + " item" + ((selectedCount % 10 != 1 || selectedCount == 11) ? "s" : "")).c_str());

    if (playingParticles)
    {
        ImGui::PopStyleColor();
        POP_FRAME(3);
    }

    if (!changeAllowed) POP_FRAME(4); // disabledText popped as well
}

void mapSelectionCombo(std::string name, int& selectedIndex, bool addEmpty)
{
    /*if (ImGui::BeginCombo(name.c_str(), (selectedIndex == -1 ? "-" : KERNEL.mapDatas[selectedIndex].name.c_str())))
    {
        for (int m = (addEmpty ? -1 : 0); m < KERNEL.MAP_COUNT; m++)
        {
            bool isSelected = selectedIndex == m;
            ImGuiSelectableFlags selectableFlags = 0;

            if (selectedIndex == m) selectableFlags = ImGuiSelectableFlags_Disabled;
            if (m == -1)
            {
                if (ImGui::Selectable(("-##-_" + name).c_str(), isSelected, selectableFlags)) selectedIndex = m;
            }
            else
            {
                if (ImGui::Selectable((KERNEL.mapDatas[m].name + "##" + KERNEL.mapDatas[m].name + "_" + name).c_str(), isSelected, selectableFlags)) selectedIndex = m;
            }
        }
        ImGui::EndCombo();
    }*/

    if (ImGui::BeginCombo(name.c_str(), (selectedIndex == -1 ? "-" : indices[(AnalysisIndex)selectedIndex].name.c_str())))
    {
        for (int i = (addEmpty ? -1 : 0); i < indices.size(); i++)
        {
            bool isSelected = selectedIndex == i;
            ImGuiSelectableFlags selectableFlags = 0;

            if (selectedIndex == i) selectableFlags = ImGuiSelectableFlags_Disabled;
            if (i == -1)
            {
                if (ImGui::Selectable(("-##-_" + name).c_str(), isSelected, selectableFlags)) selectedIndex = i;
            }
            else
            {
                if (ImGui::Selectable((indices[(AnalysisIndex)i].name + 
                    "##" + indices[(AnalysisIndex)i].name + "_" + name).c_str(), isSelected, selectableFlags)) selectedIndex = i;
            }
        }
        ImGui::EndCombo();
    }
}

void mapValueSelectionCombo(int index, int channelIndex, std::string windowName, HeatmapProperties* heatmap)
{
    if (index == -1) return;
    MapData* mapData = &(KERNEL.mapDatas[index]);
    bool isSingleValue = mapData->valueCount == 1;
    if (!isSingleValue)
    {
        ImGui::DragInt(("##" + windowName + "_index" + std::to_string(index) + "valueInChannel" + std::to_string(channelIndex)).c_str(),
            channelIndex == -1 ? &(heatmap->values.mapValueIndex) : &(heatmap->channel[channelIndex].mapValueIndex), 1.0f, 0, mapData->valueCount - 1, "%d", 0);
    }
}

bool isParameterUnconstrainted(int index)
{
    if (KERNELNEWCURRENT.stepType == ST_Parameter && index == KERNELNEWCURRENT.PARAM_COUNT - 1) return true;

    Attribute* param = &(KERNELNEWCURRENT.parameters[index]);
    Constraint* constraint = &(KERNELNEWCURRENT.constraints[index]);
    if (!constraint->hasConstraints) return true;

    bool isUnconstrained = false;
    for (int i = 0; i < constraint->count; i++)
    {
        int lhs = findParameterByName(constraint->lhs[i]);
        std::string enumName = constraint->rhs[i];
        if (isEnumEnabledByString(KERNELNEWCURRENT.parameters[lhs], enumName)) return true;
    }

    return isUnconstrained;
}