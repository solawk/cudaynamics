#pragma once
#include <string>
#include "../heatmapProperties.hpp"
#include "../attribute_struct.h"
#include "../plotWindow.h"
#include "../kernel_struct.h"
#include "../imgui_utils.h"

//#include "imgui_main.hpp"

// UI drawing-related macros
#define CUSTOM_COLOR(c)     ImGui::GetStyle().Colors[ImGuiCol_C_##c]
#define PUSH_DISABLED_FRAME {ImGui::PushStyleColor(ImGuiCol_FrameBg, CUSTOM_COLOR(DisabledBg)); \
                            ImGui::PushStyleColor(ImGuiCol_FrameBgActive, CUSTOM_COLOR(DisabledBg)); \
                            ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, CUSTOM_COLOR(DisabledBg));}
#define PUSH_UNSAVED_FRAME  {ImGui::PushStyleColor(ImGuiCol_FrameBg, CUSTOM_COLOR(Unsaved)); \
                            ImGui::PushStyleColor(ImGuiCol_FrameBgActive, CUSTOM_COLOR(UnsavedActive)); \
                            ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, CUSTOM_COLOR(UnsavedHovered));}
#define PUSH_HIRES_FRAME  {ImGui::PushStyleColor(ImGuiCol_FrameBg, CUSTOM_COLOR(Hires)); \
                            ImGui::PushStyleColor(ImGuiCol_FrameBgActive, CUSTOM_COLOR(HiresActive)); \
                            ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, CUSTOM_COLOR(HiresHovered));}
#define POP_FRAME(n)        {ImGui::PopStyleColor(n);}
#define CLAMP01(x)          if (x < 0.0f) x = 0.0f; if (x > 1.0f) x = 1.0f;
#define TOOLTIP(text)       if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) ImGui::SetTooltip(text);

void listAttrRanging(Attribute* attr, bool isChanged);

void listAttrNumb(Attribute* attr, numb* field, std::string name, std::string inner, bool isChanged);

void listAttrInt(Attribute* attr, int* field, std::string name, std::string inner, bool isChanged, int minimum);

void listVariable(int i);

void listParameter(int i);

void listEnum(int i);

void mapSelectionCombo(std::string name, int& selectedIndex, bool addEmpty);

void mapValueSelectionCombo(int index, int channelIndex, std::string windowName, HeatmapProperties* heatmap);

bool isParameterUnconstrainted(int index);