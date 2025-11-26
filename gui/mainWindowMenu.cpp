#include "mainWindowMenu.h"

void mainWindowMenu()
{
	if (ImGui::BeginMenuBar())
	{
        if (ImGui::BeginMenu("View"))
        {
            ImGui::SeparatorText("Appearance");
            bool isDark = appStyle == ImGuiCustomStyle::Dark;
            if (ImGui::Checkbox("Dark theme", &isDark))
            {
                if (isDark) appStyle = ImGuiCustomStyle::Dark;
                else appStyle = ImGuiCustomStyle::Light;
                //SetupImGuiStyle(appStyle);
            }

            ImGui::ColorEdit4("##cudaColorPicker", (float*)(&cudaColor)); ImGui::SameLine(); ImGui::Text("CUDA mode color");
            ImGui::ColorEdit4("##openmpColorPicker", (float*)(&openmpColor)); ImGui::SameLine(); ImGui::Text("OpenMP mode color");
            ImGui::ColorEdit4("##hiresColorPicker", (float*)(&hiresColor)); ImGui::SameLine(); ImGui::Text("Hi-Res mode color");

            ImGui::SeparatorText("Font");
            FontMenu(nullptr);

            ImGui::SeparatorText("Attributes");

            ImGui::SetNextItemWidth(150.0f);
            ImGui::InputFloat("Value drag speed", &(dragChangeSpeed));
            TOOLTIP("Drag speed of attribute values, allows for precise automatic parameter setting");

            bool tempPreciseNumbDrags = preciseNumbDrags;
            if (ImGui::Checkbox("Precise numb values", &tempPreciseNumbDrags))
            {
                preciseNumbDrags = tempPreciseNumbDrags;
            }
            TOOLTIP("Enable 12-digit fraction for attribute values");

            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Config"))
        {
            ImGui::SeparatorText("CUDA/OpenMP");

            bool tempCPUinter_mode = CPU_mode_interactive;
            ImGui::Checkbox("Use OpenMP in interactive mode", &tempCPUinter_mode);
            TOOLTIP("Use CPU with OpenMP instead of GPU with CUDA in non-Hi-Res mode");
            CPU_mode_interactive = tempCPUinter_mode;

            bool tempCPUhires_mode = CPU_mode_hires;
            ImGui::Checkbox("Use OpenMP in Hi-Res mode", &tempCPUhires_mode);
            TOOLTIP("Use CPU with OpenMP instead of GPU with CUDA in Hi-Res mode");
            CPU_mode_hires = tempCPUhires_mode;

            ImGui::SeparatorText("Analysis");

            // TODO: Re-enable this as a setting that requires recomputation
            /*popStyle = false;
            if (kernelNew.mapWeight != KERNEL.mapWeight)
            {
                anyChanged = true;
                PUSH_UNSAVED_FRAME;
                popStyle = true;
            }*/
            float tempContinuousMaps = kernelNew.mapWeight;
            ImGui::InputFloat("Map weight", &tempContinuousMaps);
            kernelNew.mapWeight = tempContinuousMaps;
            TOOLTIP("1.0 to create new map each buffer, 0.0 to continuously calculate the average, 0.0-1.0 defines the weight of each new map");
            //if (popStyle) POP_FRAME(3);

            ImGui::EndMenu();
        }

        ImGui::EndMenuBar();
    }
}