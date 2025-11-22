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
                SetupImGuiStyle(appStyle);
            }

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

        ImGui::EndMenuBar();
    }
}