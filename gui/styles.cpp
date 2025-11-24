#include "styles.h"

void SetupImGuiStyle(ImGuiCustomStyle cs)
{
    ImGuiStyle& style = ImGui::GetStyle();
    ImPlot3DStyle& styleImplot3d = ImPlot3D::GetStyle();

    // light style from Pacôme Danhiez (user itamago) https://github.com/ocornut/imgui/pull/511#issuecomment-175719267
    style.Alpha = 1.0f;
    style.FrameRounding = 3.0f;

    switch (cs)
    {
    case ImGuiCustomStyle::Dark:
        style.Colors[ImGuiCol_Text] = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
        style.Colors[ImGuiCol_TextDisabled] = ImVec4(0.60f, 0.60f, 0.60f, 1.00f);
        style.Colors[ImGuiCol_WindowBg] = ImVec4(0.94f, 0.94f, 0.94f, 0.94f);
        //style.Colors[ImGuiCol_ChildWindowBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
        style.Colors[ImGuiCol_PopupBg] = ImVec4(1.00f, 1.00f, 1.00f, 0.94f);
        style.Colors[ImGuiCol_Border] = ImVec4(0.00f, 0.00f, 0.00f, 0.39f);
        style.Colors[ImGuiCol_BorderShadow] = ImVec4(1.00f, 1.00f, 1.00f, 0.10f);
        style.Colors[ImGuiCol_FrameBg] = ImVec4(0.90f, 0.90f, 0.90f, 0.94f);
        style.Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.40f);
        style.Colors[ImGuiCol_FrameBgActive] = ImVec4(0.26f, 0.59f, 0.98f, 0.67f);
        style.Colors[ImGuiCol_Tab] = style.Colors[ImGuiCol_TabDimmed] = style.Colors[ImGuiCol_TabDimmedSelected] = style.Colors[ImGuiCol_TabDimmedSelectedOverline] =
            style.Colors[ImGuiCol_TabSelected] = style.Colors[ImGuiCol_TabSelectedOverline] = ImVec4(0.26f, 0.59f, 0.98f, 0.60f);
        style.Colors[ImGuiCol_TabHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.80f);
        style.Colors[ImGuiCol_TitleBg] = ImVec4(0.96f, 0.96f, 0.96f, 1.00f);
        style.Colors[ImGuiCol_TitleBgCollapsed] = ImVec4(1.00f, 1.00f, 1.00f, 0.51f);
        style.Colors[ImGuiCol_TitleBgActive] = ImVec4(0.82f, 0.82f, 0.82f, 1.00f);
        style.Colors[ImGuiCol_MenuBarBg] = ImVec4(0.86f, 0.86f, 0.86f, 1.00f);
        style.Colors[ImGuiCol_ScrollbarBg] = ImVec4(0.98f, 0.98f, 0.98f, 0.53f);
        style.Colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.69f, 0.69f, 0.69f, 1.00f);
        style.Colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.59f, 0.59f, 0.59f, 1.00f);
        style.Colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.49f, 0.49f, 0.49f, 1.00f);
        //style.Colors[ImGuiCol_ComboBg] = ImVec4(0.86f, 0.86f, 0.86f, 0.99f);
        style.Colors[ImGuiCol_CheckMark] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
        style.Colors[ImGuiCol_SliderGrab] = ImVec4(0.24f, 0.52f, 0.88f, 1.00f);
        style.Colors[ImGuiCol_SliderGrabActive] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
        style.Colors[ImGuiCol_Button] = ImVec4(0.26f, 0.59f, 0.98f, 0.40f);
        style.Colors[ImGuiCol_ButtonHovered] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
        style.Colors[ImGuiCol_ButtonActive] = ImVec4(0.06f, 0.53f, 0.98f, 1.00f);
        style.Colors[ImGuiCol_Header] = ImVec4(0.26f, 0.59f, 0.98f, 0.31f);
        style.Colors[ImGuiCol_HeaderHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.80f);
        style.Colors[ImGuiCol_HeaderActive] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
        //style.Colors[ImGuiCol_Column] = ImVec4(0.39f, 0.39f, 0.39f, 1.00f);
        //style.Colors[ImGuiCol_ColumnHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.78f);
        //style.Colors[ImGuiCol_ColumnActive] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
        style.Colors[ImGuiCol_ResizeGrip] = ImVec4(1.00f, 1.00f, 1.00f, 0.50f);
        style.Colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.67f);
        style.Colors[ImGuiCol_ResizeGripActive] = ImVec4(0.26f, 0.59f, 0.98f, 0.95f);
        //style.Colors[ImGuiCol_CloseButton] = ImVec4(0.59f, 0.59f, 0.59f, 0.50f);
        //style.Colors[ImGuiCol_CloseButtonHovered] = ImVec4(0.98f, 0.39f, 0.36f, 1.00f);
        //style.Colors[ImGuiCol_CloseButtonActive] = ImVec4(0.98f, 0.39f, 0.36f, 1.00f);
        style.Colors[ImGuiCol_PlotLines] = ImVec4(0.39f, 0.39f, 0.39f, 1.00f);
        style.Colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.00f, 0.43f, 0.35f, 1.00f);
        style.Colors[ImGuiCol_PlotHistogram] = ImVec4(0.90f, 0.70f, 0.00f, 1.00f);
        style.Colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.00f, 0.60f, 0.00f, 1.00f);
        style.Colors[ImGuiCol_TextSelectedBg] = ImVec4(0.26f, 0.59f, 0.98f, 0.35f);
        //style.Colors[ImGuiCol_ModalWindowDarkening] = ImVec4(0.20f, 0.20f, 0.20f, 0.35f);
        style.Colors[ImGuiCol_C_Disabled] = ImVec4(0.137f * 0.5f, 0.271f * 0.5f, 0.427f * 0.5f, 1.0f);
        style.Colors[ImGuiCol_C_DisabledBg] = ImVec4(0.5f, 0.5f, 0.5f, 0.5f);
        style.Colors[ImGuiCol_C_DisabledText] = ImVec4(0.00f, 0.00f, 0.00f, 0.5f);
        styleImplot3d.Colors[ImPlot3DCol_FrameBg] = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);

        for (int i = 0; i <= ImGuiCol_COUNT; i++)
        {
            ImVec4& col = style.Colors[i];
            float Hcomp, Scomp, Vcomp;
            ImGui::ColorConvertRGBtoHSV(col.x, col.y, col.z, Hcomp, Scomp, Vcomp);

            if (Hcomp < 0.1f)
            {
                Vcomp = 1.0f - Vcomp;
            }
            ImGui::ColorConvertHSVtoRGB(Hcomp, Scomp, Vcomp, col.x, col.y, col.z);
        }

        style.Colors[ImGuiCol_C_Unsaved] = ImVec4(0.427f, 0.427f, 0.137f, 1.0f);
        style.Colors[ImGuiCol_C_UnsavedHovered] = ImVec4(0.427f * 1.3f, 0.427f * 1.3f, 0.137f * 1.3f, 1.0f);
        style.Colors[ImGuiCol_C_UnsavedActive] = ImVec4(0.427f * 1.5f, 0.427f * 1.5f, 0.137f * 1.5f, 1.0f);
        style.Colors[ImGuiCol_C_Hires] = ImVec4(0.427f, 0.137f, 0.427f, 1.0f);
        style.Colors[ImGuiCol_C_HiresHovered] = ImVec4(0.427f * 1.3f, 0.137f * 1.3f, 0.427f * 1.3f, 1.0f);
        style.Colors[ImGuiCol_C_HiresActive] = ImVec4(0.427f * 1.5f, 0.137f * 1.5f, 0.427f * 1.5f, 1.0f);
        style.Colors[ImGuiCol_C_XAxis] = ImVec4(0.75f, 0.3f, 0.3f, 1.0f);
        style.Colors[ImGuiCol_C_YAxis] = ImVec4(0.33f, 0.67f, 0.4f, 1.0f);
        style.Colors[ImGuiCol_C_ZAxis] = ImVec4(0.3f, 0.45f, 0.7f, 1.0f);

        break;
    case ImGuiCustomStyle::Light:

        style.Colors[ImGuiCol_Text] = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
        style.Colors[ImGuiCol_TextDisabled] = ImVec4(0.60f, 0.60f, 0.60f, 1.00f);
        style.Colors[ImGuiCol_WindowBg] = ImVec4(0.94f, 0.94f, 0.94f, 0.94f);
        //style.Colors[ImGuiCol_ChildWindowBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
        style.Colors[ImGuiCol_PopupBg] = ImVec4(1.00f, 1.00f, 1.00f, 0.94f);
        style.Colors[ImGuiCol_Border] = ImVec4(0.00f, 0.00f, 0.00f, 0.39f);
        style.Colors[ImGuiCol_BorderShadow] = ImVec4(1.00f, 1.00f, 1.00f, 0.10f);
        style.Colors[ImGuiCol_FrameBg] = ImVec4(0.90f, 0.90f, 0.90f, 0.94f);
        style.Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.40f);
        style.Colors[ImGuiCol_FrameBgActive] = ImVec4(0.26f, 0.59f, 0.98f, 0.67f);
        style.Colors[ImGuiCol_Tab] = style.Colors[ImGuiCol_TabDimmed] = style.Colors[ImGuiCol_TabDimmedSelected] = style.Colors[ImGuiCol_TabDimmedSelectedOverline] =
            style.Colors[ImGuiCol_TabSelected] = style.Colors[ImGuiCol_TabSelectedOverline] = ImVec4(0.26f, 0.59f, 0.98f, 0.60f);
        style.Colors[ImGuiCol_TabHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.80f);
        style.Colors[ImGuiCol_TitleBg] = ImVec4(0.96f, 0.96f, 0.96f, 1.00f);
        style.Colors[ImGuiCol_TitleBgCollapsed] = ImVec4(1.00f, 1.00f, 1.00f, 0.51f);
        style.Colors[ImGuiCol_TitleBgActive] = ImVec4(0.82f, 0.82f, 0.82f, 1.00f);
        style.Colors[ImGuiCol_MenuBarBg] = ImVec4(0.86f, 0.86f, 0.86f, 1.00f);
        style.Colors[ImGuiCol_ScrollbarBg] = ImVec4(0.98f, 0.98f, 0.98f, 0.53f);
        style.Colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.69f, 0.69f, 0.69f, 1.00f);
        style.Colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.59f, 0.59f, 0.59f, 1.00f);
        style.Colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.49f, 0.49f, 0.49f, 1.00f);
        //style.Colors[ImGuiCol_ComboBg] = ImVec4(0.86f, 0.86f, 0.86f, 0.99f);
        style.Colors[ImGuiCol_CheckMark] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
        style.Colors[ImGuiCol_SliderGrab] = ImVec4(0.24f, 0.52f, 0.88f, 1.00f);
        style.Colors[ImGuiCol_SliderGrabActive] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
        style.Colors[ImGuiCol_Button] = ImVec4(0.26f, 0.59f, 0.98f, 0.40f);
        style.Colors[ImGuiCol_ButtonHovered] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
        style.Colors[ImGuiCol_ButtonActive] = ImVec4(0.06f, 0.53f, 0.98f, 1.00f);
        style.Colors[ImGuiCol_Header] = ImVec4(0.26f, 0.59f, 0.98f, 0.31f);
        style.Colors[ImGuiCol_HeaderHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.80f);
        style.Colors[ImGuiCol_HeaderActive] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
        //style.Colors[ImGuiCol_Column] = ImVec4(0.39f, 0.39f, 0.39f, 1.00f);
        //style.Colors[ImGuiCol_ColumnHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.78f);
        //style.Colors[ImGuiCol_ColumnActive] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
        style.Colors[ImGuiCol_ResizeGrip] = ImVec4(1.00f, 1.00f, 1.00f, 0.50f);
        style.Colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.67f);
        style.Colors[ImGuiCol_ResizeGripActive] = ImVec4(0.26f, 0.59f, 0.98f, 0.95f);
        //style.Colors[ImGuiCol_CloseButton] = ImVec4(0.59f, 0.59f, 0.59f, 0.50f);
        //style.Colors[ImGuiCol_CloseButtonHovered] = ImVec4(0.98f, 0.39f, 0.36f, 1.00f);
        //style.Colors[ImGuiCol_CloseButtonActive] = ImVec4(0.98f, 0.39f, 0.36f, 1.00f);
        style.Colors[ImGuiCol_PlotLines] = ImVec4(0.39f, 0.39f, 0.39f, 1.00f);
        style.Colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.00f, 0.43f, 0.35f, 1.00f);
        style.Colors[ImGuiCol_PlotHistogram] = ImVec4(0.90f, 0.70f, 0.00f, 1.00f);
        style.Colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.00f, 0.60f, 0.00f, 1.00f);
        style.Colors[ImGuiCol_TextSelectedBg] = ImVec4(0.26f, 0.59f, 0.98f, 0.35f);
        //style.Colors[ImGuiCol_ModalWindowDarkening] = ImVec4(0.20f, 0.20f, 0.20f, 0.35f);
        styleImplot3d.Colors[ImPlot3DCol_FrameBg] = ImVec4(0.96f, 0.96f, 0.96f, 1.00f);

        style.Colors[ImGuiCol_C_Unsaved] = ImVec4(0.827f, 0.827f, 0.437f, 1.0f);
        style.Colors[ImGuiCol_C_UnsavedHovered] = ImVec4(0.827f * 1.3f, 0.827f * 1.3f, 0.437f * 1.3f, 1.0f);
        style.Colors[ImGuiCol_C_UnsavedActive] = ImVec4(0.827f * 1.5f, 0.827f * 1.5f, 0.437f * 1.5f, 1.0f);
        style.Colors[ImGuiCol_C_Hires] = ImVec4(0.427f * 2, 0.137f * 5, 0.427f * 2, 1.0f);
        style.Colors[ImGuiCol_C_HiresHovered] = ImVec4(0.427f * 1.3f * 2, 0.137f * 1.3f * 5, 0.427f * 1.3f * 2, 1.0f);
        style.Colors[ImGuiCol_C_HiresActive] = ImVec4(0.427f * 1.5f * 2, 0.137f * 1.5f * 5, 0.427f * 1.5f * 2, 1.0f);
        style.Colors[ImGuiCol_C_Disabled] = ImVec4(0.137f * 0.5f, 0.271f * 0.5f, 0.427f * 0.5f, 1.0f);
        style.Colors[ImGuiCol_C_DisabledBg] = ImVec4(0.5f, 0.5f, 0.5f, 0.5f);
        style.Colors[ImGuiCol_C_DisabledText] = ImVec4(0.00f, 0.00f, 0.00f, 0.5f);

        float axisDarkening = 0.7f;
        style.Colors[ImGuiCol_C_XAxis] = ImVec4(0.75f, 0.3f * axisDarkening, 0.3f * axisDarkening, 1.0f);
        style.Colors[ImGuiCol_C_YAxis] = ImVec4(0.33f * axisDarkening, 0.67f, 0.4f * axisDarkening, 1.0f);
        style.Colors[ImGuiCol_C_ZAxis] = ImVec4(0.3f * axisDarkening, 0.45f * axisDarkening, 0.7f, 1.0f);

        break;
    }
}