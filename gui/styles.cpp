#include "styles.h"

ImVec4 tcol(ImVec4 thm, float rorig, float gorig, float borig, float aorig)
{
	return ImVec4(thm.x * (rorig / 0.16f), thm.y * (gorig / 0.36f), thm.z * (borig / 0.60f), thm.w * (aorig / 1.0f));
}

void SetupImGuiStyle(ImGuiCustomStyle cs, ImVec4 normal, ImVec4 hires, ImVec4 cpuMode, bool isHires, bool isCPU)
{
	// Original blue tab color – ImVec4(0.16f, 0.36f, 0.60f, 1.00f)
	ImGuiStyle& style = ImGui::GetStyle();
	ImPlot3DStyle& styleImplot3d = ImPlot3D::GetStyle();

	ImVec4 theme = !isHires ? (!isCPU ? normal : cpuMode) : hires;

	// light style from Pacôme Danhiez (user itamago) https://github.com/ocornut/imgui/pull/511#issuecomment-175719267
	style.Alpha = 1.0f;
	style.FrameRounding = 3.0f;

	// Grayscale colors go first and get inverted
	switch (cs)
	{
	case ImGuiCustomStyle::Dark:
		style.Colors[ImGuiCol_Text] = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
		style.Colors[ImGuiCol_TextDisabled] = ImVec4(0.60f, 0.60f, 0.60f, 1.00f);
		style.Colors[ImGuiCol_WindowBg] = ImVec4(0.94f, 0.94f, 0.94f, 0.94f);
		style.Colors[ImGuiCol_PopupBg] = ImVec4(1.00f, 1.00f, 1.00f, 0.94f);
		style.Colors[ImGuiCol_Border] = ImVec4(0.00f, 0.00f, 0.00f, 0.39f);
		style.Colors[ImGuiCol_BorderShadow] = ImVec4(1.00f, 1.00f, 1.00f, 0.10f);
		style.Colors[ImGuiCol_FrameBg] = ImVec4(0.90f, 0.90f, 0.90f, 0.94f);
		style.Colors[ImGuiCol_TitleBg] = ImVec4(0.96f, 0.96f, 0.96f, 1.00f);
		style.Colors[ImGuiCol_TitleBgCollapsed] = ImVec4(1.00f, 1.00f, 1.00f, 0.51f);
		style.Colors[ImGuiCol_TitleBgActive] = ImVec4(0.82f, 0.82f, 0.82f, 1.00f);
		style.Colors[ImGuiCol_MenuBarBg] = ImVec4(0.86f, 0.86f, 0.86f, 1.00f);
		style.Colors[ImGuiCol_ScrollbarBg] = ImVec4(0.98f, 0.98f, 0.98f, 0.53f);
		style.Colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.69f, 0.69f, 0.69f, 1.00f);
		style.Colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.59f, 0.59f, 0.59f, 1.00f);
		style.Colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.49f, 0.49f, 0.49f, 1.00f);
		style.Colors[ImGuiCol_ResizeGrip] = ImVec4(1.00f, 1.00f, 1.00f, 0.50f);
		style.Colors[ImGuiCol_PlotLines] = ImVec4(0.39f, 0.39f, 0.39f, 1.00f);
		style.Colors[ImGuiCol_C_DisabledBg] = ImVec4(0.5f, 0.5f, 0.5f, 0.5f);
		style.Colors[ImGuiCol_C_DisabledText] = ImVec4(0.00f, 0.00f, 0.00f, 0.5f);
		styleImplot3d.Colors[ImPlot3DCol_FrameBg] = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
		style.Colors[ImGuiCol_C_DisabledBg] = ImVec4(0.5f, 0.5f, 0.5f, 0.5f);
		style.Colors[ImGuiCol_C_DisabledText] = ImVec4(0.00f, 0.00f, 0.00f, 0.5f);

		for (int i = 0; i <= ImGuiCol_COUNT; i++)
		{
			ImVec4& col = style.Colors[i];
			/*float Hcomp, Scomp, Vcomp;
			ImGui::ColorConvertRGBtoHSV(col.x, col.y, col.z, Hcomp, Scomp, Vcomp);
			Vcomp = 1.0f - Vcomp;
			ImGui::ColorConvertHSVtoRGB(Hcomp, Scomp, Vcomp, col.x, col.y, col.z);*/
			col.x = 1.0f - col.x;
			col.y = 1.0f - col.y;
			col.z = 1.0f - col.z;
		}

		break;
	case ImGuiCustomStyle::Light:
		style.Colors[ImGuiCol_Text] = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
		style.Colors[ImGuiCol_TextDisabled] = ImVec4(0.60f, 0.60f, 0.60f, 1.00f);
		style.Colors[ImGuiCol_WindowBg] = ImVec4(0.94f, 0.94f, 0.94f, 0.94f);
		style.Colors[ImGuiCol_PopupBg] = ImVec4(1.00f, 1.00f, 1.00f, 0.94f);
		style.Colors[ImGuiCol_Border] = ImVec4(0.00f, 0.00f, 0.00f, 0.39f);
		style.Colors[ImGuiCol_BorderShadow] = ImVec4(1.00f, 1.00f, 1.00f, 0.10f);
		style.Colors[ImGuiCol_FrameBg] = ImVec4(0.90f, 0.90f, 0.90f, 0.94f);
		style.Colors[ImGuiCol_TitleBg] = ImVec4(0.96f, 0.96f, 0.96f, 1.00f);
		style.Colors[ImGuiCol_TitleBgCollapsed] = ImVec4(1.00f, 1.00f, 1.00f, 0.51f);
		style.Colors[ImGuiCol_TitleBgActive] = ImVec4(0.82f, 0.82f, 0.82f, 1.00f);
		style.Colors[ImGuiCol_MenuBarBg] = ImVec4(0.86f, 0.86f, 0.86f, 1.00f);
		style.Colors[ImGuiCol_ScrollbarBg] = ImVec4(0.98f, 0.98f, 0.98f, 0.53f);
		style.Colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.69f, 0.69f, 0.69f, 1.00f);
		style.Colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.59f, 0.59f, 0.59f, 1.00f);
		style.Colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.49f, 0.49f, 0.49f, 1.00f);
		style.Colors[ImGuiCol_ResizeGrip] = ImVec4(1.00f, 1.00f, 1.00f, 0.50f);
		style.Colors[ImGuiCol_PlotLines] = ImVec4(0.39f, 0.39f, 0.39f, 1.00f);
		style.Colors[ImGuiCol_C_DisabledBg] = ImVec4(0.5f, 0.5f, 0.5f, 0.5f);
		style.Colors[ImGuiCol_C_DisabledText] = ImVec4(0.00f, 0.00f, 0.00f, 0.5f);
		styleImplot3d.Colors[ImPlot3DCol_FrameBg] = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
		style.Colors[ImGuiCol_C_DisabledBg] = ImVec4(0.5f, 0.5f, 0.5f, 0.5f);
		style.Colors[ImGuiCol_C_DisabledText] = ImVec4(0.00f, 0.00f, 0.00f, 0.5f);

		break;
	}
	
	// Non-invertible colors (dependent and independent of the theme color) go next
	// Original blue tab color – ImVec4(0.16f, 0.36f, 0.60f, 1.00f)
	switch (cs)
	{
	case ImGuiCustomStyle::Dark:
		style.Colors[ImGuiCol_FrameBgHovered] = tcol(theme, 0.26f, 0.59f, 0.98f, 0.40f);
		style.Colors[ImGuiCol_FrameBgActive] = tcol(theme, 0.26f, 0.59f, 0.98f, 0.67f);
		style.Colors[ImGuiCol_Tab] = style.Colors[ImGuiCol_TabDimmed] = style.Colors[ImGuiCol_TabDimmedSelected] = style.Colors[ImGuiCol_TabDimmedSelectedOverline] =
			style.Colors[ImGuiCol_TabSelected] = style.Colors[ImGuiCol_TabSelectedOverline] = theme;
		style.Colors[ImGuiCol_TabHovered] = tcol(theme, 0.26f, 0.59f, 0.98f, 0.80f);
		style.Colors[ImGuiCol_CheckMark] = tcol(theme, 0.26f, 0.59f, 0.98f, 1.00f);
		style.Colors[ImGuiCol_SliderGrab] = tcol(theme, 0.24f, 0.52f, 0.88f, 1.00f);
		style.Colors[ImGuiCol_SliderGrabActive] = tcol(theme, 0.26f, 0.59f, 0.98f, 1.00f);
		style.Colors[ImGuiCol_Button] = tcol(theme, 0.26f, 0.59f, 0.98f, 0.40f);
		style.Colors[ImGuiCol_ButtonHovered] = tcol(theme, 0.26f, 0.59f, 0.98f, 1.00f);
		style.Colors[ImGuiCol_ButtonActive] = tcol(theme, 0.06f, 0.53f, 0.98f, 1.00f);
		style.Colors[ImGuiCol_Header] = tcol(theme, 0.26f, 0.59f, 0.98f, 0.31f);
		style.Colors[ImGuiCol_HeaderHovered] = tcol(theme, 0.26f, 0.59f, 0.98f, 0.80f);
		style.Colors[ImGuiCol_HeaderActive] = tcol(theme, 0.26f, 0.59f, 0.98f, 1.00f);
		style.Colors[ImGuiCol_ResizeGripHovered] = tcol(theme, 0.26f, 0.59f, 0.98f, 0.67f);
		style.Colors[ImGuiCol_ResizeGripActive] = tcol(theme, 0.26f, 0.59f, 0.98f, 0.95f);
		style.Colors[ImGuiCol_PlotLinesHovered] = tcol(theme, 1.00f, 0.43f, 0.35f, 1.00f);
		style.Colors[ImGuiCol_PlotHistogram] = tcol(theme, 0.90f, 0.70f, 0.00f, 1.00f);
		style.Colors[ImGuiCol_PlotHistogramHovered] = tcol(theme, 1.00f, 0.60f, 0.00f, 1.00f);
		style.Colors[ImGuiCol_TextSelectedBg] = tcol(theme, 0.26f, 0.59f, 0.98f, 0.35f);
		style.Colors[ImGuiCol_C_Disabled] = tcol(theme, 0.068f, 0.135f, 0.213f, 1.0f);

		style.Colors[ImGuiCol_C_Unsaved] = ImVec4(0.427f, 0.427f, 0.137f, 1.0f);
		style.Colors[ImGuiCol_C_UnsavedHovered] = ImVec4(0.427f * 1.3f, 0.427f * 1.3f, 0.137f * 1.3f, 1.0f);
		style.Colors[ImGuiCol_C_UnsavedActive] = ImVec4(0.427f * 1.5f, 0.427f * 1.5f, 0.137f * 1.5f, 1.0f);
		style.Colors[ImGuiCol_C_Hires] = tcol(theme, 0.427f, 0.137f, 0.427f, 1.0f);
		style.Colors[ImGuiCol_C_HiresHovered] = tcol(theme, 0.427f * 1.3f, 0.137f * 1.3f, 0.427f * 1.3f, 1.0f);
		style.Colors[ImGuiCol_C_HiresActive] = tcol(theme, 0.427f * 1.5f, 0.137f * 1.5f, 0.427f * 1.5f, 1.0f);
		style.Colors[ImGuiCol_C_XAxis] = ImVec4(0.75f, 0.3f, 0.3f, 1.0f);
		style.Colors[ImGuiCol_C_YAxis] = ImVec4(0.33f, 0.67f, 0.4f, 1.0f);
		style.Colors[ImGuiCol_C_ZAxis] = ImVec4(0.3f, 0.45f, 0.7f, 1.0f);

		break;
	case ImGuiCustomStyle::Light:
		style.Colors[ImGuiCol_FrameBgHovered] = tcol(theme, 0.26f, 0.59f, 0.98f, 0.40f);
		style.Colors[ImGuiCol_FrameBgActive] = tcol(theme, 0.26f, 0.59f, 0.98f, 0.67f);
		style.Colors[ImGuiCol_Tab] = style.Colors[ImGuiCol_TabDimmed] = style.Colors[ImGuiCol_TabDimmedSelected] = style.Colors[ImGuiCol_TabDimmedSelectedOverline] =
			style.Colors[ImGuiCol_TabSelected] = style.Colors[ImGuiCol_TabSelectedOverline] = tcol(theme, 0.21f, 0.48f, 0.80f, 1.00f);
		style.Colors[ImGuiCol_TabHovered] = tcol(theme, 0.26f, 0.59f, 0.98f, 0.80f);
		style.Colors[ImGuiCol_CheckMark] = tcol(theme, 0.26f, 0.59f, 0.98f, 1.00f);
		style.Colors[ImGuiCol_SliderGrab] = tcol(theme, 0.24f, 0.52f, 0.88f, 1.00f);
		style.Colors[ImGuiCol_SliderGrabActive] = tcol(theme, 0.26f, 0.59f, 0.98f, 1.00f);
		style.Colors[ImGuiCol_Button] = tcol(theme, 0.26f, 0.59f, 0.98f, 0.40f);
		style.Colors[ImGuiCol_ButtonHovered] = tcol(theme, 0.26f, 0.59f, 0.98f, 1.00f);
		style.Colors[ImGuiCol_ButtonActive] = tcol(theme, 0.06f, 0.53f, 0.98f, 1.00f);
		style.Colors[ImGuiCol_Header] = tcol(theme, 0.26f, 0.59f, 0.98f, 0.31f);
		style.Colors[ImGuiCol_HeaderHovered] = tcol(theme, 0.26f, 0.59f, 0.98f, 0.80f);
		style.Colors[ImGuiCol_HeaderActive] = tcol(theme, 0.26f, 0.59f, 0.98f, 1.00f);
		style.Colors[ImGuiCol_ResizeGripHovered] = tcol(theme, 0.26f, 0.59f, 0.98f, 0.67f);
		style.Colors[ImGuiCol_ResizeGripActive] = tcol(theme, 0.26f, 0.59f, 0.98f, 0.95f);
		style.Colors[ImGuiCol_PlotLinesHovered] = tcol(theme, 1.00f, 0.43f, 0.35f, 1.00f);
		style.Colors[ImGuiCol_PlotHistogram] = tcol(theme, 0.90f, 0.70f, 0.00f, 1.00f);
		style.Colors[ImGuiCol_PlotHistogramHovered] = tcol(theme, 1.00f, 0.60f, 0.00f, 1.00f);
		style.Colors[ImGuiCol_TextSelectedBg] = tcol(theme, 0.26f, 0.59f, 0.98f, 0.35f);
		style.Colors[ImGuiCol_C_Disabled] = tcol(theme, 0.068f, 0.135f, 0.213f, 1.0f);

		style.Colors[ImGuiCol_C_Unsaved] = ImVec4(0.427f, 0.427f, 0.137f, 1.0f);
		style.Colors[ImGuiCol_C_UnsavedHovered] = ImVec4(0.427f * 1.3f, 0.427f * 1.3f, 0.137f * 1.3f, 1.0f);
		style.Colors[ImGuiCol_C_UnsavedActive] = ImVec4(0.427f * 1.5f, 0.427f * 1.5f, 0.137f * 1.5f, 1.0f);
		style.Colors[ImGuiCol_C_Hires] = tcol(theme, 0.427f, 0.137f, 0.427f, 1.0f);
		style.Colors[ImGuiCol_C_HiresHovered] = tcol(theme, 0.427f * 1.3f, 0.137f * 1.3f, 0.427f * 1.3f, 1.0f);
		style.Colors[ImGuiCol_C_HiresActive] = tcol(theme, 0.427f * 1.5f, 0.137f * 1.5f, 0.427f * 1.5f, 1.0f);
		float axisDarkening = 0.7f;
		style.Colors[ImGuiCol_C_XAxis] = ImVec4(0.75f, 0.3f * axisDarkening, 0.3f * axisDarkening, 1.0f);
		style.Colors[ImGuiCol_C_YAxis] = ImVec4(0.33f * axisDarkening, 0.67f, 0.4f * axisDarkening, 1.0f);
		style.Colors[ImGuiCol_C_ZAxis] = ImVec4(0.3f * axisDarkening, 0.45f * axisDarkening, 0.7f, 1.0f);

		break;
	}
}