#include "mainWindowMenu.h"

void mainWindowMenu()
{
	if (ImGui::BeginMenuBar())
	{
        if (ImGui::BeginMenu("File"))
        {
            if (ImGui::MenuItem("Save computed system configuration")) 
            {
                JSONWrite(saveCfg(hiresIndex != IND_NONE, false), "testCfg.cfg", true);
            }
            TOOLTIP("Save as it has been last computed")

            if (ImGui::MenuItem("Save edited system configuration"))
            {
                JSONWrite(saveCfg(hiresIndex != IND_NONE, true), "testCfg.cfg", true);
            }
            TOOLTIP("Save with edited fields preserved")

            if (ImGui::MenuItem("Load system configuration"))
            {
                json::jobject cfg;
                if (JSONRead("testCfg.cfg", &cfg, true))
                {
                    bool systemChanged = (std::string)cfg["system"] != selectedKernel;
                    if (systemChanged) switchToSystem((std::string)cfg["system"]);
                    loadCfg(cfg, false, false, false);
                    //initializeKernel(false);
                }
            }

            bool allowHiresExport = computationHires.ready && computationHires.marshal.maps != nullptr && hiresIndex != IND_NONE;
            if (allowHiresExport)
            {
                if (ImGui::MenuItem("Export Hi-Res to .csv"))
                {
                    std::string systemName = kernels[selectedKernel].name;

                    Index* index = &(indices[hiresIndex]);
                    std::string indexName = index->name;

                    char ts[32];
                    std::time_t t = std::time(nullptr);
                    std::tm tm{};
                    localtime_s(&tm, &t);
                    std::snprintf(ts, sizeof(ts), "%04d-%02d-%02d_%02d-%02d-%02d",
                        tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                        tm.tm_hour, tm.tm_min, tm.tm_sec);
                    std::string timestamp = (std::string)ts;

                    exportPath = systemName + "_" + indexName + "_" + timestamp;

                    if (exportHires())
                    {
                        MessageBoxA(NULL, ("Exported succesfully to " + exportPath).c_str(), "Export", MB_OK | MB_ICONINFORMATION);
                    }
                    else
                    {
                        MessageBoxA(NULL, "Export failed (empty data or I/O error)", "Export", MB_OK | MB_ICONERROR);
                    }
                }
                TOOLTIP("Save with edited fields preserved")
            }

            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("View"))
        {
            ImGui::SeparatorText("Appearance");
            bool isDark = applicationSettings.appStyle == ImGuiCustomStyle::Dark;
            if (ImGui::Checkbox("Dark theme", &isDark))
            {
                if (isDark) applicationSettings.appStyle = ImGuiCustomStyle::Dark;
                else applicationSettings.appStyle = ImGuiCustomStyle::Light;
                //SetupImGuiStyle(appStyle);
            }

            ImGui::ColorEdit4("##cudaColorPicker", (float*)(&applicationSettings.cudaColor)); ImGui::SameLine(); ImGui::Text("CUDA mode color");
            ImGui::ColorEdit4("##openmpColorPicker", (float*)(&applicationSettings.openmpColor)); ImGui::SameLine(); ImGui::Text("OpenMP mode color");
            ImGui::ColorEdit4("##hiresColorPicker", (float*)(&applicationSettings.hiresColor)); ImGui::SameLine(); ImGui::Text("Hi-Res mode color");

            ImGui::SeparatorText("Font");
            FontMenu(nullptr);

            ImGui::SeparatorText("Attributes");

            ImGui::SetNextItemWidth(150.0f);
            ImGui::InputFloat("Value drag speed", &(applicationSettings.dragChangeSpeed));
            TOOLTIP("Drag speed of attribute values, allows for precise automatic parameter setting");
            if (ImGui::IsItemDeactivatedAfterEdit()) { applicationSettings.Save(); };

            bool tempPreciseNumbDrags = applicationSettings.preciseNumbDrags;
            if (ImGui::Checkbox("Precise numb values", &tempPreciseNumbDrags))
            {
                applicationSettings.preciseNumbDrags = tempPreciseNumbDrags;
                applicationSettings.Save();
            }
            TOOLTIP("Enable 12-digit fraction for attribute values");

            if (ImGui::Checkbox("Sliders for ranging", &(applicationSettings.sliderRanging)))
            {
                applicationSettings.Save();
            }
            TOOLTIP("Use sliders instead of input fields for attribute ranging");

            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Settings"))
        {
            ImGui::SeparatorText("CUDA/OpenMP");

            bool tempCPUinter_mode = applicationSettings.CPU_mode_interactive;
            ImGui::Checkbox("Use OpenMP in interactive mode", &tempCPUinter_mode);
            TOOLTIP("Use CPU with OpenMP instead of GPU with CUDA in non-Hi-Res mode");
            applicationSettings.CPU_mode_interactive = tempCPUinter_mode;

            bool tempCPUhires_mode = applicationSettings.CPU_mode_hires;
            ImGui::Checkbox("Use OpenMP in Hi-Res mode", &tempCPUhires_mode);
            TOOLTIP("Use CPU with OpenMP instead of GPU with CUDA in Hi-Res mode");
            applicationSettings.CPU_mode_hires = tempCPUhires_mode;

            ImGui::InputInt("CUDA Threads per block", &applicationSettings.threadsPerBlock, 1, 10, 0);
            if (applicationSettings.threadsPerBlock < 1) applicationSettings.threadsPerBlock = 1;
            if (ImGui::IsItemDeactivatedAfterEdit()) { applicationSettings.Save(); };

            ImGui::SeparatorText("Simulation");
            ImGui::Checkbox("Compute on open", &(applicationSettings.computeOnLaunch));
            TOOLTIP("Compute when opening a new system");

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

            bool tempCalcDeltaDecay = applicationSettings.calculateDeltaDecay;
            ImGui::Checkbox("Calculate delta and decay", &tempCalcDeltaDecay);
            TOOLTIP("Calculate index delta and decay maps");
            applicationSettings.calculateDeltaDecay = tempCalcDeltaDecay;

            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("About"))
        {
            ImGui::Text("CUDAynamics Interactive Suite");
            ImGui::NewLine();

            ImGui::SeparatorText("Developed by");
            ImGui::Text("Alexander Khanov, Ivan Guitor,");
            ImGui::Text("Nikita Belyaev, Ksenia Shinkar,");
            ImGui::Text("Maksim Gozhan, Anastasia Karpenko");
            ImGui::NewLine();

            ImGui::SeparatorText("Directed by");
            ImGui::Text("Valerii Ostrovskii");
            ImGui::NewLine();

            ImGui::SeparatorText("Dependencies");
            ImGui::Text("Dear ImGui by Omar Cornut");
            ImGui::Text("ImPlot by Evan Pezent");
            ImGui::Text("CUDA Toolkit 12.6.2 by NVIDIA");
            ImGui::Text("OpenMP 2.0 standard");

            ImGui::EndMenu();
        }

        ImGui::EndMenuBar();
    }
}