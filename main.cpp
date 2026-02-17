#include "main.h"

#define IS_CONSOLE 1

#if IS_CONSOLE
#pragma comment( linker, "/subsystem:console" )
#else
#pragma comment( linker, "/subsystem:windows" )
#endif

void common_main()
{
    // Kernels
    addKernel(lorenz);
    addKernel(lorenz84);
    addKernel(lorenzVar);
    addKernel(hindmarsh_rose);
    addKernel(izhikevich);
    addKernel(jj_mcrls);
    addKernel(jj_rcls);
    addKernel(chen_lee);
    addKernel(dadras_momeni);
    addKernel(fourwing);
    addKernel(halvorsen);
    addKernel(langford);
    addKernel(rossler);
    addKernel(sprott14);
    addKernel(three_scroll);
    addKernel(wilson);
    addKernel(rabinovich_fabrikant);
    addKernel(sang25);
    addKernel(thomas);
    addKernel(fitzhugh_nagumo);
    addKernel(bolshakov);
    addKernel(mishchenko);
    addKernel(mixed);
    addKernel(ostrovskii);
    addKernel(hodgkin_huxley);
    addKernel(sang26);
    addKernel(pala_machaczek);
    addKernel(ullah);

    //selectKernel(kernels.begin()->first);
    selectKernel(lorenz);

    // Indices
    addIndex(IND_MAX, "Maximum variable value", MINMAX, 1);
    addIndex(IND_MIN, "Minimum variable value", MINMAX, 1);
    addIndex(IND_LLE, "Largest Lyapunov exponent", LLE, 1);
    addIndex(IND_PERIOD, "Period", PERIOD, 1);
    addIndex(IND_MNMPEAK, "Minimum peak", PERIOD, 1);
    addIndex(IND_MNMINT, "Minimum interval", PERIOD, 1);
    addIndex(IND_MNPEAK, "Mean peak", PERIOD, 1);
    addIndex(IND_MNINT, "Mean interval", PERIOD, 1);
    addIndex(IND_MXMPEAK, "Maximum peak", PERIOD, 1);
    addIndex(IND_MXMINT, "Maximum interval", PERIOD, 1);
    addIndex(IND_PV, "Phase volume", PV, 1);

    // Initialization
    computationsInit();

    // "One shot" is launching the suite with a config file to perform a single hi-res computation, export it and return
    if (!launchedAsOneShot)
    {
        imgui_main(0, 0);
    }
    else
    {
        kernelNew.CopyFrom(&KERNEL);
        if (!loadCfg(launchConfig, true)) return;
        KERNEL.CopyFrom(&kernelNew);
        if (hiresIndex == IND_NONE)
        {
            printf("FAIL: No Hi-res index is selected\n");
            return;
        }

        hiresComputationSetup();
        KERNEL.PrepareAttributes();
        computationHires.marshal.kernel.CopyFrom(&KERNEL);
        printf("Starting!\n");
        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        int computationResult = compute(&computationHires);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        computationHires.ready = true;
        if (computationResult > 0)
        {
            printf("FAIL: Computation has failed\n");
            return;
        }
        printf("Computed successfully in %f s\n", (float)(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) / 1000.0f);

        exportHires();
        printf("SUCCESS: Exported succesfully\n");
    }
}

int main(int argc, char** argv)
{
    if (!readLaunchOptions(argc, argv))
    {
        printf("FAIL: Couldn't read launch options, aborting...\n");
        return -1;
    }

    common_main();

    return 0;
}

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PWSTR pCmdLine, int nCmdShow)
{
    common_main();

    return 0;
}