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
    addKernel(lorenz83);
    addKernel(lorenzVar);
    addKernel(hindmarsh_rose);
    addKernel(izhikevich);
    addKernel(jj_mcrls);
    addKernel(jj_rcls);
    addKernel(chen);
    addKernel(dadras);
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
    addKernel(kazantsev_gordleeva_matrosov);

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

    imgui_main(0, 0);
}

int main()
{
    common_main();

    return 0;
}

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PWSTR pCmdLine, int nCmdShow)
{
    common_main();

    return 0;
}