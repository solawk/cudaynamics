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
    addKernel(jj_mrlcs);
    addKernel(jj_rlcs);
    addKernel(chen);
    addKernel(dadras);
    addKernel(fourwing);
    addKernel(halvorsen);
    addKernel(langford);
    addKernel(rossler);
    addKernel(sprott14);
    addKernel(sprottJm);
    addKernel(three_scroll);
    addKernel(wilson);
    addKernel(rabinovich_fabrikant);
    addKernel(sang);
    addKernel(thomas);
    addKernel(vnm);
    addKernel(fitzhugh_nagumo);
    addKernel(bolshakov);
    addKernel(mishchenko);
    addKernel(mixed);
    addKernel(ostrovskii);

    //selectKernel(kernels.begin()->first);
    selectKernel(jj_mrlcs);

    // Indices
    addIndex(IND_MAX, "Maximum variable value", MINMAX, 1);
    addIndex(IND_MIN, "Minimum variable value", MINMAX, 1);
    addIndex(IND_LLE, "Largest Lyapunov exponent", LLE, 1);
    addIndex(IND_PERIOD, "Period", PERIOD, 1);
    addIndex(IND_MNPEAK, "Mean peak", PERIOD, 1);
    addIndex(IND_MNINT, "Mean interval", PERIOD, 1);

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