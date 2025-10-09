#include "main.h"

#define IS_CONSOLE 1

#if IS_CONSOLE
#pragma comment( linker, "/subsystem:console" )
#else
#pragma comment( linker, "/subsystem:windows" )
#endif

std::map<std::string, Kernel> kernels;
std::map<std::string, int> kernelTPBs;
std::map<std::string, void(*)(Computation*)> kernelPrograms;
//std::map<std::string, void(*)(numb*, numb*, numb*)> kernelFDSs;
std::string selectedKernel;

void common_main()
{
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

    //selectKernel(kernels.begin()->first);
    selectKernel(lorenz);

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