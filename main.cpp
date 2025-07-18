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
std::map<std::string, void(*)(numb*, numb*, numb*, numb)> kernelFDSs;
std::string selectedKernel;

void common_main()
{
    addKernel(lorenz2);
    addKernel(lorenzMod);
    addKernel(mrlcs_jj);
    addKernel(rlcs_jj);
    addKernel(chen);
    addKernel(dadras);
    addKernel(fourwing);
    addKernel(halvorsen);
    addKernel(langford);
    addKernel(rossler);
    addKernel(sprott);
    addKernel(three_scroll);
    addKernel(wilson);
    addKernel(msprottj);
    addKernel(thomas);

    //selectKernel(kernels.begin()->first);
    selectKernel(lorenz2);

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