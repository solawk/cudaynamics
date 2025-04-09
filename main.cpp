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
    addKernel(halvorsen);
    addKernel(mrlcs_jj);

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