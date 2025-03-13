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

int main()
{
    addKernel(lorenz2);
    addKernel(halvorsen);

    //selectedKernel = kernels.begin()->first;
    selectedKernel = "lorenz2";

    imgui_main(0, 0);

    return 0;
}

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PWSTR pCmdLine, int nCmdShow)
{
    imgui_main(0, 0);

    return 0;
}