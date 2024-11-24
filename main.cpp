#include "main.h"

#define IS_CONSOLE 1

#if IS_CONSOLE
#pragma comment( linker, "/subsystem:console" )
#else
#pragma comment( linker, "/subsystem:windows" )
#endif

int main()
{
    imgui_main(0, 0);

    return 0;
}

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PWSTR pCmdLine, int nCmdShow)
{
    imgui_main(0, 0);

    return 0;
}