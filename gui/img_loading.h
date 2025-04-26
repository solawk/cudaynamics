#pragma once
#include "../imgui_main.hpp"

bool LoadTextureFromRaw(unsigned char** data, int width, int height, ID3D11ShaderResourceView** out_srv, ID3D11Device* g_pd3dDevice);

//bool LoadTextureFromFile(const char* file_name, ID3D11ShaderResourceView** out_srv, int* out_width, int* out_height, ID3D11Device* g_pd3dDevice);