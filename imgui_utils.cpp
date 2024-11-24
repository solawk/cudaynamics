#include "imgui_main.h"
#include "implot/implot.h"
#include <tchar.h>
#include <stdio.h>
#include <string>
#include <objects.h>
#include <vector>
#include <implot_internal.h>
#include "imgui_utils.h"

const float DEG2RAD = 3.141592f / 180.0f;

void populateAxisBuffer(float* buffer, float x, float y, float z)
{
	for (int i = 0; i < 18; i++) buffer[i] = 0;

	buffer[0] = x;
	buffer[3] = -x * 0.5f;

	buffer[7] = y;
	buffer[10] = -y * 0.5f;

	buffer[14] = z;
	buffer[17] = -z * 0.5f;
}

void rotateOffsetBuffer(float* buffer, int pointCount, float pitch, float yaw, float xo, float yo, float zo)
{
	float x, y, z, zt;

	for (int i = 0; i < pointCount; i++)
	{
		x = ((float*)buffer)[i * 3 + 0] + xo;
		y = ((float*)buffer)[i * 3 + 1] + yo;
		z = ((float*)buffer)[i * 3 + 2] + zo;

		((float*)buffer)[i * 3 + 1] = y * cosf(-pitch * DEG2RAD) - z * sinf(-pitch * DEG2RAD);
		((float*)buffer)[i * 3 + 2] = zt = y * sinf(-pitch * DEG2RAD) + z * cosf(-pitch * DEG2RAD);

		((float*)buffer)[i * 3 + 0] = x * cosf(yaw * DEG2RAD) + zt * sinf(yaw * DEG2RAD);
	}
}