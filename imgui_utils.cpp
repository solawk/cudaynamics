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

void rotateOffsetBuffer(float* buffer, int pointCount, int varCount, int xdo, int ydo, int zdo, float pitch, float yaw, ImVec4 offset, ImVec4 scale)
{
	float x, y, z, zt;

	for (int i = 0; i < pointCount; i++)
	{
		x = ((float*)buffer)[i * varCount + xdo] * scale.x + offset.x;
		y = ((float*)buffer)[i * varCount + ydo] * scale.y + offset.y;
		z = ((float*)buffer)[i * varCount + zdo] * scale.z + offset.z;

		((float*)buffer)[i * varCount + 1] = y * cosf(-pitch * DEG2RAD) - z * sinf(-pitch * DEG2RAD);
		((float*)buffer)[i * varCount + 2] = zt = y * sinf(-pitch * DEG2RAD) + z * cosf(-pitch * DEG2RAD);

		((float*)buffer)[i * varCount + 0] = x * cosf(yaw * DEG2RAD) + zt * sinf(yaw * DEG2RAD);
	}
}

void populateEggslicerBuffer(float* buffer) // One axis of a grid
{
	// 11 ticks per axis, 4+1=5 vertices per tick, 3 coordinates per vertex
	for (int i = 0; i < 11*5*3; i++) buffer[i] = 0;

	for (int x = -5; x <= 5; x++)
	{
		float xf = x * 1.0f;
		int i = x + 5;
		
		buffer[i * 5 * 3 + 0] = xf;
		buffer[i * 5 * 3 + 1] = -5.0f;
		buffer[i * 5 * 3 + 2] = -5.0f;

		buffer[i * 5 * 3 + 3] = xf;
		buffer[i * 5 * 3 + 4] = 5.0f;
		buffer[i * 5 * 3 + 5] = -5.0f;

		buffer[i * 5 * 3 + 6] = xf;
		buffer[i * 5 * 3 + 7] = 5.0f;
		buffer[i * 5 * 3 + 8] = 5.0f;

		buffer[i * 5 * 3 + 9] = xf;
		buffer[i * 5 * 3 + 10] = -5.0f;
		buffer[i * 5 * 3 + 11] = 5.0f;

		buffer[i * 5 * 3 + 12] = xf;
		buffer[i * 5 * 3 + 13] = -5.0f;
		buffer[i * 5 * 3 + 14] = -5.0f;
	}
}

void eggslicerX2Y(float* buffer)
{
	for (int i = 0; i < 11*5; i++)
	{
		float z = buffer[i * 3 + 2];
		buffer[i * 3 + 2] = buffer[i * 3 + 0];
		buffer[i * 3 + 0] = z;
	}
}

void eggslicerY2Z(float* buffer)
{
	for (int i = 0; i < 11 * 5; i++)
	{
		float z = buffer[i * 3 + 2];
		buffer[i * 3 + 2] = buffer[i * 3 + 1];
		buffer[i * 3 + 1] = z;
	}
}