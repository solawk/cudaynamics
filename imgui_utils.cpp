#include "imgui_utils.h"

std::string memoryString(unsigned long int bytes)
{
	if (bytes < 1024)
	{
		// B
		return to_string(bytes) + " B";
	}
	else if (bytes < 1024 * 1024)
	{
		// kB
		return to_string((int)(bytes / 1024)) + " kB";
	}
	else if (bytes < 1024 * 1024 * 1024)
	{
		// MB
		return to_string((int)(bytes / (1024 * 1024))) + " MB";
	}
	else
	{
		// GB
		return to_string((int)(bytes / (1024 * 1024 * 1024))) + " GB";
	}
}

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

void populateGridBuffer(float* buffer) // One axis of a grid
{
	// 10 ticks per axis, 4+1=5 vertices per tick, 3 coordinates per vertex, 2 directions
	//for (int i = 0; i < 10*5*3*2; i++) buffer[i] = 0;

	for (int x = -5; x < 5; x++)
	{
		float xf = x * 1.0f;
		int i = x + 5;
		
		buffer[i * 5 * 3 + 0] = 0.0f;
		buffer[i * 5 * 3 + 1] = xf;
		buffer[i * 5 * 3 + 2] = -5.0f;

		buffer[i * 5 * 3 + 3] = 0.0f;
		buffer[i * 5 * 3 + 4] = xf + 1.0f;
		buffer[i * 5 * 3 + 5] = -5.0f;

		buffer[i * 5 * 3 + 6] = 0.0f;
		buffer[i * 5 * 3 + 7] = xf + 1.0f;
		buffer[i * 5 * 3 + 8] = 5.0f;

		buffer[i * 5 * 3 + 9] = 0.0f;
		buffer[i * 5 * 3 + 10] = xf;
		buffer[i * 5 * 3 + 11] = 5.0f;

		buffer[i * 5 * 3 + 12] = 0.0f;
		buffer[i * 5 * 3 + 13] = xf;
		buffer[i * 5 * 3 + 14] = -5.0f;
	}

	for (int x = -5; x < 5; x++)
	{
		float xf = x * 1.0f;
		int i = x + 5;

		buffer[150 + i * 5 * 3 + 0] = 0.0f;
		buffer[150 + i * 5 * 3 + 2] = xf;
		buffer[150 + i * 5 * 3 + 1] = -5.0f;

		buffer[150 + i * 5 * 3 + 3] = 0.0f;
		buffer[150 + i * 5 * 3 + 5] = xf + 1.0f;
		buffer[150 + i * 5 * 3 + 4] = -5.0f;

		buffer[150 + i * 5 * 3 + 6] = 0.0f;
		buffer[150 + i * 5 * 3 + 8] = xf + 1.0f;
		buffer[150 + i * 5 * 3 + 7] = 5.0f;

		buffer[150 + i * 5 * 3 + 9] = 0.0f;
		buffer[150 + i * 5 * 3 + 11] = xf;
		buffer[150 + i * 5 * 3 + 10] = 5.0f;

		buffer[150 + i * 5 * 3 + 12] = 0.0f;
		buffer[150 + i * 5 * 3 + 14] = xf;
		buffer[150 + i * 5 * 3 + 13] = -5.0f;
	}
}

void gridX2Y(float* buffer)
{
	for (int i = 0; i < 10 * 5 * 2; i++)
	{
		float z = buffer[i * 3 + 2];
		buffer[i * 3 + 2] = buffer[i * 3 + 0];
		buffer[i * 3 + 0] = z;
	}
}

void gridY2Z(float* buffer)
{
	for (int i = 0; i < 10 * 5 * 2; i++)
	{
		float z = buffer[i * 3 + 2];
		buffer[i * 3 + 2] = buffer[i * 3 + 1];
		buffer[i * 3 + 1] = z;
	}
}