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

std::string scaleString(float scale)
{
	if (scale > 0.5f && scale < 5.0f) return "1.0";

	if (scale > 0.05f && scale < 0.5f) return "0.1";
	if (scale > 0.005f && scale < 0.05f) return "0.01";
	if (scale > 0.0005f && scale < 0.005f) return "0.001";

	if (scale > 5.0f && scale < 50.0f) return "10";
	if (scale > 50.0f && scale < 500.0f) return "100";
	if (scale > 500.0f && scale < 5000.0f) return "1000";

	if (scale < 1.0f)
	{
		for (int i = 4; i < 100; i++)
		{
			if (scale > powf(0.1f, (float)i) * 0.5f && scale < powf(0.1f, i - 1.0f) * 0.5f) return "10e-" + std::to_string(i);
		}

		return "<10e-100";
	}
	else
	{
		for (int i = 4; i < 100; i++)
		{
			if (scale > powf(10.0f, (float)i) * 0.5f && scale < powf(10.0f, i + 1.0f) * 0.5f) return "10e" + std::to_string(i);
		}

		return ">10e100";
	}
}

void populateAxisBuffer(float* buffer, float x, float y, float z)
{
	for (int i = 0; i < 18; i++) buffer[i] = 0.0f;

	buffer[0] = x;
	buffer[3] = -x * 0.5f;

	buffer[7] = y;
	buffer[10] = -y * 0.5f;

	buffer[14] = z;
	buffer[17] = -z * 0.5f;
}

void rotateOffsetBuffer2(float* buffer, int pointCount, int varCount, int xdo, int ydo, int zdo, float pitch, float yaw, ImVec4 offset, ImVec4 scale)
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

void rotateOffsetBuffer(float* buffer, int pointCount, int varCount, int xdo, int ydo, int zdo, ImVec4 rotation, ImVec4 offset, ImVec4 scale)
{
	float x, y, z;

	for (int i = 0; i < pointCount; i++)
	{
		x = ((float*)buffer)[i * varCount + xdo] * scale.x + offset.x;
		y = ((float*)buffer)[i * varCount + ydo] * scale.y + offset.y;
		z = ((float*)buffer)[i * varCount + zdo] * scale.z + offset.z;

		float alpha = rotation.x; // yaw
		float beta = rotation.y; // pitch
		float gamma = rotation.z; // roll

		((float*)buffer)[i * varCount + 0] = (x * cosf(beta) * cosf(gamma)) + (y * (sin(alpha) * sin(beta) * cos(gamma) - (cos(alpha) * sin(gamma)))) + (z * (cos(alpha) * sin(beta) * cos(gamma) + sin(alpha) * sin(gamma)));
		((float*)buffer)[i * varCount + 1] = (x * cosf(beta) * sin(gamma)) + (y * (sin(alpha) * sin(beta) * sin(gamma) + (cos(alpha) * cos(gamma)))) + (z * (cos(alpha) * sin(beta) * sin(gamma) - sin(alpha) * cos(gamma)));
		((float*)buffer)[i * varCount + 2] = (x * -sinf(beta)) + (y * (sin(alpha) * cos(beta))) + (z * (cos(alpha) * cos(beta)));
	}
}

void populateRulerBuffer(float* buffer, float s, int dim)
{
	for (int i = 0; i < (51*3); i++) buffer[i] = 0.0f;

	switch (dim)
	{
	case 0:
		for (int i = 0; i < 10; i++)
		{
			buffer[3 + (i * 15) + 0] = (i + 1.0f) * 1.0f;

			buffer[3 + (i * 15) + 3] = (i + 1.0f) * 1.0f;
			buffer[3 + (i * 15) + 4] = 0.5f;

			buffer[3 + (i * 15) + 6] = (i + 1.0f) * 1.0f;

			buffer[3 + (i * 15) + 9] = (i + 1.0f) * 1.0f;
			buffer[3 + (i * 15) + 11] = 0.5f;

			buffer[3 + (i * 15) + 12] = (i + 1.0f) * 1.0f;
		}
		break;

	case 1:
		for (int i = 0; i < 10; i++)
		{
			buffer[3 + (i * 15) + 1] = (i + 1.0f) * 1.0f;

			buffer[3 + (i * 15) + 4] = (i + 1.0f) * 1.0f;
			buffer[3 + (i * 15) + 5] = 0.5f;

			buffer[3 + (i * 15) + 7] = (i + 1.0f) * 1.0f;

			buffer[3 + (i * 15) + 10] = (i + 1.0f) * 1.0f;
			buffer[3 + (i * 15) + 9] = 0.5f;

			buffer[3 + (i * 15) + 13] = (i + 1.0f) * 1.0f;
		}
		break;

	case 2:
		for (int i = 0; i < 10; i++)
		{
			buffer[3 + (i * 15) + 2] = (i + 1.0f) * 1.0f;

			buffer[3 + (i * 15) + 5] = (i + 1.0f) * 1.0f;
			buffer[3 + (i * 15) + 3] = 0.5f;

			buffer[3 + (i * 15) + 8] = (i + 1.0f) * 1.0f;

			buffer[3 + (i * 15) + 11] = (i + 1.0f) * 1.0f;
			buffer[3 + (i * 15) + 10] = 0.5f;

			buffer[3 + (i * 15) + 14] = (i + 1.0f) * 1.0f;
		}
		break;
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

ImVec4 ToEulerAngles(ImVec4 q)
{
	ImVec4 angles;

	// roll (x-axis rotation)
	float sinr_cosp = 2.0f * (q.w * q.x + q.y * q.z);
	float cosr_cosp = 1.0f - 2.0f * (q.x * q.x + q.y * q.y);
	angles.x = std::atan2f(sinr_cosp, cosr_cosp);

	// pitch (y-axis rotation)
	float sinp = std::sqrt(1 + 2 * (q.w * q.y - q.x * q.z));
	float cosp = std::sqrt(1 - 2 * (q.w * q.y - q.x * q.z));
	angles.y = 2.0f * std::atan2f(sinp, cosp) - 3.141592f / 2.0f;

	// yaw (z-axis rotation)
	float siny_cosp = 2.0f * (q.w * q.z + q.x * q.y);
	float cosy_cosp = 1.0f - 2.0f * (q.y * q.y + q.z * q.z);
	angles.z = std::atan2f(siny_cosp, cosy_cosp);

	return angles;
}