#include "imgui_utils.h"

std::string memoryString(unsigned long long bytes)
{
	if (bytes < 1024)
	{
		// B
		return std::to_string(bytes) + " B";
	}
	else if (bytes < 1024 * 1024)
	{
		// kB
		return std::to_string((int)(bytes / 1024.0)) + " kB";
	}
	else if (bytes < 1024 * 1024 * 1024)
	{
		// MB
		return std::to_string((int)(bytes / (1024.0 * 1024))) + " MB";
	}
	else
	{
		// GB
		return std::to_string((int)(bytes / (1024.0 * 1024 * 1024))) + " GB";
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

void rotateOffsetBuffer(float* buffer, int pointCount, int varCount, int xdo, int ydo, int zdo, ImVec4 rotation, ImVec4 offset, ImVec4 scale)
{
	float x, y, z;

	float alpha = rotation.x; // yaw
	float beta = rotation.y; // pitch
	float gamma = rotation.z; // roll

	float ac = cosf(alpha);
	float bc = cosf(beta);
	float gc = cosf(gamma);

	float as = sinf(alpha);
	float bs = sinf(beta);
	float gs = sinf(gamma);

	for (int i = 0; i < pointCount; i++)
	{
		x = -(buffer[i * varCount + xdo] * scale.x + offset.x);
		y = buffer[i * varCount + ydo] * scale.y + offset.y;
		z = buffer[i * varCount + zdo] * scale.z + offset.z;

		buffer[i * varCount + 0] = (x * bc * gc) + (y * (as * bs * gc - (ac * gs))) + (z * (ac * bs * gc + as * gs));
		buffer[i * varCount + 1] = (x * bc * gs) + (y * (as * bs * gs + (ac * gc))) + (z * (ac * bs * gs - as * gc));
		buffer[i * varCount + 2] = (x * -bs) + (y * (as * bc)) + (z * (ac * bc));
	}
}

void rotateOffsetBuffer(double* buffer, int pointCount, int varCount, int xdo, int ydo, int zdo, ImVec4 rotation, ImVec4 offset, ImVec4 scale)
{
	double x, y, z;

	float alpha = rotation.x; // yaw
	float beta = rotation.y; // pitch
	float gamma = rotation.z; // roll

	float ac = cosf(alpha);
	float bc = cosf(beta);
	float gc = cosf(gamma);

	float as = sinf(alpha);
	float bs = sinf(beta);
	float gs = sinf(gamma);

	for (int i = 0; i < pointCount; i++)
	{
		x = -(buffer[i * varCount + xdo] * scale.x + offset.x);
		y = buffer[i * varCount + ydo] * scale.y + offset.y;
		z = buffer[i * varCount + zdo] * scale.z + offset.z;

		buffer[i * varCount + 0] = (x * bc * gc) + (y * (as * bs * gc - (ac * gs))) + (z * (ac * bs * gc + as * gs));
		buffer[i * varCount + 1] = (x * bc * gs) + (y * (as * bs * gs + (ac * gc))) + (z * (ac * bs * gs - as * gc));
		buffer[i * varCount + 2] = (x * -bs) + (y * (as * bc)) + (z * (ac * bc));
	}
}

void rotateOffsetBufferQuat(float* buffer, int pointCount, int varCount, int xdo, int ydo, int zdo, ImVec4 rotation, ImVec4 offset, ImVec4 scale)
{
	float x, y, z;

	float r = rotation.w;
	float i = rotation.x;
	float j = rotation.y;
	float k = rotation.z;

	float r2 = r * r;
	float i2 = i * i;
	float j2 = j * j;
	float k2 = k * k;

	for (int i = 0; i < pointCount; i++)
	{
		x = buffer[i * varCount + xdo] * scale.x + offset.x;
		y = buffer[i * varCount + ydo] * scale.y + offset.y;
		z = buffer[i * varCount + zdo] * scale.z + offset.z;

		buffer[i * varCount + 0] = (x * (1.0f - 2.0f * j2 - 2.0f * k2))		+ (y * (2.0f * (i * j - k * r)))		+ (z * (2.0f * (i * k + j * r)));
		buffer[i * varCount + 1] = (x * (2.0f * (i * j + k * r)))			+ (y * (1.0f - 2.0f * i2 - 2.0f * k2))	+ (z * (2.0f * (j * k - i * r)));
		buffer[i * varCount + 2] = (x * (2.0f * (i * k - j * r)))			+ (y * (2.0f * (j * k + i * r)))		+ (z * (1.0f - 2.0f * i2 - 2.0f * j2));
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

// Cut everything that is outside the cutoff rectangle (minX, minY, maxX, maxY)
// The rectangle is inclusive
void cutoff2D(numb* data, numb* dst, int width, int height, int minX, int minY, int maxX, int maxY)
{
	if (minX == 0 && minY == 0 && maxX == (width - 1) && maxY == (height - 1))
	{
		memcpy(dst, data, width * height * sizeof(numb));
		return;
	}

	int newWidthInclusive = maxX - minX + 1;
	int newHeightInclusive = maxY - minY + 1;

	for (int y = 0; y < newHeightInclusive; y++)
		for (int x = 0; x < newWidthInclusive; x++)
			dst[y * newWidthInclusive + x] = data[(minY + y) * width + minX + x];
}

void getMinMax(numb* data, int size, numb* min, numb* max)
{
	*min = data[0];
	*max = data[0];

	for (int i = 1; i < size; i++)
	{
		if (isnan(data[i]) || isinf(data[i])) continue;

		if (data[i] < *min) *min = data[i];
		if (data[i] > *max) *max = data[i];
	}

	if (isnan(*min)) *min = 0.0f;
	if (isnan(*max)) *max = 1.0f;
}

void getMinMax2D(numb* data, int size, ImVec2* min, ImVec2* max, int varCount)
{
	(*min).x = (float)data[0];
	(*min).y = (float)data[1];
	(*max).x = (float)data[0];
	(*max).y = (float)data[1];

	int x, y;

	for (int i = 1; i < size; i++)
	{
		x = i * varCount;
		y = i * varCount + 1;

		if (isnan(data[x]) || isinf(data[x]) || isnan(data[y]) || isinf(data[y])) continue;

		if ((float)data[x] < (*min).x) (*min).x = (float)data[x];
		if ((float)data[y] < (*min).y) (*min).y = (float)data[y];
		if ((float)data[x] > (*max).x) (*max).x = (float)data[x];
		if ((float)data[y] > (*max).y) (*max).y = (float)data[y];
	}

	if (isnan((*min).x)) (*min).x = 0.0f;
	if (isnan((*min).y)) (*min).y = 0.0f;
	if (isnan((*max).x)) (*max).x = 1.0f;
	if (isnan((*max).y)) (*max).y = 1.0f;
}

std::string padString(std::string str, int length)
{
	std::string strPadded = str;
	for (int j = (int)str.length(); j < length; j++)
		strPadded += ' ';
	return strPadded;
}

void addDeltaQuatRotation(PlotWindow* window, float deltax, float deltay)
{
	quaternion::Quaternion<float> quat(window->quatRot.w, window->quatRot.x, window->quatRot.y, window->quatRot.z);
	quaternion::Quaternion<float> quatY(cosf(deltax * 0.5f * DEG2RAD), 0.0f, sinf(deltax * 0.5f * DEG2RAD), 0.0f);
	quaternion::Quaternion<float> quatX(cosf(deltay * 0.5f * DEG2RAD), sinf(deltay * 0.5f * DEG2RAD), 0.0f, 0.0f);
	quat = quatY * quatX * quat;
	quat = quaternion::normalize(quat);
	window->quatRot.w = quat.a();
	window->quatRot.x = quat.b();
	window->quatRot.y = quat.c();
	window->quatRot.z = quat.d();
}

int getVariationGroup(colorLUT* lut, int variation)
{
	int variationGroup = -1;
	int lutsize;
	for (int g = 0; g < lut->lutGroups && variationGroup < 0; g++)
	{
		lutsize = lut->lutSizes[g];
		for (int v = 0; v < lutsize && variationGroup < 0; v++)
			if (variation == lut->lut[g][v]) variationGroup = g;
	}

	return variationGroup;
}