#pragma once
#include "imgui/backends/imgui_impl_win32.h"
#include "imgui/backends/imgui_impl_dx11.h"
#include "../quaternion.h"

const float DEG2RAD = 3.141592f / 180.0f;

// Translate-Rotation-Scale (for phase plots)
struct TRS
{
	ImVec4 offset;
	ImVec4 scale;
	ImVec4 quatRot;
	ImVec4 autorotate; // euler angles
	ImVec2 deltarotation; // euler angles

	TRS()
	{
		offset = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
		scale = ImVec4(1.0f, 1.0f, 1.0f, 0.0f);

		quatRot = ImVec4(1.0f, 0.0f, 0.0f, 0.0f);
		autorotate = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
		deltarotation = ImVec2(0.0f, 0.0f);
	}

	void KeepScaleAboveZero()
	{
		if (scale.x < 0.0f) scale.x = 0.0f; 
		if (scale.y < 0.0f) scale.y = 0.0f; 
		if (scale.z < 0.0f) scale.z = 0.0f;
	}

	void RotateQuaternionByEulerDrag(ImVec4 deltaEuler)
	{
		quaternion::Quaternion<float> quatEditable(1.0f, 0.0f, 0.0f, 0.0f);
		quaternion::Quaternion<float> quatRotQ(quatRot.w, quatRot.x, quatRot.y, quatRot.z);
		quaternion::Quaternion<float> quatZ(cosf(deltaEuler.z * 0.5f * DEG2RAD), 0.0f, 0.0f, sinf(deltaEuler.z * 0.5f * DEG2RAD));
		quaternion::Quaternion<float> quatY(cosf(deltaEuler.y * 0.5f * DEG2RAD), 0.0f, sinf(deltaEuler.y * 0.5f * DEG2RAD), 0.0f);
		quaternion::Quaternion<float> quatX(cosf(deltaEuler.x * 0.5f * DEG2RAD), sinf(deltaEuler.x * 0.5f * DEG2RAD), 0.0f, 0.0f);

		if (deltaEuler.x != 0.0f) quatEditable = quatX * quatEditable;
		if (deltaEuler.y != 0.0f) quatEditable = quatY * quatEditable;
		if (deltaEuler.z != 0.0f) quatEditable = quatZ * quatEditable;

		quatEditable = quatRotQ * quatEditable;
		quatEditable = quaternion::normalize(quatEditable);

		quatRot = ImVec4(quatEditable.b(), quatEditable.c(), quatEditable.d(), quatEditable.a());
	}
};