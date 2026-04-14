#version 310 es

//============================================================================================================
//
//
//                  Copyright (c) 2025, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================

precision mediump float;
precision highp int;

////////////////////////
// USER CONFIGURATION //
////////////////////////

#define EdgeThreshold (8.0 / 255.0)
#define EdgeSharpness 2.0

////////////////////////
////////////////////////
////////////////////////

uniform highp vec4 ViewportInfo;
uniform mediump sampler2D ps0;

in highp vec4 in_TEXCOORD0;
out vec4 out_Target0;

float fastLanczos2(float x)
{
	float wA = x - 4.0;
	float wB = x * wA - wA;
	return wB * (wA * wA);
}

vec2 weightY(vec2 d, float c, vec3 data)
{
	float std = data.x;
	vec2 dir = data.yz;

	float edgeDis = dot(d, dir.yx);
	float x = dot(d, d) + (edgeDis * edgeDis) * (clamp((c * c) * std, 0.0, 1.0) * 0.7 - 1.0);

	return vec2(1.0, c) * fastLanczos2(x);
}

vec2 edgeDirection(vec4 left, vec4 right)
{
	float RxLz = right.x - left.z;
	float RwLy = right.w - left.y;
	vec2 delta = vec2(RxLz + RwLy, RxLz - RwLy);
	float lengthInv = inversesqrt(dot(delta, delta) + 3.075740e-05);

	return delta * lengthInv;
}

void main()
{
	vec4 color = textureLod(ps0, in_TEXCOORD0.xy, 0.0);

	highp vec2 imgCoord = (in_TEXCOORD0.xy * ViewportInfo.zw) + vec2(-0.5, 0.5);
	highp vec2 coord = floor(imgCoord) * ViewportInfo.xy;

	vec2 pl = fract(imgCoord);

	vec4 left = textureGather(ps0, coord, 1);

	float edgeVote = abs(left.z - left.y) + abs(color.y - left.y) + abs(color.y - left.z);

	if(edgeVote > EdgeThreshold)
	{
		coord.x += ViewportInfo.x;

		vec4 right = textureGather(ps0, coord + vec2(ViewportInfo.x, 0.0), 1);
		vec4 upDown;
		upDown.xy = textureGather(ps0, coord + vec2(0.0, -ViewportInfo.y), 1).wz;
		upDown.zw = textureGather(ps0, coord + vec2(0.0, ViewportInfo.y), 1).yx;

		float mean = (left.y + left.z + right.x + right.w) * 0.25;

		left -= mean;
		right -= mean;
		upDown -= mean;
		color.w = color.y - mean;

		float sum = abs(left.x) + abs(left.y) + abs(left.z) + abs(left.w)
		          + abs(right.x) + abs(right.y) + abs(right.z) + abs(right.w)
		          + abs(upDown.x) + abs(upDown.y) + abs(upDown.z) + abs(upDown.w);

		float sumMean = 10.14185 / sum;
		float std = sumMean * sumMean;

		vec3 data = vec3(std, edgeDirection(left, right));

		vec2 aWY = weightY(pl + vec2(0.0, 1.0), upDown.x, data);
		aWY += weightY(pl + vec2(-1.0, 1.0), upDown.y, data);
		aWY += weightY(pl + vec2(-1.0, -2.0), upDown.z, data);
		aWY += weightY(pl + vec2(0.0, -2.0), upDown.w, data);
		aWY += weightY(pl + vec2(1.0, -1.0), left.x, data);
		aWY += weightY(pl + vec2(0.0, -1.0), left.y, data);
		aWY += weightY(pl, left.z, data);
		aWY += weightY(pl + vec2(1.0, 0.0), left.w, data);
		aWY += weightY(pl + vec2(-1.0, -1.0), right.x, data);
		aWY += weightY(pl + vec2(-2.0, -1.0), right.y, data);
		aWY += weightY(pl + vec2(-2.0, 0.0), right.z, data);
		aWY += weightY(pl + vec2(-1.0, 0.0), right.w, data);

		float finalY = aWY.y / aWY.x;
		float maxY = max(max(left.y, left.z), max(right.x, right.w));
		float minY = min(min(left.y, left.z), min(right.x, right.w));

		float deltaY = clamp(EdgeSharpness * finalY, minY, maxY) - color.w;

		deltaY = clamp(deltaY, -23.0 / 255.0, 23.0 / 255.0);

		color.rgb = clamp(color.rgb + deltaY, 0.0, 1.0);
	}

	color.w = 1.0;
	out_Target0 = color;
}
