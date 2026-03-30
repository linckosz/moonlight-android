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

// Optimisation: vec4 simple au lieu de tableau pour registres uniformes constants.
uniform highp vec4 ViewportInfo;
uniform mediump sampler2D ps0;

in highp vec4 in_TEXCOORD0;
out vec4 out_Target0;

float fastLanczos2(float x)
{
	float wA = x - 4.0;
	float wB = x * wA - wA;
	return wB * (wA * wA); // Optimisation FMA: (wA * wA) calculé directement
}

vec2 weightY(float dx, float dy, float c, vec3 data)
{
	float std = data.x;
	vec2 dir = data.yz;

	float edgeDis = ((dx * dir.y) + (dy * dir.x));
	float x = (((dx * dx) + (dy * dy)) + ((edgeDis * edgeDis) * ((clamp(((c * c) * std), 0.0, 1.0) * 0.7) - 1.0)));

	float w = fastLanczos2(x);
	return vec2(w, w * c);	
}

vec2 edgeDirection(vec4 left, vec4 right)
{
	float RxLz = right.x - left.z;
	float RwLy = right.w - left.y;
	vec2 delta = vec2(RxLz + RwLy, RxLz - RwLy);
	float lengthInv = inversesqrt(delta.x * delta.x + 3.075740e-05 + delta.y * delta.y);
	return delta * lengthInv; // Optimisation: Vectorisation de la multiplication
}

void main()
{
	// Optimisation: Lecture directe sans branchement (OperationMode = 1 fixe)
	vec4 color = textureLod(ps0, in_TEXCOORD0.xy, 0.0);

	highp vec2 imgCoord = (in_TEXCOORD0.xy * ViewportInfo.zw) + vec2(-0.5, 0.5);
	highp vec2 coord = floor(imgCoord) * ViewportInfo.xy;

	// Optimisation: fract() matérielle native au lieu de soustraction de floor()
	vec2 pl = fract(imgCoord);

	// Optimisation: constante `1` dans le textureGather (canal y)
	vec4 left = textureGather(ps0, coord, 1);

	// Optimisation: accède directement à color.y (swizzle) au lieu de color[mode]
	float edgeVote = abs(left.z - left.y) + abs(color.y - left.y) + abs(color.y - left.z);

	if(edgeVote > EdgeThreshold)
	{
		coord.x += ViewportInfo.x;

		vec4 right = textureGather(ps0, coord + vec2(ViewportInfo.x, 0.0), 1);
		vec4 upDown;
		upDown.xy = textureGather(ps0, coord + vec2(0.0, -ViewportInfo.y), 1).wz;
		upDown.zw = textureGather(ps0, coord + vec2(0.0, ViewportInfo.y), 1).yx;

		float mean = (left.y + left.z + right.x + right.w) * 0.25;

		// Vectorisation des soustractions de moyenne
		left -= mean;
		right -= mean;
		upDown -= mean;
		color.w = color.y - mean;

		// Simplification de la somme avec un seul arbre d'addition
		float sum = abs(left.x) + abs(left.y) + abs(left.z) + abs(left.w)
		          + abs(right.x) + abs(right.y) + abs(right.z) + abs(right.w)
		          + abs(upDown.x) + abs(upDown.y) + abs(upDown.z) + abs(upDown.w);

		float sumMean = 10.14185 / sum;
		float std = sumMean * sumMean;

		vec3 data = vec3(std, edgeDirection(left, right));

		vec2 aWY = weightY(pl.x, pl.y + 1.0, upDown.x, data);
		aWY += weightY(pl.x - 1.0, pl.y + 1.0, upDown.y, data);
		aWY += weightY(pl.x - 1.0, pl.y - 2.0, upDown.z, data);
		aWY += weightY(pl.x, pl.y - 2.0, upDown.w, data);
		aWY += weightY(pl.x + 1.0, pl.y - 1.0, left.x, data);
		aWY += weightY(pl.x, pl.y - 1.0, left.y, data);
		aWY += weightY(pl.x, pl.y, left.z, data);
		aWY += weightY(pl.x + 1.0, pl.y, left.w, data);
		aWY += weightY(pl.x - 1.0, pl.y - 1.0, right.x, data);
		aWY += weightY(pl.x - 2.0, pl.y - 1.0, right.y, data);
		aWY += weightY(pl.x - 2.0, pl.y, right.z, data);
		aWY += weightY(pl.x - 1.0, pl.y, right.w, data);

		float finalY = aWY.y / aWY.x;
		float maxY = max(max(left.y, left.z), max(right.x, right.w));
		float minY = min(min(left.y, left.z), min(right.x, right.w));

		float deltaY = clamp(EdgeSharpness * finalY, minY, maxY) - color.w;

		// Lissage des contrastes élevés
		deltaY = clamp(deltaY, -23.0 / 255.0, 23.0 / 255.0);

		// Optimisation: Vectorisation du clamp sur RGB en une seule passe
		color.rgb = clamp(color.rgb + deltaY, 0.0, 1.0);
	}

	color.w = 1.0;  // Le canal alpha n'est pas utilisé (opaque)
	out_Target0 = color;
}
