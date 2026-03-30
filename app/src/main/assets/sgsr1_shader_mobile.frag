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

#define EdgeThreshold (16.0 / 255.0)
#define EdgeSharpness 2.0

// Optimisation: Utiliser un vec4 simple au lieu d'un tableau permet au GPU de
// placer cette variable dans les registres constants les plus rapides, sans overhead d'indexation.
uniform highp vec4 ViewportInfo;
uniform mediump sampler2D ps0;

in highp vec4 in_TEXCOORD0;
out vec4 out_Target0;

float fastLanczos2(float x)
{
	float wA = x - 4.0;
	float wB = x * wA - wA;
	return wB * (wA * wA); // Optimisation mathématique: (wA * wA) est évalué et multiplié directement
}

vec2 weightY(float dx, float dy, float c, float std)
{
	float x = ((dx * dx) + (dy * dy)) * 0.55 + clamp(abs(c) * std, 0.0, 1.0);
	float w = fastLanczos2(x);
	return vec2(w, w * c);	
}

void main()
{
	// Optimisation: OperationMode == 1 est hardcodé. On enlève le if/else
	vec4 color = textureLod(ps0, in_TEXCOORD0.xy, 0.0);

	highp vec2 imgCoord = (in_TEXCOORD0.xy * ViewportInfo.zw) + vec2(-0.5, 0.5);
	highp vec2 coord = floor(imgCoord) * ViewportInfo.xy;

	// Optimisation: fract() est une instruction matérielle native (1 cycle)
	// au lieu de l'addition/soustraction `imgCoord + (-imgCoordPixel)`.
	vec2 pl = fract(imgCoord);

	// Optimisation: textureGather attend un index constant. On donne `1` (canal Green/Y) directement.
	vec4 left = textureGather(ps0, coord, 1);

	// Optimisation: color[mode] est remplacé par color.y. Accéder dynamiquement
	// à un vecteur (.xyz[mode]) peut bloquer des optimisations de registre sur certains GPU mobiles.
	float edgeVote = abs(left.z - left.y) + abs(color.y - left.y) + abs(color.y - left.z);

	if(edgeVote > EdgeThreshold)
	{
		coord.x += ViewportInfo.x;

		vec4 right = textureGather(ps0, coord + vec2(ViewportInfo.x, 0.0), 1);
		vec4 upDown;
		upDown.xy = textureGather(ps0, coord + vec2(0.0, -ViewportInfo.y), 1).wz;
		upDown.zw = textureGather(ps0, coord + vec2(0.0, ViewportInfo.y), 1).yx;

		float mean = (left.y + left.z + right.x + right.w) * 0.25;

		// Vectorisation des soustractions de la moyenne
		left -= mean;
		right -= mean;
		upDown -= mean;
		color.w = color.y - mean;

		// Simplification de la somme avec un seul arbre d'addition
		float sum = abs(left.x) + abs(left.y) + abs(left.z) + abs(left.w)
		          + abs(right.x) + abs(right.y) + abs(right.z) + abs(right.w)
		          + abs(upDown.x) + abs(upDown.y) + abs(upDown.z) + abs(upDown.w);

		float std = 2.181818 / sum;

		vec2 aWY = weightY(pl.x, pl.y + 1.0, upDown.x, std);
		aWY += weightY(pl.x - 1.0, pl.y + 1.0, upDown.y, std);
		aWY += weightY(pl.x - 1.0, pl.y - 2.0, upDown.z, std);
		aWY += weightY(pl.x, pl.y - 2.0, upDown.w, std);
		aWY += weightY(pl.x + 1.0, pl.y - 1.0, left.x, std);
		aWY += weightY(pl.x, pl.y - 1.0, left.y, std);
		aWY += weightY(pl.x, pl.y, left.z, std);
		aWY += weightY(pl.x + 1.0, pl.y, left.w, std);
		aWY += weightY(pl.x - 1.0, pl.y - 1.0, right.x, std);
		aWY += weightY(pl.x - 2.0, pl.y - 1.0, right.y, std);
		aWY += weightY(pl.x - 2.0, pl.y, right.z, std);
		aWY += weightY(pl.x - 1.0, pl.y, right.w, std);

		float finalY = aWY.y / aWY.x;

		float maxY = max(max(left.y, left.z), max(right.x, right.w));
		float minY = min(min(left.y, left.z), min(right.x, right.w));
		finalY = clamp(EdgeSharpness * finalY, minY, maxY);

		float deltaY = finalY - color.w;

		// Lissage des contrastes élevés
		deltaY = clamp(deltaY, -23.0 / 255.0, 23.0 / 255.0);

		// Optimisation: Vectorisation du clamp sur RGB en une seule passe plutôt que 3 appels séparés
		color.rgb = clamp(color.rgb + deltaY, 0.0, 1.0);
	}

	color.w = 1.0;  // Le canal alpha n'est pas utilisé (opaque)
	out_Target0 = color;
}
