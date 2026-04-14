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

// OPTIMISATION DES PERFORMANCES :
// Puisque la résolution d'entrée ne change jamais au cours de la session,
// nous remplaçons l'uniform 'ViewportInfo' par des macros (#define).
// Cela permet au compilateur du GPU de faire du "Constant Folding" (pré-calcul)
// et d'alléger considérablement l'ALU.
// Côté Java, lors du chargement de ce fichier, vous remplacerez "1280.0" et "720.0"
// par vos vraies résolutions d'entrée avec un simple String.replace() avant de compiler le shader.
#define INPUT_WIDTH 1280.0
#define INPUT_HEIGHT 720.0

#define VIEWPORT_X (1.0 / INPUT_WIDTH)
#define VIEWPORT_Y (1.0 / INPUT_HEIGHT)
#define VIEWPORT_Z INPUT_WIDTH
#define VIEWPORT_W INPUT_HEIGHT

#define ViewportInfo vec4(VIEWPORT_X, VIEWPORT_Y, VIEWPORT_Z, VIEWPORT_W)

////////////////////////
////////////////////////
////////////////////////

uniform mediump sampler2D ps0;

in highp vec4 in_TEXCOORD0;
out mediump vec4 out_Target0;

// Utiliser half precision (mediump) partout où l'extreme précision (highp)
// n'est pas strictement nécessaire permet d'économiser de la bande passante
// et des cycles ALU sur les GPU mobiles (Mali, Adreno).
mediump float fastLanczos2(mediump float x)
{
	mediump float wA = x - 4.0;
	mediump float wB = x * wA - wA;
	return wB * (wA * wA);
}

mediump vec2 weightY(mediump vec2 d, mediump float c, mediump vec3 data)
{
	mediump float std = data.x;
	mediump vec2 dir = data.yz;

	mediump float edgeDis = dot(d, dir.yx);
	mediump float x = dot(d, d) + (edgeDis * edgeDis) * (clamp((c * c) * std, 0.0, 1.0) * 0.7 - 1.0);

	return vec2(1.0, c) * fastLanczos2(x);
}

mediump vec2 edgeDirection(mediump vec4 left, mediump vec4 right)
{
	mediump float RxLz = right.x - left.z;
	mediump float RwLy = right.w - left.y;
	mediump vec2 delta = vec2(RxLz + RwLy, RxLz - RwLy);
	mediump float lengthInv = inversesqrt(dot(delta, delta) + 3.075740e-05);

	return delta * lengthInv;
}

void main()
{
	// 1. Texture input YUV
	mediump vec4 base_yuv = textureLod(ps0, in_TEXCOORD0.xy, 0.0);

	// 2. Extract Y (luma) et UV (chroma)
	mediump float y_luma = base_yuv.r;
	mediump vec2 uv_chroma = base_yuv.gb;

	highp vec2 imgCoord = (in_TEXCOORD0.xy * ViewportInfo.zw) + vec2(-0.5, 0.5);
	highp vec2 coord = floor(imgCoord) * ViewportInfo.xy;

	mediump vec2 pl = fract(imgCoord);

	// 3. Texture gather
	mediump vec4 left = textureGather(ps0, coord, 0);

	// Edge vote basé sur la luma (Y)
	mediump float edgeVote = abs(left.z - left.y) + abs(y_luma - left.y) + abs(y_luma - left.z);

	// Applique SGSR à Y (luma) si le bord est suffisamment marqué
	if(edgeVote > EdgeThreshold)
	{
		coord.x += ViewportInfo.x;

		mediump vec4 right = textureGather(ps0, coord + vec2(ViewportInfo.x, 0.0), 0);
		mediump vec4 upDown;
		upDown.xy = textureGather(ps0, coord + vec2(0.0, -ViewportInfo.y), 0).wz;
		upDown.zw = textureGather(ps0, coord + vec2(0.0, ViewportInfo.y), 0).yx;

		mediump float mean = (left.y + left.z + right.x + right.w) * 0.25;

		left -= mean;
		right -= mean;
		upDown -= mean;

		mediump float y_mean_diff = y_luma - mean;

		// Utiliser la fonction vectorielle intrinsèque 'abs' et 'dot' pour sommer
		// les valeurs absolues est souvent plus rapide que 12 additions séparées
		mediump float sum = dot(abs(left), vec4(1.0))
		                  + dot(abs(right), vec4(1.0))
		                  + dot(abs(upDown), vec4(1.0));

		mediump float sumMean = 10.14185 / sum;
		mediump float std = sumMean * sumMean;

		mediump vec3 data = vec3(std, edgeDirection(left, right));

		mediump vec2 aWY = weightY(pl + vec2(0.0, 1.0), upDown.x, data);
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

		mediump float finalY = aWY.y / aWY.x;

		// Vectorisation du min/max
		mediump vec2 leftYZ_rightXW = vec2(max(left.y, left.z), max(right.x, right.w));
		mediump float maxY = max(leftYZ_rightXW.x, leftYZ_rightXW.y);

		mediump vec2 min_leftYZ_rightXW = vec2(min(left.y, left.z), min(right.x, right.w));
		mediump float minY = min(min_leftYZ_rightXW.x, min_leftYZ_rightXW.y);

		mediump float deltaY = clamp(EdgeSharpness * finalY, minY, maxY) - y_mean_diff;

		deltaY = clamp(deltaY, -23.0 / 255.0, 23.0 / 255.0);

		// Application du sharpening (SGSR) uniquement sur la luma (Y)
		y_luma = clamp(y_luma + deltaY, 0.0, 1.0);
	}

	// 4. Applique Bilinear à UV (chroma) (fait via textureLod plus haut)

	// 5. Recompose en texture de sortie YUV
	out_Target0 = vec4(y_luma, uv_chroma.x, uv_chroma.y, 1.0);
}