#version 310 es
// =============================================================================
// FSR RCAS - Robust Contrast Adaptive Sharpening
// Pass 2 of 2 for AMD FidelityFX Super Resolution v1.0.2
//
// Based on the mpv port by agyild (MIT):
//   https://gist.github.com/agyild/82219c545228d70c5604f865ce0b0ce5
// Original FSR by AMD, (c) 2021 Advanced Micro Devices, Inc.
//
// Adaptations from the mpv port:
//   - Converted to standalone GLSL ES 3.10 fragment shader
//   - Operates on RGB (3 channels) instead of LUMA
//   - Uses mediump precision
//   - Removed all mpv //!HOOK metadata
//
// Pipeline:
//   LowRes --> [fsr_easu.frag] --> EASU_TEX --> [fsr_rcas.frag] --> Output
//
// IMPORTANT: bind the output of the EASU pass to u_source and set u_sourceSize
// to the EASU output size (i.e. the target display size). u_outputSize must
// match u_sourceSize for RCAS (1:1 sharpening pass).
//
// Required uniforms:
//   u_source     : sampler2D, EASU output, LINEAR CLAMP sampler
//   u_sourceSize : vec2, EASU output size in pixels
// =============================================================================

precision mediump float;
precision mediump int;

uniform sampler2D u_source;
uniform vec2 u_sourceSize;

in  vec2 v_texCoord;
out vec4 o_color;

// ---- User tunables ----------------------------------------------------------
// Sharpening strength. Scale: 0.0 = maximum sharpening, 2.0 = very mild.
// AMD recommends ~0.2 as a reasonable default.
#define SHARPNESS 0.2

// Noise-aware attenuation. Disabling saves a few ALUs but over-sharpens noisy
// input (streaming artifacts, grain). 0 or 1.
#define FSR_RCAS_DENOISE 1

// ---- Helpers ----------------------------------------------------------------
#define FSR_RCAS_LIMIT (0.25 - (1.0 / 16.0))

float APrxMedRcpF1(float a) {
    float b = uintBitsToFloat(uint(0x7ef19fff) - floatBitsToUint(a));
    return b * (-b * a + 2.0);
}
float AMax3F1(float x, float y, float z) { return max(x, max(y, z)); }
float AMin3F1(float x, float y, float z) { return min(x, min(y, z)); }

// BT.709 luma.
float luma(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

// Fetch a pixel by absolute offset in pixels relative to the current fragment.
vec3 fetch(vec2 offset) {
    vec2 uv = v_texCoord + offset / u_sourceSize;
    return texture(u_source, uv).rgb;
}

void main() {
    // 3x3 neighborhood in '+' shape around current pixel 'e'.
    //       b
    //     d e f
    //       h
    vec3 Bc = fetch(vec2( 0.0, -1.0));
    vec3 Dc = fetch(vec2(-1.0,  0.0));
    vec3 Ec = texture(u_source, v_texCoord).rgb;
    vec3 Fc = fetch(vec2( 1.0,  0.0));
    vec3 Hc = fetch(vec2( 0.0,  1.0));

    // Work in luma space for the lobe computation (matches AMD's design),
    // then apply the resulting filter coefficients to RGB at the end.
    float b = luma(Bc);
    float d = luma(Dc);
    float e = luma(Ec);
    float f = luma(Fc);
    float h = luma(Hc);

    float mn1L = min(AMin3F1(b, d, f), h);
    float mx1L = max(AMax3F1(b, d, f), h);

    vec2 peakC = vec2(1.0, -4.0);

    float hitMinL = min(mn1L, e) / (4.0 * mx1L);
    float hitMaxL = (peakC.x - max(mx1L, e)) / (4.0 * mn1L + peakC.y);
    float lobeL   = max(-hitMinL, hitMaxL);
    float lobe    = max(-FSR_RCAS_LIMIT, min(lobeL, 0.0))
                  * exp2(-clamp(float(SHARPNESS), 0.0, 2.0));

#if (FSR_RCAS_DENOISE == 1)
    // Attenuate sharpening on noisy regions.
    float nz = 0.25 * b + 0.25 * d + 0.25 * f + 0.25 * h - e;
    nz = clamp(abs(nz) * APrxMedRcpF1(
            AMax3F1(AMax3F1(b, d, e), f, h) -
            AMin3F1(AMin3F1(b, d, e), f, h)),
         0.0, 1.0);
    nz = -0.5 * nz + 1.0;
    lobe *= nz;
#endif

    float rcpL = APrxMedRcpF1(4.0 * lobe + 1.0);
    // Apply the same lobe coefficients to each RGB channel.
    vec3 rgb = (lobe * Bc + lobe * Dc + lobe * Hc + lobe * Fc + Ec) * rcpL;

    o_color = vec4(clamp(rgb, 0.0, 1.0), 1.0);
}
