#version 310 es
// =============================================================================
// FSR EASU - Edge Adaptive Spatial Upsampling
// Pass 1 of 2 for AMD FidelityFX Super Resolution v1.0.2
//
// Based on the mpv port by agyild (MIT):
//   https://gist.github.com/agyild/82219c545228d70c5604f865ce0b0ce5
// Original FSR by AMD, (c) 2021 Advanced Micro Devices, Inc.
//
// Adaptations from the mpv port:
//   - Converted to standalone GLSL ES 3.10 fragment shader
//   - Operates on RGB (3 channels) instead of LUMA, since Moonlight-Android
//     receives decoded frames as RGB via SurfaceTexture. Direction analysis
//     is still performed on luma derived from RGB (BT.709 weights).
//   - Removed all mpv //!HOOK metadata
//   - Uses mediump precision by default (suitable for Mali/Adreno low-end).
//     Switch to highp only if you see banding on very large upscales.
//
// Pipeline:
//   LowRes RGB  --> [fsr_easu.frag]  --> EASU RGB  --> [fsr_rcas.frag] --> Output
//
// Required uniforms from host (set once per frame, or when sizes change):
//   u_source        : sampler2D, input low-res texture, LINEAR CLAMP sampler
//   u_sourceSize    : vec2, input texture size in pixels    (e.g. 1280,720)
//   u_outputSize    : vec2, output render target size in px (e.g. 2560,1440)
// =============================================================================

precision mediump float;
precision mediump int;

uniform sampler2D u_source;
uniform vec2 u_sourceSize;
uniform vec2 u_outputSize;

in  vec2 v_texCoord; // fullscreen-quad output coord in [0,1]
out vec4 o_color;

// ---- User tunables ----------------------------------------------------------
// Disabling DERING gives a small perf win on weak GPUs at the cost of minor
// overshoot/undershoot visible on very sharp edges. 0 or 1.
#define FSR_EASU_DERING 1

// Single-pass direction analysis is faster but slightly less accurate. 0 or 1.
#define FSR_EASU_SIMPLE_ANALYSIS 0

// Skip EASU for flat (non-edge) pixels and fall back to bilinear. Big perf win
// on content with large uniform areas (UI, skies, etc.). 0 or 1.
#define FSR_EASU_QUIT_EARLY 0

#if (FSR_EASU_QUIT_EARLY == 1)
    #define FSR_EASU_DIR_THRESHOLD 64.0
#else
    #define FSR_EASU_DIR_THRESHOLD 32768.0
#endif

// ---- Helpers ----------------------------------------------------------------
// AMD's bit-hack reciprocal / rsqrt approximations. GLSL ES 3.10 supports
// uintBitsToFloat / floatBitsToUint natively.
float APrxLoRcpF1(float a) {
    return uintBitsToFloat(uint(0x7ef07ebb) - floatBitsToUint(a));
}
float APrxLoRsqF1(float a) {
    return uintBitsToFloat(uint(0x5f347d74) - (floatBitsToUint(a) >> uint(1)));
}
float AMin3F1(float x, float y, float z) { return min(x, min(y, z)); }
float AMax3F1(float x, float y, float z) { return max(x, max(y, z)); }

// BT.709 luma for direction analysis. FSR was designed to work on luma;
// feeding it RGB.g directly is often used as a cheap approximation but
// proper luma is barely more expensive and gives better edge detection.
float luma(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

// ---- FSR EASU kernel --------------------------------------------------------
void FsrEasuTap(
    inout vec3 aC,    // accumulated color (RGB)
    inout float aW,   // accumulated weight
    vec2 off,         // pixel offset from resolve position to tap
    vec2 dir,         // gradient direction
    vec2 len,         // length
    float lob,        // negative lobe strength
    float clp,        // clipping point
    vec3 c            // tap color
) {
    // Rotate offset by direction.
    vec2 v;
    v.x = (off.x *  dir.x) + (off.y * dir.y);
    v.y = (off.x * -dir.y) + (off.y * dir.x);
    // Anisotropy.
    v *= len;
    // Compute distance^2.
    float d2 = v.x * v.x + v.y * v.y;
    d2 = min(d2, clp);
    // Approximation of lanczos2 window.
    float wB = (2.0 / 5.0) * d2 - 1.0;
    float wA = lob * d2 - 1.0;
    wB *= wB;
    wA *= wA;
    wB = (25.0 / 16.0) * wB - ((25.0 / 16.0) - 1.0);
    float w = wB * wA;
    aC += c * w;
    aW += w;
}

void FsrEasuSet(
    inout vec2 dir, inout float len, vec2 pp,
#if (FSR_EASU_SIMPLE_ANALYSIS == 1)
    float b, float c, float i, float j, float f, float e,
    float k, float l, float h, float g, float o, float n
#else
    bool biS, bool biT, bool biU, bool biV,
    float lA, float lB, float lC, float lD, float lE
#endif
) {
#if (FSR_EASU_SIMPLE_ANALYSIS == 1)
    vec4 w;
    w.x = (1.0 - pp.x) * (1.0 - pp.y);
    w.y =  pp.x        * (1.0 - pp.y);
    w.z = (1.0 - pp.x) *  pp.y;
    w.w =  pp.x        *  pp.y;
    float lA = dot(w, vec4(b, c, f, g));
    float lB = dot(w, vec4(e, f, i, j));
    float lC = dot(w, vec4(f, g, j, k));
    float lD = dot(w, vec4(g, h, k, l));
    float lE = dot(w, vec4(j, k, n, o));
#else
    float w = 0.0;
    if (biS) w = (1.0 - pp.x) * (1.0 - pp.y);
    if (biT) w =  pp.x        * (1.0 - pp.y);
    if (biU) w = (1.0 - pp.x) *  pp.y;
    if (biV) w =  pp.x        *  pp.y;
#endif
    float dc = lD - lC;
    float cb = lC - lB;
    float lenX = max(abs(dc), abs(cb));
    lenX = APrxLoRcpF1(lenX);
    float dirX = lD - lB;
    lenX = clamp(abs(dirX) * lenX, 0.0, 1.0);
    lenX *= lenX;

    float ec = lE - lC;
    float ca = lC - lA;
    float lenY = max(abs(ec), abs(ca));
    lenY = APrxLoRcpF1(lenY);
    float dirY = lE - lA;
    lenY = clamp(abs(dirY) * lenY, 0.0, 1.0);
    lenY *= lenY;

#if (FSR_EASU_SIMPLE_ANALYSIS == 1)
    len = lenX + lenY;
    dir = vec2(dirX, dirY);
#else
    dir += vec2(dirX, dirY) * w;
    len += dot(vec2(w), vec2(lenX, lenY));
#endif
}

// Helper: fetch a tap at integer-offset (fp + off) in source pixels,
// returning the RGB color at that pixel. Uses LINEAR CLAMP sampler.
vec3 tapRGB(vec2 fp, vec2 off) {
    vec2 uv = (fp + off) / u_sourceSize;
    return texture(u_source, uv).rgb;
}

void main() {
    // Position in source pixel space (F = top-left of the 2x2 block around
    // the resolve point).
    vec2 pp = v_texCoord * u_sourceSize - vec2(0.5);
    vec2 fp = floor(pp);
    pp -= fp;

    // 12-tap kernel. Gather the RGB of each tap...
    vec3 Bc = tapRGB(fp, vec2(0.5, -0.5));
    vec3 Cc = tapRGB(fp, vec2(1.5, -0.5));

    vec3 Ec = tapRGB(fp, vec2(-0.5,  0.5));
    vec3 Fc = tapRGB(fp, vec2( 0.5,  0.5));
    vec3 Gc = tapRGB(fp, vec2( 1.5,  0.5));
    vec3 Hc = tapRGB(fp, vec2( 2.5,  0.5));

    vec3 Ic = tapRGB(fp, vec2(-0.5,  1.5));
    vec3 Jc = tapRGB(fp, vec2( 0.5,  1.5));
    vec3 Kc = tapRGB(fp, vec2( 1.5,  1.5));
    vec3 Lc = tapRGB(fp, vec2( 2.5,  1.5));

    vec3 Nc = tapRGB(fp, vec2(0.5, 2.5));
    vec3 Oc = tapRGB(fp, vec2(1.5, 2.5));

    // ...and derive luma for direction analysis.
    float bL = luma(Bc);
    float cL = luma(Cc);
    float eL = luma(Ec);
    float fL = luma(Fc);
    float gL = luma(Gc);
    float hL = luma(Hc);
    float iL = luma(Ic);
    float jL = luma(Jc);
    float kL = luma(Kc);
    float lL = luma(Lc);
    float nL = luma(Nc);
    float oL = luma(Oc);

    // Edge direction and length.
    vec2 dir = vec2(0.0);
    float len = 0.0;
#if (FSR_EASU_SIMPLE_ANALYSIS == 1)
    FsrEasuSet(dir, len, pp, bL, cL, iL, jL, fL, eL, kL, lL, hL, gL, oL, nL);
#else
    FsrEasuSet(dir, len, pp, true,  false, false, false, bL, eL, fL, gL, jL);
    FsrEasuSet(dir, len, pp, false, true,  false, false, cL, fL, gL, hL, kL);
    FsrEasuSet(dir, len, pp, false, false, true,  false, fL, iL, jL, kL, nL);
    FsrEasuSet(dir, len, pp, false, false, false, true,  gL, jL, kL, lL, oL);
#endif

    // Normalize direction.
    vec2 dir2 = dir * dir;
    float dirR = dir2.x + dir2.y;
    bool zro = dirR < (1.0 / FSR_EASU_DIR_THRESHOLD);
    dirR = APrxLoRsqF1(dirR);

#if (FSR_EASU_QUIT_EARLY == 1)
    if (zro) {
        // Flat region: just bilinear blend the inner 4 taps and exit.
        vec4 w;
        w.x = (1.0 - pp.x) * (1.0 - pp.y);
        w.y =  pp.x        * (1.0 - pp.y);
        w.z = (1.0 - pp.x) *  pp.y;
        w.w =  pp.x        *  pp.y;
        vec3 rgb = Fc * w.x + Gc * w.y + Jc * w.z + Kc * w.w;
        o_color = vec4(clamp(rgb, 0.0, 1.0), 1.0);
        return;
    }
#else
    dirR  = zro ? 1.0 : dirR;
    dir.x = zro ? 1.0 : dir.x;
#endif
    dir *= vec2(dirR);

    len = len * 0.5;
    len *= len;

    float stretch = (dir.x * dir.x + dir.y * dir.y)
                  * APrxLoRcpF1(max(abs(dir.x), abs(dir.y)));
    vec2  len2    = vec2(1.0 + (stretch - 1.0) * len, 1.0 - 0.5 * len);
    float lob     = 0.5 + ((1.0 / 4.0 - 0.04) - 0.5) * len;
    float clp     = APrxLoRcpF1(lob);

    // Accumulate all 12 taps, weighted by the rotated/stretched lanczos2 kernel.
    vec3  aC = vec3(0.0);
    float aW = 0.0;
    FsrEasuTap(aC, aW, vec2( 0.0, -1.0) - pp, dir, len2, lob, clp, Bc);
    FsrEasuTap(aC, aW, vec2( 1.0, -1.0) - pp, dir, len2, lob, clp, Cc);
    FsrEasuTap(aC, aW, vec2(-1.0,  1.0) - pp, dir, len2, lob, clp, Ic);
    FsrEasuTap(aC, aW, vec2( 0.0,  1.0) - pp, dir, len2, lob, clp, Jc);
    FsrEasuTap(aC, aW, vec2( 0.0,  0.0) - pp, dir, len2, lob, clp, Fc);
    FsrEasuTap(aC, aW, vec2(-1.0,  0.0) - pp, dir, len2, lob, clp, Ec);
    FsrEasuTap(aC, aW, vec2( 1.0,  1.0) - pp, dir, len2, lob, clp, Kc);
    FsrEasuTap(aC, aW, vec2( 2.0,  1.0) - pp, dir, len2, lob, clp, Lc);
    FsrEasuTap(aC, aW, vec2( 2.0,  0.0) - pp, dir, len2, lob, clp, Hc);
    FsrEasuTap(aC, aW, vec2( 1.0,  0.0) - pp, dir, len2, lob, clp, Gc);
    FsrEasuTap(aC, aW, vec2( 1.0,  2.0) - pp, dir, len2, lob, clp, Oc);
    FsrEasuTap(aC, aW, vec2( 0.0,  2.0) - pp, dir, len2, lob, clp, Nc);

    vec3 rgb = aC / aW;

#if (FSR_EASU_DERING == 1)
    // Deringing clamp against the inner 2x2 RGB neighborhood.
    vec3 mn = min(min(Fc, Gc), min(Jc, Kc));
    vec3 mx = max(max(Fc, Gc), max(Jc, Kc));
    rgb = clamp(rgb, mn, mx);
#endif

    o_color = vec4(clamp(rgb, 0.0, 1.0), 1.0);
}
