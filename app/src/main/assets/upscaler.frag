#version 310 es
precision mediump float;

in vec4 in_TEXCOORD0;
out vec4 fragColor;

// Uniformes passés par GlesPassthroughBridge
uniform sampler2D sTexture;
uniform vec2 uInputSize;
uniform vec2 uOutputSize;
uniform vec2 uInvInputSize;

// ============================================================================
// Option 1 : Fast Bicubic Upscaling (5-tap)
// Approximation d'un filtre Catmull-Rom en utilisant l'interpolation
// bilinéaire matérielle (Hardware Bilinear) pour réduire le nombre
// de fetchs de texture de 16 à seulement 5. Très bonne qualité et rapide.
// ============================================================================
vec4 fastBicubic(sampler2D tex, vec2 uv, vec2 texSize, vec2 invTexSize) {
    vec2 p = uv * texSize - 0.5;
    vec2 i = floor(p);
    vec2 f = p - i;

    vec2 f2 = f * f;
    vec2 f3 = f2 * f;

    // Calcul des poids pour Catmull-Rom
    vec2 w0 = f2 - 0.5 * (f3 + f);
    vec2 w1 = 1.5 * f3 - 2.5 * f2 + 1.0;
    vec2 w2 = -1.5 * f3 + 2.0 * f2 + 0.5 * f;
    vec2 w3 = 0.5 * (f3 - f2);

    vec2 w12 = w1 + w2;
    vec2 offset12 = w2 / w12;

    vec2 t0 = (i - 1.0) * invTexSize;
    vec2 t3 = (i + 2.0) * invTexSize;
    vec2 t12 = (i + offset12) * invTexSize;

    vec4 col = texture(tex, vec2(t12.x, t0.y))  * w12.x * w0.y  +
               texture(tex, vec2(t0.x,  t12.y)) * w0.x  * w12.y +
               texture(tex, vec2(t12.x, t12.y)) * w12.x * w12.y +
               texture(tex, vec2(t3.x,  t12.y)) * w3.x  * w12.y +
               texture(tex, vec2(t12.x, t3.y))  * w12.x * w3.y;

    return col;
}

// ============================================================================
// Option 2 : Sharper Bilinear (1-tap)
// Utilise la fonction smoothstep pour affiner l'interpolation bilinéaire
// matérielle. Nettement plus tranchant que le bilinéaire classique tout en
// étant quasiment au même coût en performances.
// ============================================================================
vec4 sharperBilinear(sampler2D tex, vec2 uv, vec2 texSize, vec2 invTexSize) {
    vec2 p = uv * texSize - 0.5;
    vec2 i = floor(p);
    vec2 f = p - i;

    // Smoothstep appliqué à la fraction
    f = f * f * (3.0 - 2.0 * f);

    vec2 tc = (i + f + 0.5) * invTexSize;
    return texture(tex, tc);
}

void main() {
    vec2 uv = in_TEXCOORD0.xy;

    // Choisissez l'option d'upscaling ici en commentant/décommentant :

    // --- Option 1 : Bicubic ---
    // vec4 color = fastBicubic(sTexture, uv, uInputSize, uInvInputSize);

    // --- Option 2 : Sharper Bilinear ---
    // vec4 color = sharperBilinear(sTexture, uv, uInputSize, uInvInputSize);

    // --- Option 3 : Passthrough (Hardware Bilinear de base) ---
    vec4 color = texture(sTexture, uv);

    fragColor = vec4(clamp(color.rgb, 0.0, 1.0), 1.0);
}