# FSR 1.0.2 for Android GLES 3.10 — Moonlight integration notes

Two-pass fragment shader port of AMD FidelityFX Super Resolution, adapted from
agyild's mpv port to standalone GLSL ES 3.10, operating on RGB (not luma).

## Files
- `fsr_easu.frag` — pass 1: edge-adaptive spatial upscaling
- `fsr_rcas.frag` — pass 2: robust contrast-adaptive sharpening

Both target `#version 310 es`, `precision mediump float;`, and validated with
`glslangValidator`.

## Pipeline

```
[decoded RGB texture]  --(fsr_easu.frag)-->  [FBO @ output size]
                                                     |
                                                     v
                                             (fsr_rcas.frag)
                                                     |
                                                     v
                                              [backbuffer]
```

You need **one intermediate FBO/texture sized at the final output resolution**
to hold the EASU result between passes.

## Vertex shader
Same trivial fullscreen-quad vertex shader you already use for SGSR1 should work.
Expected varying:
```glsl
#version 310 es
in vec2 a_position;   // [-1, 1]
in vec2 a_texCoord;   // [0, 1]
out vec2 v_texCoord;
void main() {
    v_texCoord = a_texCoord;
    gl_Position = vec4(a_position, 0.0, 1.0);
}
```

## Sampler setup (both passes)
```
GL_TEXTURE_MIN_FILTER = GL_LINEAR
GL_TEXTURE_MAG_FILTER = GL_LINEAR
GL_TEXTURE_WRAP_S     = GL_CLAMP_TO_EDGE
GL_TEXTURE_WRAP_T     = GL_CLAMP_TO_EDGE
```

## Uniforms

### EASU pass
| Uniform         | Value                                              |
|-----------------|----------------------------------------------------|
| `u_source`      | texture unit bound to the low-res decoded RGB      |
| `u_sourceSize`  | low-res width, height (e.g. `1280, 720`)           |
| `u_outputSize`  | final display size (e.g. `2560, 1440`)             |

Bind to an FBO whose attached color texture is sized `u_outputSize`.

### RCAS pass
| Uniform         | Value                                              |
|-----------------|----------------------------------------------------|
| `u_source`      | the EASU FBO color texture                         |
| `u_sourceSize`  | same as EASU's `u_outputSize`                      |

Bind to the default framebuffer (or your final render target).

## Tunables (edit the `#define`s at the top of each file)

### EASU
- `FSR_EASU_DERING` — default `1`. Set `0` for ~5% perf win, minor overshoot on hard edges.
- `FSR_EASU_SIMPLE_ANALYSIS` — default `0`. Set `1` for faster but slightly softer edge detection.
- `FSR_EASU_QUIT_EARLY` — default `0`. Set `1` to fall back to bilinear on flat regions; big win on mostly-uniform frames.

For Moonlight on low-end GPUs, a reasonable "perf" profile is:
```glsl
#define FSR_EASU_DERING 0
#define FSR_EASU_SIMPLE_ANALYSIS 1
#define FSR_EASU_QUIT_EARLY 1
```

### RCAS
- `SHARPNESS` — `0.0` = max sharp, `2.0` = barely any. Default `0.2`.
- `FSR_RCAS_DENOISE` — default `1`. Keep on for streamed content (stream compression adds noise that RCAS would otherwise amplify).

## Notes
- FSR1 is designed for **up to 4x area scale** (2x per dimension). Beyond that the
  results degrade. If you target >2x per dimension, apply FSR once then let the
  final blit handle the rest with bilinear.
- The shader expects the input to be in a **display-referred color space**
  (post tone-mapping, gamma-corrected). Standard decoded game streams already
  are, so no extra work is needed.
- Compared to SGSR1: FSR1 is more expensive (~12 taps vs ~4) but generally
  produces cleaner edges on photographic/3D content. Position it as the "quality"
  tier in your UI, SGSR1 as the "performance" tier.
- If you see a pure-red screen on some GLES drivers it's typically a missing
  `uintBitsToFloat` / `floatBitsToUint` support; these are core in GLES 3.00+
  so it shouldn't happen on anything modern, but worth keeping in mind.
