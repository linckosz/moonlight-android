package com.limelight.binding.video;

import android.content.Context;
import android.graphics.SurfaceTexture;
import android.opengl.EGL14;
import android.opengl.EGLConfig;
import android.opengl.EGLContext;
import android.opengl.EGLDisplay;
import android.opengl.EGLExt;
import android.opengl.EGLSurface;
import android.opengl.GLES11Ext;
import android.opengl.GLES31;
import android.view.Surface;
import com.limelight.LimeLog;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

/**
 * Cette classe agit comme un pont pour l'upscaling vidéo YUV natif.
 * Elle utilise un pipeline OpenGL ES 3.1 simplifié avec un shader d'upscaling FSR EASU.
 */
public class GlesUpscalerBridge implements SurfaceTexture.OnFrameAvailableListener {
    private static final String TAG = "GlesUpscalerBridge";

    // --- PASS 1: OES to 2D Texture ---
    // Vertex shader pour la première passe (décodage matériel vers FBO)
    private static final String BLIT_VERTEX_SHADER =
            "#version 310 es\n" +
            "uniform mat4 uSTMatrix;\n" +
            "in vec4 aPosition;\n" +
            "in vec4 aTexCoord;\n" +
            "out vec2 vTexCoord;\n" +
            "void main() {\n" +
            "  gl_Position = aPosition;\n" +
            "  vTexCoord = (uSTMatrix * aTexCoord).xy;\n" +
            "}\n";

    // Fragment shader pour la première passe
    private static final String BLIT_FRAGMENT_SHADER =
            "#version 310 es\n" +
            "#extension GL_OES_EGL_image_external_essl3 : require\n" +
            "precision mediump float;\n" +
            "in vec2 vTexCoord;\n" +
            "uniform samplerExternalOES sTexture;\n" +
            "out vec4 fragColor;\n" +
            "void main() {\n" +
            "  fragColor = texture(sTexture, vTexCoord);\n" +
            "}\n";

    // --- PASS 2: 2D Texture to Screen (Upscaler) ---
    private static final String POST_VERTEX_SHADER =
            "#version 310 es\n" +
            "in vec4 aPosition;\n" +
            "in vec4 aTexCoord;\n" +
            "out vec2 v_texCoord;\n" +
            "void main() {\n" +
            "  gl_Position = aPosition;\n" +
            "  v_texCoord = aTexCoord.xy;\n" +
            "}\n";

    private final Context context;

    // EGL State
    private EGLDisplay eglDisplay = EGL14.EGL_NO_DISPLAY;
    private EGLContext eglContext = EGL14.EGL_NO_CONTEXT;
    private EGLSurface eglSurface = EGL14.EGL_NO_SURFACE;
    
    // Shader Programs
    private int blitProgram = 0;
    private int postProgram = 0;
    
    // Textures and Framebuffers
    private int oesTextureId;
    private int fboTextureId;
    private int fboId;
    
    // Surface management
    private SurfaceTexture surfaceTexture;
    private Surface decoderSurface;
    private final float[] transformMatrix = new float[16];

    // Handles for Pass 1 (Blit)
    private int blit_uSTMatrixHandle;
    private int blit_aPositionHandle;
    private int blit_aTexCoordHandle;
    private int blit_sTextureHandle;

    // Handles for Pass 2 (Custom Shader)
    private int post_aPositionHandle;
    private int post_aTexCoordHandle;
    private int post_sTextureHandle;
    private int post_uSourceSizeHandle;
    private int post_uOutputSizeHandle;
    
    // Full-screen quad vertices
    private static final float[] VERTICES = {
            -1.0f, -1.0f, 0.0f, 0.0f, // Bottom Left
             1.0f, -1.0f, 1.0f, 0.0f, // Bottom Right
            -1.0f,  1.0f, 0.0f, 1.0f, // Top Left
             1.0f,  1.0f, 1.0f, 1.0f, // Top Right
    };
    private FloatBuffer vertexBuffer;
    
    // Concurrency and synchronization
    private final Object frameAvailableLock = new Object();
    private boolean frameAvailable = false;
    
    // Resolution information
    private int streamWidth, streamHeight;
    private int displayWidth, displayHeight;

    public GlesUpscalerBridge(Context context) {
        this.context = context;
        
        vertexBuffer = ByteBuffer.allocateDirect(VERTICES.length * 4)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer();
        vertexBuffer.put(VERTICES).position(0);
    }

    private String loadAsset(String fileName) {
        StringBuilder sb = new StringBuilder();
        try (InputStream is = context.getAssets().open(fileName);
             BufferedReader reader = new BufferedReader(new InputStreamReader(is))) {
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line).append("\n");
            }
        } catch (IOException e) {
            LimeLog.severe(TAG + ": Failed to load asset: " + fileName + " - " + e.getMessage());
            return null;
        }
        return sb.toString();
    }

    public void initialize(Surface targetSurface, int streamWidth, int streamHeight) {
        this.streamWidth = streamWidth;
        this.streamHeight = streamHeight;

        // Initialize EGL Display
        eglDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY);
        int[] version = new int[2];
        EGL14.eglInitialize(eglDisplay, version, 0, version, 1);

        int[] configAttribs = {
                EGL14.EGL_RENDERABLE_TYPE, EGLExt.EGL_OPENGL_ES3_BIT_KHR,
                EGL14.EGL_RED_SIZE, 8, EGL14.EGL_GREEN_SIZE, 8, EGL14.EGL_BLUE_SIZE, 8,
                EGL14.EGL_NONE
        };
        EGLConfig[] configs = new EGLConfig[1];
        int[] numConfigs = new int[1];
        EGL14.eglChooseConfig(eglDisplay, configAttribs, 0, configs, 0, 1, numConfigs, 0);

        int[] contextAttribs = { EGL14.EGL_CONTEXT_CLIENT_VERSION, 3, EGL14.EGL_NONE };
        eglContext = EGL14.eglCreateContext(eglDisplay, configs[0], EGL14.EGL_NO_CONTEXT, contextAttribs, 0);
        
        eglSurface = EGL14.eglCreateWindowSurface(eglDisplay, configs[0], targetSurface, new int[]{EGL14.EGL_NONE}, 0);
        
        if (eglSurface == EGL14.EGL_NO_SURFACE) {
            throw new IllegalStateException("eglCreateWindowSurface failed");
        }
        
        if (!EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)) {
            throw new IllegalStateException("eglMakeCurrent failed");
        }

        EGL14.eglSwapInterval(eglDisplay, 0);

        // Récupère la taille de l'écran cible (Output resolution)
        int[] queryW = new int[1], queryH = new int[1];
        EGL14.eglQuerySurface(eglDisplay, eglSurface, EGL14.EGL_WIDTH, queryW, 0);
        EGL14.eglQuerySurface(eglDisplay, eglSurface, EGL14.EGL_HEIGHT, queryH, 0);
        this.displayWidth = queryW[0];
        this.displayHeight = queryH[0];

        // 1. Setup Blit Program
        blitProgram = createProgram(BLIT_VERTEX_SHADER, BLIT_FRAGMENT_SHADER);
        blit_uSTMatrixHandle = GLES31.glGetUniformLocation(blitProgram, "uSTMatrix");
        blit_aPositionHandle = GLES31.glGetAttribLocation(blitProgram, "aPosition");
        blit_aTexCoordHandle = GLES31.glGetAttribLocation(blitProgram, "aTexCoord");
        blit_sTextureHandle = GLES31.glGetUniformLocation(blitProgram, "sTexture");

        // 2. Setup Post-Process Program (Notre nouveau shader)
        setupPostProgram();

        // 3. Generate Textures
        int[] textures = new int[2];
        GLES31.glGenTextures(2, textures, 0);
        oesTextureId = textures[0];
        fboTextureId = textures[1];

        // Configure OES Texture
        // Changé de GL_LINEAR à GL_NEAREST à votre demande. Idéal pour conserver la valeur exacte 
        // des pixels bruts lors du transfert (blit) de la texture OES vers le FBO de même taille.
        GLES31.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, oesTextureId);
        GLES31.glTexParameterf(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES31.GL_TEXTURE_MIN_FILTER, GLES31.GL_NEAREST);
        GLES31.glTexParameterf(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES31.GL_TEXTURE_MAG_FILTER, GLES31.GL_NEAREST);
        GLES31.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES31.GL_TEXTURE_WRAP_S, GLES31.GL_CLAMP_TO_EDGE);
        GLES31.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES31.GL_TEXTURE_WRAP_T, GLES31.GL_CLAMP_TO_EDGE);

        // Configure 2D Texture (FBO)
        // ATTENTION: Celle-ci doit rester en GL_LINEAR car le shader d'upscale (Pass 2) s'appuie 
        // sur l'interpolation bilinéaire matérielle pour l'upscale natif des UV (chroma) et le SGSR.
        GLES31.glBindTexture(GLES31.GL_TEXTURE_2D, fboTextureId);
        GLES31.glTexImage2D(GLES31.GL_TEXTURE_2D, 0, GLES31.GL_RGBA, streamWidth, streamHeight, 0, GLES31.GL_RGBA, GLES31.GL_UNSIGNED_BYTE, null);
        GLES31.glTexParameteri(GLES31.GL_TEXTURE_2D, GLES31.GL_TEXTURE_MIN_FILTER, GLES31.GL_LINEAR);
        GLES31.glTexParameteri(GLES31.GL_TEXTURE_2D, GLES31.GL_TEXTURE_MAG_FILTER, GLES31.GL_LINEAR);
        GLES31.glTexParameteri(GLES31.GL_TEXTURE_2D, GLES31.GL_TEXTURE_WRAP_S, GLES31.GL_CLAMP_TO_EDGE);
        GLES31.glTexParameteri(GLES31.GL_TEXTURE_2D, GLES31.GL_TEXTURE_WRAP_T, GLES31.GL_CLAMP_TO_EDGE);

        // 4. Setup FBO
        int[] fbos = new int[1];
        GLES31.glGenFramebuffers(1, fbos, 0);
        fboId = fbos[0];
        GLES31.glBindFramebuffer(GLES31.GL_FRAMEBUFFER, fboId);
        GLES31.glFramebufferTexture2D(GLES31.GL_FRAMEBUFFER, GLES31.GL_COLOR_ATTACHMENT0, GLES31.GL_TEXTURE_2D, fboTextureId, 0);
        
        int status = GLES31.glCheckFramebufferStatus(GLES31.GL_FRAMEBUFFER);
        if (status != GLES31.GL_FRAMEBUFFER_COMPLETE) {
            LimeLog.severe(TAG + ": Framebuffer incomplete: " + status);
        }
        GLES31.glBindFramebuffer(GLES31.GL_FRAMEBUFFER, 0);

        surfaceTexture = new SurfaceTexture(oesTextureId);
        surfaceTexture.setOnFrameAvailableListener(this);
        decoderSurface = new Surface(surfaceTexture);
        
        // IMPORTANT: Libérer le contexte EGL de ce thread d'initialisation.
        // Si on ne le fait pas, renderFrame() plantera avec "EGL_BAD_ACCESS" (error 3002) 
        // car il essayera d'utiliser ce contexte sur un autre thread (le thread de rendu).
        EGL14.eglMakeCurrent(eglDisplay, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT);

        LimeLog.info(TAG + ": GlesUpscalerBridge initialized: Stream=" + streamWidth + "x" + streamHeight + " Display=" + displayWidth + "x" + displayHeight);
    }

    private void setupPostProgram() {
        String postSource = loadAsset("fsr_easu.frag");
        
        if (postSource == null || postSource.trim().isEmpty()) {
            LimeLog.severe(TAG + ": Impossible de charger fsr_easu.frag");
            return;
        }

        postProgram = createProgram(POST_VERTEX_SHADER, postSource);
        post_aPositionHandle = GLES31.glGetAttribLocation(postProgram, "aPosition");
        post_aTexCoordHandle = GLES31.glGetAttribLocation(postProgram, "aTexCoord");
        post_sTextureHandle = GLES31.glGetUniformLocation(postProgram, "u_source");
        post_uSourceSizeHandle = GLES31.glGetUniformLocation(postProgram, "u_sourceSize");
        post_uOutputSizeHandle = GLES31.glGetUniformLocation(postProgram, "u_outputSize");
    }

    public Surface getDecoderSurface() {
        return decoderSurface;
    }

    @Override
    public void onFrameAvailable(SurfaceTexture surfaceTexture) {
        synchronized (frameAvailableLock) {
            frameAvailable = true;
            frameAvailableLock.notifyAll();
        }
    }

    public void renderFrame(long presentationTimeNanos) {
        if (eglDisplay == EGL14.EGL_NO_DISPLAY) return;

        synchronized (frameAvailableLock) {
            long startTime = System.currentTimeMillis();
            while (!frameAvailable) {
                try {
                    frameAvailableLock.wait(50);
                    if (System.currentTimeMillis() - startTime > 50) break;
                } catch (InterruptedException e) { return; }
            }
            frameAvailable = false;
        }

        if (!EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)) {
            LimeLog.severe(TAG + ": eglMakeCurrent failed in renderFrame");
            return;
        }
        
        surfaceTexture.updateTexImage();
        surfaceTexture.getTransformMatrix(transformMatrix);

        // PASS 1: Blit OES vers FBO
        GLES31.glBindFramebuffer(GLES31.GL_FRAMEBUFFER, fboId);
        GLES31.glViewport(0, 0, streamWidth, streamHeight);
        GLES31.glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        GLES31.glClear(GLES31.GL_COLOR_BUFFER_BIT);

        GLES31.glUseProgram(blitProgram);
        
        if (blit_uSTMatrixHandle >= 0) GLES31.glUniformMatrix4fv(blit_uSTMatrixHandle, 1, false, transformMatrix, 0);
        if (blit_sTextureHandle >= 0) GLES31.glUniform1i(blit_sTextureHandle, 0);

        GLES31.glActiveTexture(GLES31.GL_TEXTURE0);
        GLES31.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, oesTextureId);
        
        drawQuad(blit_aPositionHandle, blit_aTexCoordHandle);

        // PASS 2: Shader d'upscaling FSR EASU vers l'écran
        GLES31.glBindFramebuffer(GLES31.GL_FRAMEBUFFER, 0);
        GLES31.glViewport(0, 0, displayWidth, displayHeight);
        GLES31.glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        GLES31.glClear(GLES31.GL_COLOR_BUFFER_BIT);

        GLES31.glUseProgram(postProgram);

        if (post_sTextureHandle >= 0) GLES31.glUniform1i(post_sTextureHandle, 0);
        if (post_uSourceSizeHandle >= 0) GLES31.glUniform2f(post_uSourceSizeHandle, (float)streamWidth, (float)streamHeight);
        if (post_uOutputSizeHandle >= 0) GLES31.glUniform2f(post_uOutputSizeHandle, (float)displayWidth, (float)displayHeight);

        GLES31.glActiveTexture(GLES31.GL_TEXTURE0);
        GLES31.glBindTexture(GLES31.GL_TEXTURE_2D, fboTextureId);

        drawQuad(post_aPositionHandle, post_aTexCoordHandle);
        
        EGLExt.eglPresentationTimeANDROID(eglDisplay, eglSurface, presentationTimeNanos);
        EGL14.eglSwapBuffers(eglDisplay, eglSurface);
    }

    private void drawQuad(int posHandle, int texHandle) {
        if (posHandle >= 0) {
            GLES31.glEnableVertexAttribArray(posHandle);
            vertexBuffer.position(0);
            GLES31.glVertexAttribPointer(posHandle, 2, GLES31.GL_FLOAT, false, 16, vertexBuffer);
        }
        if (texHandle >= 0) {
            GLES31.glEnableVertexAttribArray(texHandle);
            vertexBuffer.position(2);
            GLES31.glVertexAttribPointer(texHandle, 2, GLES31.GL_FLOAT, false, 16, vertexBuffer);
        }

        GLES31.glDrawArrays(GLES31.GL_TRIANGLE_STRIP, 0, 4);

        if (posHandle >= 0) GLES31.glDisableVertexAttribArray(posHandle);
        if (texHandle >= 0) GLES31.glDisableVertexAttribArray(texHandle);
    }

    private int createProgram(String vertexSource, String fragmentSource) {
        int vertexShader = loadShader(GLES31.GL_VERTEX_SHADER, vertexSource);
        if (vertexShader == 0) return 0;
        int fragmentShader = loadShader(GLES31.GL_FRAGMENT_SHADER, fragmentSource);
        if (fragmentShader == 0) return 0;
        
        int program = GLES31.glCreateProgram();
        if (program != 0) {
            GLES31.glAttachShader(program, vertexShader);
            GLES31.glAttachShader(program, fragmentShader);
            GLES31.glLinkProgram(program);
            
            int[] linkStatus = new int[1];
            GLES31.glGetProgramiv(program, GLES31.GL_LINK_STATUS, linkStatus, 0);
            if (linkStatus[0] != GLES31.GL_TRUE) {
                LimeLog.severe(TAG + ": Could not link program: " + GLES31.glGetProgramInfoLog(program));
                GLES31.glDeleteProgram(program);
                program = 0;
            }
        }
        return program;
    }

    private int loadShader(int type, String shaderCode) {
        int shader = GLES31.glCreateShader(type);
        GLES31.glShaderSource(shader, shaderCode);
        GLES31.glCompileShader(shader);
        
        int[] compiled = new int[1];
        GLES31.glGetShaderiv(shader, GLES31.GL_COMPILE_STATUS, compiled, 0);
        if (compiled[0] == 0) {
            LimeLog.severe(TAG + ": Could not compile shader " + type + ": " + GLES31.glGetShaderInfoLog(shader));
            GLES31.glDeleteShader(shader);
            shader = 0;
        }
        return shader;
    }

    public void release() {
        if (eglDisplay != EGL14.EGL_NO_DISPLAY) {
            EGL14.eglMakeCurrent(eglDisplay, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT);
            if (eglSurface != EGL14.EGL_NO_SURFACE) {
                EGL14.eglDestroySurface(eglDisplay, eglSurface);
            }
            EGL14.eglDestroyContext(eglDisplay, eglContext);
            EGL14.eglReleaseThread();
            EGL14.eglTerminate(eglDisplay);
        }
        if (surfaceTexture != null) {
            surfaceTexture.release();
        }
        if (decoderSurface != null) {
            decoderSurface.release();
        }
        eglDisplay = EGL14.EGL_NO_DISPLAY;
        eglContext = EGL14.EGL_NO_CONTEXT;
        eglSurface = EGL14.EGL_NO_SURFACE;
    }
}
