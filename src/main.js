import * as GaussianSplats3D from '/node_modules/gaussian-splat-renderer-for-lam/build/gaussian-splat-renderer-for-lam.module.js';
window.GaussianSplats3D = GaussianSplats3D;

const currentHost = window.location.hostname;

const CONFIG = {
    // TFLite model path
    modelPath: 'models/lam_audio2exp_optimized_float32.tflite',
    uploadServer: `http://${currentHost}:3001`,
    sampleRate: 16000,
    fps: 30,
    chunkSize: 16000,
    bufferPrefill: 65,
    syncOffset: 0.30
};

const BLENDSHAPE_NAMES = ["browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight", "tongueOut"];

// --- STATE ---
const state = {
    worker: null,
    audioContext: null,
    renderer: null,
    animationBuffer: [],
    avatarPath: '/asset/arkit/p2-1.zip',

    // Playback State
    audioPlaying: false,
    isLiveStream: false,
    streamStartTime: 0,
    nextStartTime: 0,

    // Processing State
    pcmAccumulator: new Float32Array(0),

    // Buffering Logic
    audioQueue: [],
    isPrerolling: false,
    queuedDuration: 0,
    PREROLL_DURATION_SEC: 1.0
};

function log(msg, type='info') {
    const colors = { error: 'text-red-400', success: 'text-green-400', warn: 'text-yellow-400', info: 'text-slate-400' };
    const div = document.createElement('div');
    div.className = `${colors[type]} mb-1 break-words`;
    div.innerText = `> ${msg}`;
    const container = document.getElementById('logContainer');
    if (container) {
        container.appendChild(div);
        container.scrollTop = container.scrollHeight;
    }
    if(type === 'error') console.error(msg);
}

function updateDebug(jawVal, bufferFrames, latency) {
    if(jawVal !== undefined) {
        const bar = document.getElementById('audioVisBar');
        if(bar) bar.style.width = `${Math.min(jawVal * 200, 100)}%`;
    }
    if(bufferFrames !== undefined) {
        const seconds = (bufferFrames / CONFIG.fps).toFixed(2);
        const el = document.getElementById('debugBuffer');
        if(el) el.innerText = `${seconds}s`;
    }
    if(latency !== undefined) {
        const el = document.getElementById('debugLatency');
        if(el) el.innerText = `${latency.toFixed(0)}ms`;
    }
    const stateEl = document.getElementById('debugState');
    if(stateEl) stateEl.innerText = state.audioPlaying ? "PLAYING" : "IDLE";
}

// --- WORKER SCRIPT (TFLITE) ---
const workerScript = `
    importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-core');
    importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-cpu');
    importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite/dist/tf-tflite.min.js');

    tflite.setWasmPath('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite/dist/');

    let tfliteModel = null;
    let inputNames = { audio: null, id: null };

    let prevAudio = new Float32Array(0);
    let prevExpressions = [];
    let blinkTimer = 0;
    let nextBlinkTime = 100;

    const SAMPLE_RATE = 16000;
    const FPS = 30;
    const HISTORY_SIZE = 10;
    const SG_KERNEL = [-0.0857, 0.3429, 0.4857, 0.3429, -0.0857];
    const SILENCE_THRESHOLD = 0.001;
    const MIN_SILENCE_DURATION = 7;
    const BLEND_WINDOW = 3;

    // Full Constants
    const ALL_NAMES = ["browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight", "tongueOut"];
    const ARKitLeftRightPair = [["jawLeft","jawRight"],["mouthLeft","mouthRight"],["mouthSmileLeft","mouthSmileRight"],["mouthFrownLeft","mouthFrownRight"],["mouthDimpleLeft","mouthDimpleRight"],["mouthStretchLeft","mouthStretchRight"],["mouthPressLeft","mouthPressRight"],["mouthLowerDownLeft","mouthLowerDownRight"],["mouthUpperUpLeft","mouthUpperUpRight"],["cheekSquintLeft","cheekSquintRight"],["noseSneerLeft","noseSneerRight"],["browDownLeft","browDownRight"],["browOuterUpLeft","browOuterUpRight"],["eyeBlinkLeft","eyeBlinkRight"],["eyeLookDownLeft","eyeLookDownRight"],["eyeLookInLeft","eyeLookInRight"],["eyeLookOutLeft","eyeLookOutRight"],["eyeLookUpLeft","eyeLookUpRight"],["eyeSquintLeft","eyeSquintRight"],["eyeWideLeft","eyeWideRight"]];
    const MOUTH_SHAPE_NAMES = ["mouthDimpleLeft","mouthDimpleRight","mouthFrownLeft","mouthFrownRight","mouthFunnel","mouthLeft","mouthLowerDownLeft","mouthLowerDownRight","mouthPressLeft","mouthPressRight","mouthPucker","mouthRight","mouthRollLower","mouthRollUpper","mouthShrugLower","mouthShrugUpper","mouthSmileLeft","mouthSmileRight","mouthStretchLeft","mouthStretchRight","mouthUpperUpLeft","mouthUpperUpRight","jawForward","jawLeft","jawOpen","jawRight","noseSneerLeft","noseSneerRight","cheekPuff"];
    const BLINK_PATTERNS = [[0.365,0.950,0.956,0.917,0.367,0.119,0.025],[0.235,0.910,0.945,0.778,0.191,0.235,0.089]];
    const MOUTH_INDICES = MOUTH_SHAPE_NAMES.map(n => ALL_NAMES.indexOf(n)).filter(i => i !== -1);
    const SYMMETRIC_INDICES = ARKitLeftRightPair.map(pair => [ALL_NAMES.indexOf(pair[0]), ALL_NAMES.indexOf(pair[1])]);

    function calculateRMS(audio, frameCount) {
        const hop = Math.floor(SAMPLE_RATE / FPS);
        const rmsArray = new Float32Array(frameCount).fill(0);
        for(let i=0; i<frameCount; i++) {
            const start = i * hop;
            const end = Math.min(start + hop, audio.length);
            let sum = 0;
            if(start < end) { for(let j=start; j<end; j++) sum += audio[j] * audio[j]; rmsArray[i] = Math.sqrt(sum / (end - start)); }
        }
        return rmsArray;
    }
    function smoothMouthMovements(frames, volume) {
        const regions = []; let start = -1; let length = 0;
        for(let i=0; i<volume.length; i++) {
            if(volume[i] < SILENCE_THRESHOLD) { if(start === -1) start = i; length++; }
            else { if(start !== -1 && length >= MIN_SILENCE_DURATION) regions.push({start, end: i-1}); start = -1; length = 0; }
        }
        if(start !== -1 && length >= MIN_SILENCE_DURATION) regions.push({start, end: volume.length-1});

        regions.forEach(r => {
            for(let i=r.start; i<=r.end; i++) MOUTH_INDICES.forEach(idx => frames[i][idx] *= 0.1);
            const blendLenStart = Math.min(BLEND_WINDOW, r.start);
            if(blendLenStart > 0 && r.start > 0) {
                const preFrame = frames[r.start - 1];
                for(let i=0; i<blendLenStart; i++) {
                    const weight = (i + 1) / (blendLenStart + 1); const idx = r.start + i;
                    if(idx < frames.length) for(let j=0; j<52; j++) frames[idx][j] = preFrame[j] * (1 - weight) + frames[idx][j] * weight;
                }
            }
            const blendLenEnd = Math.min(BLEND_WINDOW, frames.length - 1 - r.end);
            if(blendLenEnd > 0 && r.end < frames.length - 1) {
                const postFrame = frames[r.end + 1];
                for(let i=0; i<blendLenEnd; i++) {
                    const weight = (i + 1) / (blendLenEnd + 1); const idx = r.end - i;
                    if(idx >= 0) for(let j=0; j<52; j++) frames[idx][j] = postFrame[j] * (1 - weight) + frames[idx][j] * weight;
                }
            }
        });
        return frames;
    }
    function applyFrameBlending(newFrames, lastProcessedFrame) {
        if (!lastProcessedFrame) return newFrames;
        const blendWindow = 5; const actualLen = Math.min(blendWindow, newFrames.length);
        for (let i = 0; i < actualLen; i++) {
            const weight = (i + 1) / (actualLen + 1);
            for (let j = 0; j < 52; j++) newFrames[i][j] = lastProcessedFrame[j] * (1 - weight) + newFrames[i][j] * weight;
        }
        return newFrames;
    }
    function applySGSmoothing(newFrames, history) {
        const result = []; const halfWin = 2;
        for (let i = 0; i < newFrames.length; i++) {
            const smoothedFrame = new Float32Array(52).fill(0);
            for (let k = -halfWin; k <= halfWin; k++) {
                const idx = i + k; let srcFrame = null;
                if (idx < 0) { const histIdx = history.length + idx; srcFrame = (histIdx >= 0) ? history[histIdx] : newFrames[0]; }
                else if (idx >= newFrames.length) { srcFrame = newFrames[newFrames.length - 1]; }
                else { srcFrame = newFrames[idx]; }
                const weight = SG_KERNEL[k + halfWin];
                for (let j = 0; j < 52; j++) smoothedFrame[j] += srcFrame[j] * weight;
            }
            for(let j=0; j<52; j++) smoothedFrame[j] = Math.max(0, Math.min(1, smoothedFrame[j]));
            result.push(Array.from(smoothedFrame));
        }
        return result;
    }
    function applySymmetry(frames) {
        for(let i=0; i<frames.length; i++) {
            SYMMETRIC_INDICES.forEach(pair => {
                const idxL = pair[0]; const idxR = pair[1];
                if(idxL !== -1 && idxR !== -1) { const avg = (frames[i][idxL] + frames[i][idxR]) / 2; frames[i][idxL] = avg; frames[i][idxR] = avg; }
            });
        }
        return frames;
    }
    function applyBlinks(frames) {
        for (let i = 0; i < frames.length; i++) {
            blinkTimer++;
            if (blinkTimer >= nextBlinkTime) {
                const pattern = BLINK_PATTERNS[Math.floor(Math.random() * BLINK_PATTERNS.length)];
                for(let b=0; b<pattern.length; b++) {
                    if (i + b < frames.length) { frames[i+b][8] = Math.max(frames[i+b][8], pattern[b]); frames[i+b][9] = Math.max(frames[i+b][9], pattern[b]); }
                }
                blinkTimer = 0; nextBlinkTime = 60 + Math.random() * 90;
            }
        }
        return frames;
    }

    self.onmessage = async (e) => {
        const { type, payload } = e.data;

        if (type === 'INIT') {
            try {
                tfliteModel = await tflite.loadTFLiteModel(payload.modelPath);

                tfliteModel.inputs.forEach(input => {
                    const shape = input.shape;
                    if (shape[1] === 16000) inputNames.audio = input.name;
                    else if (shape[1] === 12) inputNames.id = input.name;
                });

                if (!inputNames.audio || !inputNames.id) throw new Error("Model input shape mismatch");

                tf.tidy(() => {
                    const dummyAudio = tf.zeros([1, 16000], 'float32');
                    const dummyId = tf.zeros([1, 12], 'float32');
                    const inputs = {};
                    inputs[inputNames.audio] = dummyAudio;
                    inputs[inputNames.id] = dummyId;
                    tfliteModel.predict(inputs);
                });

                self.postMessage({ type: 'INIT_DONE' });
            } catch (err) { self.postMessage({ type: 'ERROR', payload: err.message }); }
        }

        if (type === 'RESET') { prevAudio = new Float32Array(0); prevExpressions = []; blinkTimer = 0; }

        if (type === 'INFER') {
            if (!tfliteModel) return;
            const chunk = payload;
            const tStart = performance.now();

            const WINDOW_SAMPLES = 34133;
            const combined = new Float32Array(prevAudio.length + chunk.length);
            combined.set(prevAudio); combined.set(chunk, prevAudio.length);

            let input = (combined.length < WINDOW_SAMPLES) ? combined : combined.slice(combined.length - WINDOW_SAMPLES);
            if (combined.length < WINDOW_SAMPLES) {
                let tmp = new Float32Array(WINDOW_SAMPLES);
                tmp.set(combined, WINDOW_SAMPLES-combined.length);
                input = tmp;
            }
            prevAudio = input;

            try {
                const rawFrames = tf.tidy(() => {
                    const modelInputLen = 16000;
                    let modelInput = input;
                    if (input.length > modelInputLen) {
                        modelInput = input.slice(input.length - modelInputLen);
                    } else if (input.length < modelInputLen) {
                         let tmp = new Float32Array(modelInputLen);
                         tmp.set(input, modelInputLen - input.length);
                         modelInput = tmp;
                    }

                    const audioTensor = tf.tensor(modelInput, [1, 16000], 'float32');
                    const idTensor = tf.zeros([1, 12], 'float32');

                    const inputs = {};
                    inputs[inputNames.audio] = audioTensor;
                    inputs[inputNames.id] = idTensor;

                    const outputTensor = tfliteModel.predict(inputs);
                    const flatData = outputTensor.dataSync();

                    const totalModelFrames = flatData.length / 52;
                    const newFramesCount = Math.ceil(chunk.length / SAMPLE_RATE * FPS);
                    const startFrame = Math.max(0, totalModelFrames - newFramesCount);

                    let extracted = [];
                    for(let i=startFrame; i<totalModelFrames; i++) {
                        extracted.push(Array.from(flatData.subarray(i*52, (i+1)*52)));
                    }
                    return extracted;
                });

                let processedFrames = rawFrames;
                if (processedFrames.length > 0) {
                    processedFrames = smoothMouthMovements(processedFrames, calculateRMS(chunk, processedFrames.length));
                    processedFrames = applyFrameBlending(processedFrames, prevExpressions.length > 0 ? prevExpressions[prevExpressions.length - 1] : null);
                    processedFrames = applySGSmoothing(processedFrames, prevExpressions);
                    processedFrames = applySymmetry(processedFrames);
                    processedFrames = applyBlinks(processedFrames);
                    prevExpressions = prevExpressions.concat(processedFrames).slice(-HISTORY_SIZE);
                }

                self.postMessage({ type: 'INFER_RESULT', payload: { frames: processedFrames, latency: performance.now() - tStart } });
            } catch(err) { self.postMessage({ type: 'ERROR', payload: err.message }); }
        }
    };
`;

function initWorker() {
    const blob = new Blob([workerScript], { type: 'application/javascript' });
    state.worker = new Worker(URL.createObjectURL(blob));
    state.worker.onmessage = (e) => {
        const { type, payload } = e.data;
        if (type === 'INIT_DONE') {
            document.getElementById('modelStatus').innerHTML = '<span class="text-green-500">TFLite Ready</span>';
            log("Worker Ready (TFLite)", 'success');
        } else if (type === 'INFER_RESULT') {
            payload.frames.forEach(f => state.animationBuffer.push(f));
            updateDebug(undefined, state.animationBuffer.length, payload.latency);
        } else if (type === 'ERROR') log(payload, 'error');
    };
    const absoluteModelPath = new URL(CONFIG.modelPath, window.location.href).href;
    state.worker.postMessage({ type: 'INIT', payload: { modelPath: absoluteModelPath } });
}

// --- RENDERER CALLBACK ---
function getExpressionCallback() {
    if (state.audioPlaying) {
        let currentTime = 0;
        if (state.isLiveStream) {
             currentTime = state.audioContext.currentTime - state.streamStartTime;
        } else {
             const audioEl = window.currentAudioElement;
             if (audioEl) currentTime = audioEl.currentTime;
        }

        const targetFrameIdx = Math.floor((currentTime + CONFIG.syncOffset) * CONFIG.fps);

        if (targetFrameIdx >= 0) {
            if (targetFrameIdx < state.animationBuffer.length) {
                const weights = state.animationBuffer[targetFrameIdx];
                const bs = {};
                BLENDSHAPE_NAMES.forEach((name, i) => bs[name] = weights[i]);
                updateDebug(weights[24], state.animationBuffer.length - targetFrameIdx);
                return bs;
            } else if (state.animationBuffer.length > 0) {
                const weights = state.animationBuffer[state.animationBuffer.length - 1];
                const bs = {};
                BLENDSHAPE_NAMES.forEach((name, i) => bs[name] = weights[i]);
                return bs;
            }
        }
    }
    return {};
}

async function initRenderer() {
    if (state.renderer) return;
    const container = document.getElementById('rendererContainer');
    document.getElementById('rendererPlaceholder').classList.add('hidden');
    document.getElementById('rendererLoading').classList.remove('hidden');
    try {
        state.renderer = await window.GaussianSplats3D.GaussianSplatRenderer.getInstance(
            container, state.avatarPath,
            { getChatState: () => state.audioPlaying ? "Responding" : "Idle", getExpressionData: getExpressionCallback, backgroundColor: "0x000000" }
        );
        document.getElementById('rendererLoading').classList.add('hidden');
        log("Avatar Loaded", 'success');
    } catch (e) { log("Avatar Error: " + e.message, 'error'); }
}

async function reloadRenderer() {
    if (state.renderer) { try { await state.renderer.dispose(); } catch(e){} state.renderer = null; }
    await initRenderer();
}

// --- ADAPTIVE STREAMING LOGIC ---
function queueAudioChunk(audioBuffer) {
    const source = state.audioContext.createBufferSource();
    source.buffer = audioBuffer;

    // Track duration and store source
    const chunkDuration = audioBuffer.duration;
    const chunkInfo = { source, duration: chunkDuration };
    state.audioQueue.push(chunkInfo);
    state.queuedDuration += chunkDuration;

    // 1. Check if we need to start prerolling (Cold Start)
    if (!state.audioPlaying && !state.isPrerolling) {
        state.isPrerolling = true;
        log(`Buffering... (Target: ${state.PREROLL_DURATION_SEC}s)`, 'warn');
    }

    // 2. If Prerolling, check if threshold is met
    if (state.isPrerolling) {
        log(`Buffer: ${state.queuedDuration.toFixed(2)}s / ${state.PREROLL_DURATION_SEC}s`);
        if (state.queuedDuration >= state.PREROLL_DURATION_SEC) {
            startPlaybackQueue();
        }
    } else {
        // 3. Already playing? Schedule immediately.
        scheduleNextChunk();
    }
}

function startPlaybackQueue() {
    state.isPrerolling = false;
    state.audioPlaying = true;

    // Set start time slightly in the future (0.1s) for browser stability
    state.nextStartTime = state.audioContext.currentTime + 0.1;
    state.streamStartTime = state.nextStartTime;

    log("Preroll Complete. Playing.", 'success');

    // Flush the queue
    while(state.audioQueue.length > 0) {
        scheduleNextChunk();
    }
}

function scheduleNextChunk() {
    if (state.audioQueue.length === 0) return;

    const info = state.audioQueue.shift();

    // 1. Create a GainNode for "De-clicking" envelope
    const gainNode = state.audioContext.createGain();

    // 2. Connect Source -> Gain -> Destination
    info.source.connect(gainNode);
    gainNode.connect(state.audioContext.destination);

    // 3. Calculate timing
    let scheduleTime = Math.max(state.audioContext.currentTime, state.nextStartTime);

    // 4. Apply Micro-Fades (3ms) to prevent popping
    const FADE_DURATION = 0.003;

    // Reset gain to 0
    gainNode.gain.setValueAtTime(0, scheduleTime);
    // Fade In
    gainNode.gain.linearRampToValueAtTime(1, scheduleTime + FADE_DURATION);
    // Fade Out
    gainNode.gain.setValueAtTime(1, scheduleTime + info.duration - FADE_DURATION);
    gainNode.gain.linearRampToValueAtTime(0, scheduleTime + info.duration);

    // 5. Start Playback
    info.source.start(scheduleTime);

    state.nextStartTime = scheduleTime + info.duration;
    state.queuedDuration = Math.max(0, state.queuedDuration - info.duration);

    info.source.onended = () => {
        info.source.disconnect();
        gainNode.disconnect();

        if (state.audioQueue.length === 0 && state.audioContext.currentTime >= state.nextStartTime - 0.1) {
             state.audioPlaying = false;
             state.queuedDuration = 0;
             log("Stream Finished", 'info');
             document.getElementById('btnStream').disabled = false;
             document.getElementById('btnStream').innerText = "Start Live Stream";

             feedWorker(null, true);
             updateDebug(0, 0, 0);
        }
    };
}

// --- MODEL FEEDER (WITH PADDING) ---
function feedWorker(newPcmData, isFlush = false) {
    // 1. Append data
    if (newPcmData) {
        const totalLen = state.pcmAccumulator.length + newPcmData.length;
        const temp = new Float32Array(totalLen);
        temp.set(state.pcmAccumulator);
        temp.set(newPcmData, state.pcmAccumulator.length);
        state.pcmAccumulator = temp;
    }

    // 2. Process strictly in 16k chunks
    const CHUNK = CONFIG.chunkSize;
    while (state.pcmAccumulator.length >= CHUNK) {
        const slice = state.pcmAccumulator.slice(0, CHUNK);
        state.worker.postMessage({ type: 'INFER', payload: slice });
        state.pcmAccumulator = state.pcmAccumulator.slice(CHUNK);
    }

    // 3. Flush leftover if requested (WITH PADDING)
    if (isFlush && state.pcmAccumulator.length > 0) {
        log(`Flushing final buffer: ${state.pcmAccumulator.length} samples`, 'info');

        if (state.pcmAccumulator.length < CHUNK) {
            const padded = new Float32Array(CHUNK);
            padded.set(state.pcmAccumulator);
            state.worker.postMessage({ type: 'INFER', payload: padded });
        } else {
             state.worker.postMessage({ type: 'INFER', payload: state.pcmAccumulator });
        }

        state.pcmAccumulator = new Float32Array(0);
    }
}

async function processStream(reader) {
    let leftover = new Uint8Array(0);

    while (true) {
        const { done, value } = await reader.read();

        if (done) {
            if (state.isPrerolling) {
                log(`Stream ended (Total: ${state.queuedDuration.toFixed(2)}s). Forcing playback.`, 'warn');
                startPlaybackQueue();
            }

            feedWorker(null, true);
            break;
        }

        const combined = new Uint8Array(leftover.length + value.length);
        combined.set(leftover);
        combined.set(value, leftover.length);

        try {
            const tempBuffer = combined.slice(0).buffer;
            const audioBuffer = await state.audioContext.decodeAudioData(tempBuffer);

            const pcmData = audioBuffer.getChannelData(0);

            feedWorker(pcmData, false);
            queueAudioChunk(audioBuffer);

            leftover = new Uint8Array(0);

        } catch (e) {
            leftover = combined;
        }
    }
}

document.getElementById('btnStream').addEventListener('click', async () => {
    const text = document.getElementById('ttsInput').value;
    const voiceId = document.getElementById('voiceIdInput').value;
    if (!text) { log("Enter text first.", 'error'); return; }

    const btn = document.getElementById('btnStream');
    btn.disabled = true;
    btn.innerText = "Streaming...";

    state.animationBuffer = [];
    state.pcmAccumulator = new Float32Array(0);
    state.audioQueue = [];
    state.queuedDuration = 0;
    state.nextStartTime = 0;
    state.audioPlaying = false;
    state.isLiveStream = true;
    state.worker.postMessage({ type: 'RESET' });

    if (!state.renderer) await initRenderer();
    if (!state.audioContext) state.audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: CONFIG.sampleRate });
    if (state.audioContext.state === 'suspended') await state.audioContext.resume();

    state.nextStartTime = state.audioContext.currentTime + 0.1;

    try {
        log("Connecting to WaveSpeed Stream...", 'info');
        const response = await fetch(`${CONFIG.uploadServer}/api/tts-stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, voice_id: voiceId })
        });

        if (!response.ok) throw new Error("Stream connection failed");

        const reader = response.body.getReader();
        await processStream(reader);

    } catch (e) {
        log("Stream Error: " + e.message, 'error');
        btn.disabled = false;
        btn.innerText = "Start Live Stream";
    }
});

// --- CLONING & AVATAR LOGIC ---
document.getElementById('btnClone').addEventListener('click', async () => {
    const file = document.getElementById('cloneAudioInput').files[0];
    if (!file) return log("No file selected", 'error');
    const formData = new FormData(); formData.append('audio_sample', file);
    try {
        const res = await fetch(`${CONFIG.uploadServer}/api/clone-voice`, { method: 'POST', body: formData });
        const json = await res.json();
        if(json.success) {
            document.getElementById('voiceIdInput').value = json.voice_id;
            document.getElementById('clonedVoiceId').innerText = json.voice_id;
            document.getElementById('cloneStatus').classList.remove('hidden');
            log("Voice Cloned", 'success');
        }
    } catch(e) { log("Clone Failed", 'error'); }
});

document.getElementById('avatarSelect').addEventListener('change', async (e) => {
    if (e.target.value === 'custom') document.getElementById('avatarInput').click();
    else { state.avatarPath = '/asset/arkit/p2-1.zip'; await reloadRenderer(); }
});

document.getElementById('avatarInput').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    const formData = new FormData(); formData.append('avatar', file);
    const res = await fetch(`${CONFIG.uploadServer}/upload-avatar`, { method: 'POST', body: formData });
    const json = await res.json();
    if(json.success) {
        state.avatarPath = `${CONFIG.uploadServer}${json.path}`;
        document.getElementById('avatarSelect').querySelector('option[value="custom"]').innerText = `Custom: ${file.name}`;
        document.getElementById('avatarSelect').value = 'custom';
        await reloadRenderer();
    }
});

initWorker();
initRenderer();