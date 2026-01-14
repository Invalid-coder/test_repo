import express from 'express';
import multer from 'multer';
import cors from 'cors';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';
import axios from 'axios';
import dotenv from 'dotenv';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 3001;
const WAVESPEED_API_KEY = process.env.WAVESPEED_API_KEY;

if (!WAVESPEED_API_KEY) {
    console.warn("⚠️ WARNING: WAVESPEED_API_KEY is missing in .env file.");
}

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));
app.use('/node_modules', express.static(path.join(__dirname, 'node_modules')));

// Configure Multer
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        const dir = path.join(__dirname, 'public', 'asset', 'arkit', 'custom');
        if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
        cb(null, dir);
    },
    filename: (req, file, cb) => {
        const uniqueName = `${Date.now()}-${file.originalname.replace(/[^a-zA-Z0-9.-]/g, '_')}`;
        cb(null, uniqueName);
    }
});
const upload = multer({ storage });

// --- ENDPOINTS ---

// 1. Upload Avatar
app.post('/upload-avatar', upload.single('avatar'), (req, res) => {
    if (!req.file) return res.status(400).json({ error: 'No file uploaded' });
    res.json({ success: true, path: `/asset/arkit/custom/${req.file.filename}`, filename: req.file.filename });
});

// 2. Clone Voice
async function pollWaveSpeedResult(requestId) {
    const url = `https://api.wavespeed.ai/api/v3/predictions/${requestId}/result`;
    const headers = { "Authorization": `Bearer ${WAVESPEED_API_KEY}` };
    for (let i = 0; i < 30; i++) {
        const response = await axios.get(url, { headers });
        if (response.data.data.status === 'completed') return response.data.data.outputs[0];
        if (response.data.data.status === 'failed') throw new Error(response.data.data.error || 'Task failed');
        await new Promise(resolve => setTimeout(resolve, 1000));
    }
    throw new Error("Polling timed out");
}

app.post('/api/clone-voice', upload.single('audio_sample'), async (req, res) => {
    try {
        if (!req.file) return res.status(400).json({ error: 'No audio sample' });
        const base64Audio = fs.readFileSync(req.file.path).toString('base64');
        const voiceId = `custom_voice_${Date.now()}`;

        const payload = {
            "model": "speech-02-hd",
            "text": "Voice cloning complete.",
            "audio": base64Audio,
            "custom_voice_id": voiceId
        };

        const response = await axios.post("https://api.wavespeed.ai/api/v3/minimax/voice-clone", payload, {
            headers: { "Content-Type": "application/json", "Authorization": `Bearer ${WAVESPEED_API_KEY}` }
        });

        await pollWaveSpeedResult(response.data.data.id);
        res.json({ success: true, voice_id: voiceId });
    } catch (error) {
        console.error("Clone Error:", error.message);
        res.status(500).json({ error: "Cloning failed" });
    }
});

// 3. Streaming TTS Endpoint (With Duplicate Fix)
app.post('/api/tts-stream', async (req, res) => {
    try {
        const { text, voice_id } = req.body;
        console.log(`[Stream] TTS for voice: ${voice_id}`);

        res.setHeader('Content-Type', 'audio/mpeg');
        res.setHeader('Transfer-Encoding', 'chunked');

        const response = await axios({
            method: 'post',
            url: 'https://api.wavespeed.ai/api/v3/minimax/speech-2.5-hd-preview/stream',
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${WAVESPEED_API_KEY}`
            },
            data: {
                "speed": 1,
                "volume": 1,
                "pitch": 0,
                "emotion": "happy",
                "text": text,
                "voice_id": voice_id || "vova_voice",
                "enable_sync_mode": true
            },
            responseType: 'stream'
        });

        let buffer = '';

        response.data.on('data', (chunk) => {
            buffer += chunk.toString();
            const lines = buffer.split('\n');
            buffer = lines.pop(); // Keep incomplete line

            for (const line of lines) {
                const trimmed = line.trim();
                if (trimmed.startsWith('data:')) {
                    const jsonStr = trimmed.replace('data:', '').trim();
                    if (!jsonStr || jsonStr === '[DONE]') continue;

                    try {
                        const json = JSON.parse(jsonStr);
                        // MiniMax/WaveSpeed often wraps status inside 'data' object, or at root
                        let status = json.status;
                        if (json.data && json.data.status !== undefined) {
                            status = json.data.status;
                        }

                        // If status is 2, it's the final summary packet which might contain full audio.
                        // We skip it to avoid duplication.
                        if (status === 2) {
                            console.log(" [Stream] Skipping Final Summary Packet (Status 2)");
                            continue;
                        }
                        // -----------------------------------------------

                        let hexAudio = null;
                        if (json.data && json.data.audio) {
                            hexAudio = json.data.audio;
                        } else if (json.audio) {
                            hexAudio = json.audio;
                        }

                        if (hexAudio) {
                            const audioBuf = Buffer.from(hexAudio, 'hex');
                            res.write(audioBuf);
                        }
                    } catch (e) {
                        console.error("JSON Parse Error on chunk:", e.message);
                    }
                }
            }
        });

        response.data.on('end', () => {
            console.log("[Stream] WaveSpeed stream finished.");
            res.end();
        });

        response.data.on('error', (err) => {
            console.error("[Stream] Upstream error:", err);
            res.end();
        });

    } catch (error) {
        console.error("[Stream] Error:", error.message);
        if (!res.headersSent) res.status(500).json({ error: "Streaming failed" });
        else res.end();
    }
});

app.listen(PORT, () => {
    console.log(`Streaming Server running at http://localhost:${PORT}`);
});