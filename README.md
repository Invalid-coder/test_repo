# LAM-A2E: Audio to Expression with 3D Avatar Rendering

Complete client-side pipeline for generating facial expressions from audio and rendering them on 3D avatars using ONNX Runtime Web and Gaussian Splatting.

## Features

- 🎤 Audio-to-Expression conversion using ONNX models
- 🎭 Real-time 3D avatar animation with Gaussian Splatting
- 🌐 Client-side processing with optional upload server
- 📊 52 ARKit blendshapes support
- 💾 Export blendshape data to JSON
- 🎮 Interactive playback controls
- 📦 Custom avatar upload support

## Quick Start

### One-Command Setup (Complete with Upload Server)

Run this single command to set up everything including the upload server:

```bash
npm init -y && npm pkg set scripts.dev="vite --port 8000 --host" scripts.build="vite build" scripts.preview="vite preview --port 8000" scripts.upload="node upload-server.js" scripts.start="concurrently \"npm run dev\" \"npm run upload\"" && npm install vite gaussian-splat-renderer-for-lam express multer cors axios dotenv && npm install -D concurrently && echo 'import { defineConfig } from "vite"\n\nexport default defineConfig({\n  server: {\n    port: 8000,\n    host: true,\n    headers: {\n      "Cross-Origin-Embedder-Policy": "credentialless",\n      "Cross-Origin-Opener-Policy": "same-origin",\n    }\n  },\n  preview: {\n    port: 8000,\n    host: true,\n    headers: {\n      "Cross-Origin-Embedder-Policy": "credentialless",\n      "Cross-Origin-Opener-Policy": "same-origin",\n    }\n  }\n})' > vite.config.js
```

Then launch the application with both servers:

```bash
npm start
```

Open your browser to `http://localhost:8000`

### Basic Setup (Without Upload Server)

If you only want to use the default avatar without custom uploads:

```bash
npm init -y && npm pkg set scripts.dev="vite --port 8000 --host" scripts.build="vite build" scripts.preview="vite preview --port 8000" && npm install vite gaussian-splat-renderer-for-lam && echo 'import { defineConfig } from "vite"\n\nexport default defineConfig({\n  server: {\n    port: 8000,\n    host: true,\n    headers: {\n      "Cross-Origin-Embedder-Policy": "credentialless",\n      "Cross-Origin-Opener-Policy": "same-origin",\n    }\n  },\n  preview: {\n    port: 8000,\n    host: true,\n    headers: {\n      "Cross-Origin-Embedder-Policy": "credentialless",\n      "Cross-Origin-Opener-Policy": "same-origin",\n    }\n  }\n})' > vite.config.js
```

Then launch:

```bash
npm run dev
```

Open your browser to `http://localhost:8000`