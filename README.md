# EmoConvey AI Backend

## Quick Start (Gemini API - Recommended)

1. Get a **free** API key from [Google AI Studio](https://aistudio.google.com/apikey)

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set your API key and run:

```bash
set GEMINI_API_KEY=your_key_here
python server.py
```

4. Copy the WebSocket URL from the terminal into your app's Profile screen.

## Configuration

| Mode | Command | Requirements |
|------|---------|-------------|
| **Gemini** (Default) | `set AI_MODE=gemini` | Free API key, Internet |
| **Local** (MiniCPM) | `set AI_MODE=local` | CUDA GPU, 8GB+ VRAM |
| **Mock** (Testing) | `set AI_MODE=mock` | Nothing |

## Ngrok (Remote Access)

If you have an Ngrok auth token:

```bash
set NGROK_AUTH_TOKEN=your_token_here
python server.py
```

The server will print a public URL you can use from any device.

Without Ngrok, use `ws://10.0.2.2:8000/ws/live` for Android Emulator.
