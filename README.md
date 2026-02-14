# EmoConvey AI Backend

## Quick Start (Gemini API - Recommended)

1. Get a **free** API key from [Google AI Studio](https://aistudio.google.com/apikey)

2. Install dependencies:

```bash
pip install -r requirements.txt
```


## Ngrok (Remote Access)

If you have an Ngrok auth token:

```bash
set NGROK_AUTH_TOKEN=your_token_here
python server.py
```

The server will print a public URL you can use from any device.

```bash 
Example:    wss://brysen-auxilytic-lita.ngrok-free.dev/ws/live
```