# Suno TTS backend

Stores family voice recordings and generates generic story speech (edge-tts). No voice cloning.

## Setup (Windows)

```powershell
cd C:\Suno_Project\suno-tts
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

API docs: http://localhost:8000/docs

## App URL

Set `EXPO_PUBLIC_API_URL` in the Expo app if you are not using the default:

- Web / iOS simulator: `http://localhost:8000`
- Android emulator: `http://10.0.2.2:8000`
- Physical phone: `http://YOUR_PC_LAN_IP:8000`
