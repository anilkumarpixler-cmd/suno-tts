# Suno TTS backend

Stores family voice recordings locally. On **play story**, synthesizes speech:

1. Chatterbox Multilingual on a **Google Colab GPU** (cloned narrator), if `COLAB_TTS_URL` is set and the tunnel is alive
2. **edge-tts** if Colab is down or unset
3. The Expo app falls back to on-device speech if this server is unreachable

Stories and recordings are never deleted when TTS fails.

`GET http://localhost:8002/health` tells URL-set vs tunnel-alive:

- `colabConfigured` — `.env` has `COLAB_TTS_URL`
- `colabReachable` / `colabCuda` / `colabLoaded` — Colab `/health` actually answered

## Setup (Windows)

```powershell
cd C:\Suno_Project\suno-tts
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
copy .env.example .env
python -m uvicorn app.main:app --host 0.0.0.0 --port 8002
```

API docs: http://localhost:8002/docs

## Colab GPU cloning

Use the **project notebook**, not a blank Colab + `npx localtunnel`.

1. In Google Colab: **File → Upload notebook** → [colab/chatterbox_tts.ipynb](colab/chatterbox_tts.ipynb)
2. **Runtime → Change runtime type → T4 GPU**, then **Run all**
3. Wait for `Model ready` and a **Cloudflare** URL (`https://….trycloudflare.com`). Do not use a stale `loca.lt` link.
4. Put that URL in `.env` as `COLAB_TTS_URL`
5. Restart this uvicorn process so it reloads `.env`
6. Open `https://….trycloudflare.com/health` — you want `{"ok": true, "cuda": true, "loaded": true}`
7. If an old story still plays a stock voice, delete matching Edge cache files under `data/tts/*.mp3` and play again (cloned audio is `data/tts/*.wav`)

Keep the Colab tab open. Free Colab drops the tunnel; then health shows `colabReachable: false` and cloning stops.

Optional: `COLAB_TTS_SECRET` on both Colab (`os.environ`) and local `.env`.

## App URL

Set `EXPO_PUBLIC_API_URL` in the **Expo app** (not in `suno-tts/.env`) only if the default is wrong. Use **one** of:

- Web / iOS simulator: `http://localhost:8002`
- Android emulator: `http://10.0.2.2:8002`
- Physical phone: `http://YOUR_PC_WIFI_IP:8002` (same Wi-Fi as the PC; `ipconfig` IPv4)

On a physical phone with Expo on the same Wi-Fi, you can usually leave `EXPO_PUBLIC_API_URL` unset.
