"""Kaggle test: direct cloning vs Kokoro -> ChatterboxVC, in English and Hindi.

Standalone: does not touch the server, backend or app. Stop the server cell first so the GPU is free.

Kaggle cells:
    !apt-get install -y -q espeak-ng
    !pip install -q kokoro soundfile
    %%writefile /kaggle/working/test_tts_vc.py   <- paste this file under it
    !python /kaggle/working/test_tts_vc.py

Put the narrator's prompt at /kaggle/working/prompt.wav (or set PROMPT_WAV, or add any
*.prompt.wav as a Kaggle input). Outputs go to /kaggle/working/test_tts_vc/.
"""

from __future__ import annotations
import glob, os, time
from pathlib import Path
import soundfile as sf
import torch, torchaudio

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
OUT = Path("/kaggle/working/test_tts_vc")
# Kokoro stock voices per language. Try am_michael / hm_omega for a male narrator.
KOKORO = {
    "en": {"lang_code": "a", "voice": os.environ.get("KOKORO_EN_VOICE", "af_heart")},
    "hi": {"lang_code": "h", "voice": os.environ.get("KOKORO_HI_VOICE", "hf_alpha")},
}
# About 40 words each, the size of a story chunk.
TEXTS = {
    "en": (
        "Once upon a time, in a small village near the hills, there lived a little rabbit named Coco. "
        "Every morning she hopped to the river to drink fresh water, and every evening she told her "
        "friends a new story under the big mango tree."
    ),
    "hi": (
        "एक समय की बात है, पहाड़ों के पास एक छोटे से गाँव में कोको नाम का एक नन्हा खरगोश रहता था। "
        "वह हर सुबह नदी पर ताज़ा पानी पीने जाता था, और हर शाम बड़े आम के पेड़ के नीचे "
        "अपने दोस्तों को एक नई कहानी सुनाता था।"
    ),
}


def find_prompt() -> str:
    candidates = [os.environ.get("PROMPT_WAV", ""), "/kaggle/working/prompt.wav"]
    candidates += sorted(glob.glob("/kaggle/input/**/*.prompt.wav", recursive=True))
    candidates += sorted(glob.glob("/kaggle/input/**/*.wav", recursive=True))
    for path in candidates:
        if path and Path(path).is_file():
            return path
    raise SystemExit("No prompt found: upload the narrator's prompt.wav to /kaggle/working/prompt.wav")


def timed(fn):
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()
    started = time.perf_counter()
    result = fn()
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()
    return result, time.perf_counter() - started


def save(name: str, wav: torch.Tensor, sr: int) -> float:
    wav = wav.detach().cpu().float()
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    torchaudio.save(str(OUT / name), wav, sr)
    return wav.shape[-1] / sr


def kokoro_speech(pipeline, text: str, voice: str) -> torch.Tensor:
    parts = []
    for result in pipeline(text, voice=voice):
        audio = getattr(result, "audio", None)
        if audio is None:
            audio = result[2]  # older kokoro yields (graphemes, phonemes, audio)
        if audio is not None:
            parts.append(torch.as_tensor(audio).flatten())
    if not parts:
        raise RuntimeError("Kokoro returned no audio")
    return torch.cat(parts)


def main():
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    from chatterbox.vc import ChatterboxVC
    from kokoro import KPipeline

    OUT.mkdir(parents=True, exist_ok=True)
    prompt = find_prompt()
    print("Prompt:", prompt, "| device:", DEVICE)

    print("Loading models...")
    clone = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)
    vc = ChatterboxVC.from_pretrained(device=DEVICE)
    pipelines = {lang: KPipeline(lang_code=cfg["lang_code"], repo_id="hexgrad/Kokoro-82M", device=DEVICE)
                 for lang, cfg in KOKORO.items()}

    # Prepare the narrator once for each model, like the server does at voice save time.
    _, prep_clone = timed(lambda: clone.prepare_conditionals(prompt))
    _, prep_vc = timed(lambda: vc.set_target_voice(prompt))
    print(f"Voice prepare: clone {prep_clone:.1f}s, vc {prep_vc:.1f}s")

    # Warm-up so the first timed run doesn't include CUDA start-up cost.
    torch.cuda.manual_seed(42) if DEVICE.startswith("cuda") else torch.manual_seed(42)
    clone.generate("Hello there.", language_id="en", cfg_weight=0.3, exaggeration=0.5)
    kokoro_speech(pipelines["en"], "Hello there.", KOKORO["en"]["voice"])

    rows = []
    for lang, text in TEXTS.items():
        words = len(text.split())

        # a. Direct cloning (current way), same settings as the server.
        if DEVICE.startswith("cuda"):
            torch.cuda.manual_seed(42)
        wav, took = timed(lambda: clone.generate(text, language_id=lang, cfg_weight=0.3, exaggeration=0.5))
        seconds = save(f"{lang}_clone.wav", wav, clone.sr)
        rows.append(("clone", lang, words, took, seconds))

        # b. Kokoro only (stock voice).
        voice = KOKORO[lang]["voice"]
        stock, took_tts = timed(lambda: kokoro_speech(pipelines[lang], text, voice))
        kokoro_path = OUT / f"{lang}_kokoro.wav"
        sf.write(str(kokoro_path), stock.cpu().numpy(), 24000)
        rows.append((f"kokoro ({voice})", lang, words, took_tts, stock.shape[-1] / 24000))

        # c. Kokoro -> ChatterboxVC (narrator's voice). Total time = Kokoro + VC.
        converted, took_vc = timed(lambda: vc.generate(str(kokoro_path)))
        seconds = save(f"{lang}_vc.wav", converted, vc.sr)
        rows.append(("kokoro -> vc", lang, words, took_tts + took_vc, seconds))

    print()
    print(f"{'method':<22}{'lang':<6}{'words':>6}{'gen s':>9}{'audio s':>9}{'ratio':>8}")
    for method, lang, words, took, seconds in rows:
        ratio = took / seconds if seconds else 0
        print(f"{method:<22}{lang:<6}{words:>6}{took:>9.1f}{seconds:>9.1f}{ratio:>8.2f}")
    print("\nFiles:", ", ".join(sorted(p.name for p in OUT.glob("*.wav"))))
    print("Ratio under 1 = generated faster than it plays. Download the files and compare by ear.")


if __name__ == "__main__":
    main()
