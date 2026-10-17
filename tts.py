"""Thin wrapper around Kyutai Pocket TTS for the voice agent loop.

    python -I tts.py "Oh, definitely get" "it checked." --voice anna --out out.wav
"""
import argparse
import os
import time
import wave
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np  # noqa: E402
from pocket_tts import TTSModel  # noqa: E402

SAMPLE_RATE = 24000


def load(device="cpu", language=None):
    m = TTSModel.load_model(language=language)
    return m.to(device).eval()


def voice(m, src):
    # src: a built-in name ("anna"), a .wav/.mp3 path (needs the gated cloning
    # weights), or a .safetensors state saved with save_voice().
    if Path(src).suffix in (".wav", ".mp3", ".flac"):
        return m.get_state_for_audio_prompt(Path(src), truncate=True)
    return m.get_state_for_audio_prompt(str(src))


def save_voice(state, path):
    from pocket_tts.models.model_state import export_model_state
    export_model_state(state, str(path))


def stream(m, state, text, more=False, stop=None):
    """Yield float32 mono numpy chunks at 24 kHz as they are decoded.

    more=True means the sentence continues in the next call: pocket-tts would
    otherwise add a full stop and read the chunk as a finished sentence, with
    falling pitch and ~400 ms of trailing silence.
    """
    m.append_terminal_punctuation = not more
    try:
        for chunk in m.generate_audio_stream(state, text, frames_after_eos=1 if more else None, stop=stop):
            yield chunk.float().cpu().numpy()
    finally:
        m.append_terminal_punctuation = True


def write_wav(path, audio):
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("chunks", nargs="+", help="text chunks, synthesised one after another")
    p.add_argument("--voice", default="anna")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="out.wav")
    a = p.parse_args()

    m = load(a.device)
    state = voice(m, a.voice)
    parts = []
    for i, text in enumerate(a.chunks):
        t0 = time.perf_counter()
        first = None
        for c in stream(m, state, text, more=i < len(a.chunks) - 1):
            if first is None:
                first = time.perf_counter() - t0
            parts.append(c)
        print(f"{text!r}: first audio {first * 1000:.0f} ms, total {(time.perf_counter() - t0) * 1000:.0f} ms")
    write_wav(a.out, np.concatenate(parts))


if __name__ == "__main__":
    main()
