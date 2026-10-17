"""Drive the UI server's WebSocket the way the page does, without a browser or a mic.

    python -I check.py --text "hello, have you got a minute"
    python -I check.py --wav ../scenarios/pause.wav

Prints what the page would show and checks that a reply arrived and the call
folder has its files. With --wav, a <name>.json next to the WAV is passed
on so the server also runs the loop's measurements.
"""
import argparse
import asyncio
import json
import sys
import time
import wave
from pathlib import Path

import websockets

FRAME = 1280 * 2


async def main(a):
    async with websockets.connect(a.url, max_size=None) as ws:
        hello = json.loads(await ws.recv())
        print("hello", hello)
        start = {"type": "start"}
        if a.wav and Path(a.wav).with_suffix(".json").exists():
            start["truth"] = str(Path(a.wav).with_suffix(".json").resolve())
        await ws.send(json.dumps(start))
        got = {"audio": 0, "first_audio": None, "ticks": [], "ttfw": [], "finals": [], "saved": None, "errors": []}
        t0 = time.perf_counter()
        started = asyncio.Event()
        sent_at = []

        async def rx():
            async for m in ws:
                if isinstance(m, bytes):
                    got["audio"] += (len(m) - 16) // 2
                    if got["first_audio"] is None:
                        got["first_audio"] = time.perf_counter() - t0
                    continue
                e = json.loads(m)
                if e["type"] == "started":
                    print("started", e["dir"], "stt", e["stt"])
                    started.set()
                elif e["type"] == "tick" and e["said"] != "<listen>":
                    got["ticks"].append(e)
                    print(f"{e['at']:6.2f}  agent  {e['said']:<26} (heard {e['heard']!r}, {e['ms']:.0f} ms)")
                elif e["type"] in ("final", "typed"):
                    got["finals"].append(e["text"])
                    print(f"{e['t']:6.2f}  caller {e['text']}")
                elif e["type"] == "ttfw":
                    got["ttfw"].append(e)
                    print(f"        utt {e['utt']}: {e['kind']} {e.get('ttfw', '')}")
                elif e["type"] == "cut":
                    print(f"{e['t']:6.2f}  cut {e['why']}")
                elif e["type"] == "error":
                    got["errors"].append(e["text"])
                    print("error:", e["text"])
                elif e["type"] == "saved":
                    got["saved"] = e
                    return

        task = asyncio.create_task(rx())
        await started.wait()
        if a.text:
            for line in a.text:
                sent_at.append(time.perf_counter() - t0)
                await ws.send(json.dumps({"type": "text", "text": line}))
                await asyncio.sleep(a.gap)
        else:
            with wave.open(a.wav) as w:
                pcm = w.readframes(w.getnframes())
            t1 = time.perf_counter()
            for k, i in enumerate(range(0, len(pcm), FRAME)):
                await asyncio.sleep(max(0, t1 + k * 0.08 - time.perf_counter()))
                await ws.send(pcm[i:i + FRAME])
            await asyncio.sleep(a.gap)
        await ws.send(json.dumps({"type": "end"}))
        await asyncio.wait_for(task, 30)

    ok = True
    if not got["ticks"] or not got["audio"]:
        print("FAIL: no reply")
        ok = False
    else:
        print(f"agent audio {got['audio'] / 24000:.1f} s in {len(got['ticks'])} speaking ticks")
        if a.text:
            print(f"first audio chunk {got['first_audio'] - sent_at[0]:.2f} s after the first typed line was sent")
    d = Path(got["saved"]["dir"])
    for f in ("call.wav", "events.jsonl", "summary.json"):
        size = (d / f).stat().st_size if (d / f).exists() else 0
        print(f"{d / f}: {size} bytes")
        ok &= size > 0
    with wave.open(str(d / "call.wav")) as w:
        print(f"call.wav {w.getnchannels()} ch {w.getframerate()} Hz {w.getnframes() / w.getframerate():.1f} s")
    s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    print(json.dumps({k: s[k] for k in ("utterances", "talk_over_s", "cuts", "ticks")}, indent=1))
    if "truth" in s:
        t = s["truth"]
        print("measured against the script:", json.dumps({k: t[k] for k in ("answers", "missed_gaps", "talk_over_s")}, indent=1))
    return ok


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", nargs="+", help="lines to type, one every --gap seconds")
    g.add_argument("--wav", help="16 kHz mono WAV streamed as the caller in real time")
    p.add_argument("--gap", type=float, default=6, help="seconds of silence after each line or after the WAV")
    p.add_argument("--url", default="ws://127.0.0.1:7860/ws")
    sys.exit(0 if asyncio.run(main(p.parse_args())) else 1)
