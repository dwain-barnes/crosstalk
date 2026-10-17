"""Browser front end for the Crosstalk loop.

    python -I server.py [--host 0.0.0.0] [--port 7860] [--tts-device 0]

Serves the page on http://127.0.0.1:7860 and runs one call at a time over /ws.
The browser sends 80 ms frames of 16 kHz PCM16 and gets the agent back as
24 kHz PCM16 chunks, each with a 16-byte header: the chunk's start on the
call clock and the server's clock when it was sent (two little-endian doubles).
Each call is saved under calls/<time>/ as it runs.
"""
import argparse
import asyncio
import json
import re
import statistics
import struct
import sys
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import websockets
from scipy.signal import resample_poly
from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.http11 import Response

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from loop import FRAME, LISTEN, NOVOICE, SR_IN, SR_OUT, Heard, Speaker, Turns, load_tts, measure  # noqa: E402

VOICED = 0.01   # 20 ms RMS that counts as someone talking, same as the loop's talk-over measure
HOP = int(0.02 * SR_OUT)
PAD = 64        # 16 kHz samples of context either side when resampling a block
FILES = {"/": ("index.html", "text/html; charset=utf-8"),
         "/app.js": ("app.js", "text/javascript"),
         "/style.css": ("style.css", "text/css"),
         "/mic.js": ("mic.js", "text/javascript")}
SHOWN = {"stt", "final", "typed", "tts", "cut", "bc_hold", "dropped_bc", "held_answer", "dropped_answer", "echo"}


def up(url):
    try:
        urllib.request.urlopen(url, timeout=2)
        return True
    except OSError:
        return False


def model_name():
    # the model file's folder name, for the badge on the page
    try:
        with urllib.request.urlopen(ARGS.llm + "/props", timeout=2) as r:
            return Path(json.load(r)["model_path"]).parent.name
    except (OSError, ValueError, KeyError):
        return None


def wav_header(frames):
    n = frames * 4
    return (b"RIFF" + struct.pack("<I", 36 + n) + b"WAVEfmt " +
            struct.pack("<IHHIIHH", 16, 1, 2, SR_OUT, SR_OUT * 4, 4, 16) + b"data" + struct.pack("<I", n))


class Tape:
    """The call as a stereo 24 kHz WAV (left caller, right agent), appended as it goes.

    The header is rewritten after every block, so the file is playable even if
    the server dies mid-call. Caller audio is laid down in arrival order; gaps
    in the stream (no mic, tab in the background) become silence so the two
    sides stay lined up with the call clock. Also keeps 20 ms voiced flags for
    the caller and the talk-over total.
    """

    def __init__(self, path, spk):
        self.f = open(path, "wb")
        self.f.write(wav_header(0))
        self.spk = spk
        self.caller = bytearray()
        self.first = None       # call time of the first caller frame
        self.last = -1.0
        self.done = 0           # 24 kHz frames written
        self.voiced = []
        self.over = 0.0

    def add(self, b, t):
        if self.first is None:
            self.first = t - FRAME
        behind = int((t - FRAME) * SR_IN) - len(self.caller) // 2
        if behind > 0.3 * SR_IN:
            self.caller += bytes(2 * behind)
        self.caller += b
        self.last = t

    def write(self, upto, pad=False):
        i0 = self.done * 2 // 3
        i1 = int(upto * SR_IN) // 320 * 320
        have = len(self.caller) // 2
        if pad and have < i1 + PAD:
            self.caller += bytes(2 * (i1 + PAD - have))
            have = i1 + PAD
        i1 = min(i1, (have - PAD) // 320 * 320)
        if i1 <= i0:
            return
        lo = max(0, i0 - PAD)
        x = np.frombuffer(bytes(self.caller[2 * lo:2 * (i1 + PAD)]), np.int16).astype(np.float32) / 32768
        left = resample_poly(x, 3, 2)[(i0 - lo) * 3 // 2:][:(i1 - i0) * 3 // 2]
        right = np.zeros_like(left)
        track = self.spk.track[self.done:self.done + len(left)]
        right[:len(track)] = track
        st = np.clip(np.stack([left, right], 1), -1, 1)
        self.f.write((st * 32767).astype("<i2").tobytes())
        self.done += len(left)
        self.f.seek(0)
        self.f.write(wav_header(self.done))
        self.f.seek(0, 2)
        self.f.flush()

        def rms(a):
            return np.sqrt(np.mean(a.reshape(-1, HOP) ** 2, axis=1))
        c, g = rms(left) > VOICED, rms(right) > VOICED
        self.voiced += c.tolist()
        self.over += float(np.sum(c & g)) * 0.02

    def close(self):
        self.f.close()


def options():
    return SimpleNamespace(llm=ARGS.llm, tick=0.6, prompt_tick=0.6, threshold=0.3, sym_threshold=False, lora_scale=None,
                           reask_mid_answer_only=True, bc_gate=True, answer_after_final=True, final_wait=1.0,
                           bc_words=1, cap=None, dot=False, persona=PERSONA, max_words=3,
                           echo_filter=ARGS.echo_filter)


def set_flags(o, x):
    for k, name in (("reask", "reask_mid_answer_only"), ("bc_gate", "bc_gate"), ("after_final", "answer_after_final")):
        if k in x:
            setattr(o, name, bool(x[k]))
    if "threshold" in x:
        o.threshold = float(x["threshold"])


def summary(events, ticks, utts, tape, o):
    lat = sorted(t["ms"] for t in ticks) or [0]
    tts = [e for e in events if e["ev"] == "tts"]
    res = {"utterances": [], "talk_over_s": round(tape.over, 2),
           "cuts": sum(e["ev"] == "cut" for e in events),
           "echoes": sum(e["ev"] == "echo" for e in events),
           "ticks": {"n": len(ticks), "speaking": sum(t["said"] != LISTEN for t in ticks),
                     "llm_ms_median": statistics.median(lat), "llm_ms_max": lat[-1],
                     "llm_ms_p95": lat[int(0.95 * (len(lat) - 1))],
                     "re_requests": sum(t["again"] for t in ticks)},
           "flags": {k: getattr(o, k) for k in ("threshold", "reask_mid_answer_only", "bc_gate", "answer_after_final")}}
    for u, r in sorted(utts.items()):
        said = " ".join(t["said"] for t in ticks if t.get("utt") == u)
        res["utterances"].append({"utt": u, "at": round(r["audio"], 2), "said": said, "ttfw": r.get("ttfw"),
                                  "kind": r.get("kind")})
    for k in ("dropped_bc", "held_answer", "dropped_answer"):
        res[k] = [{kk: v for kk, v in e.items() if kk != "ev"} for e in events if e["ev"] == k]
    if tts:
        res["tts"] = {"jobs": len(tts), "lag_median": statistics.median(e["lag"] for e in tts),
                      "synth_ms_median": round(statistics.median(e["gen"] - e["req"] for e in tts) * 1000)}
    return res


async def call(ws, start):
    o = options()
    set_flags(o, start)
    d = HERE / "calls" / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    d.mkdir(parents=True, exist_ok=True)
    events, lock = [], threading.Lock()
    jsonl = open(d / "events.jsonl", "w", encoding="utf-8")
    loop = asyncio.get_running_loop()
    outq = asyncio.Queue()

    def now():
        return time.perf_counter() - t0

    def send(m):
        loop.call_soon_threadsafe(outq.put_nowait, m)

    def log(ev, **kw):
        e = {"t": round(now(), 4), "ev": ev, **kw}
        with lock:
            events.append(e)
            if ev == "tts":
                utts.setdefault(kw["utt"], {"audio": kw["audio"]})
            if not jsonl.closed:
                jsonl.write(json.dumps(e) + "\n")
                jsonl.flush()
        if ev in SHOWN:
            send(json.dumps({**e, "type": ev}))

    spk = Speaker(60, now, log, ARGS.tts, engine=ENGINE)
    spk.out = lambda start, c: send(struct.pack("<dd", start, now()) + (np.clip(c, -1, 1) * 32767).astype("<i2").tobytes())
    heard = Heard(log)
    turns = Turns(o, spk, heard, log, now)
    tape = Tape(d / "call.wav", spk)
    utts = {}       # utt -> first audio time, then ttfw once the caller side is written
    typed = []
    done = asyncio.Event()
    await asyncio.to_thread(turns.warm)
    t0 = time.perf_counter()
    log("start", opts=dict(vars(o)), dir=str(d))

    stt = None
    try:
        stt = await websockets.connect(ARGS.stt, max_size=None)
        await stt.recv()
        await stt.send(json.dumps({"type": "session.update",
                                   "session": {"language": "en-US", "endpointing_ms": ARGS.endpoint_ms}}))
        await stt.recv()
    except OSError as e:
        stt = None
        send(json.dumps({"type": "error", "text": f"STT not reachable ({e}); typed lines only"}))

    async def rx():
        nonlocal stt
        try:
            async for m in stt:
                e = json.loads(m)
                if e["type"].endswith(".delta"):
                    if e["delta"]:
                        log("stt", delta=e["delta"], ap=e["audio_processed"])
                    turns.hear(e["delta"], e["audio_processed"], now())
                elif e["type"].endswith(".completed"):
                    log("final", text=e["transcript"])
                    turns.hear_final(e["transcript"], now())
        except websockets.ConnectionClosed:
            pass
        stt = None

    async def ticker():
        n = 1
        while not done.is_set():
            await asyncio.sleep(max(0, n * o.tick - now()))
            rec = await turns.tick(n, now())
            send(json.dumps({"type": "tick", **rec}))
            n = max(n + 1, int(now() / o.tick) + 1)

    async def pump():
        while not done.is_set():
            await asyncio.sleep(0.04)
            turns.pump()

    def ttfw(upto):
        # the agent's first audio minus the end of the caller's last voiced 20 ms or last typed line;
        # over speech if the caller was voiced in the 0.2 s before it
        with lock:
            todo = [(u, r) for u, r in utts.items() if "kind" not in r and r["audio"] <= upto]
        for u, r in todo:
            k = int(r["audio"] / 0.02)
            v = tape.voiced
            ends = [t for t in typed if t <= r["audio"]]
            last = next((i for i in range(k - 1, max(0, k - 1500) - 1, -1) if v[i]), None)
            if last is not None:
                ends.append((last + 1) * 0.02)
            if any(v[max(0, k - 10):k]):
                r["kind"] = "over speech"
            elif not ends:
                r["kind"] = "before caller"
            else:
                r["kind"], r["ttfw"] = "after caller", round(r["audio"] - max(ends), 3)
            send(json.dumps({"type": "ttfw", "utt": u, **r}))

    async def writer():
        k = 0
        while True:
            await asyncio.sleep(0.5)
            t = now()
            tape.write(t - 0.5, pad=t - tape.last > 0.3)
            ttfw(tape.done / SR_OUT)
            k += 1
            if k % 10 == 0:
                (d / "summary.json").write_text(json.dumps(summary(events, turns.ticks, utts, tape, o), indent=1),
                                                encoding="utf-8")
            send(json.dumps({"type": "stat", "t": round(t, 1), "over": round(tape.over, 2), "stt": stt is not None,
                             "queue": round(spk.left(), 2)}))

    async def sender():
        while True:
            m = await outq.get()
            await ws.send(m)

    def died(t):
        if not t.cancelled() and t.exception() and not isinstance(t.exception(), websockets.ConnectionClosed):
            print("task failed:", repr(t.exception()))
            send(json.dumps({"type": "error", "text": f"{t.get_coro().__name__} failed: {t.exception()!r}"}))

    tasks = [asyncio.create_task(x()) for x in (ticker, pump, writer, sender) + ((rx,) if stt else ())]
    for t in tasks:
        t.add_done_callback(died)
    await ws.send(json.dumps({"type": "started", "dir": str(d), "stt": stt is not None}))
    print("call started", d)
    try:
        async for m in ws:
            if isinstance(m, bytes):
                tape.add(m, now())
                if stt:
                    try:
                        await stt.send(m)
                    except websockets.ConnectionClosed:
                        stt = None
            else:
                x = json.loads(m)
                if x["type"] == "end":
                    break
                if x["type"] == "text" and x["text"].strip():
                    t = now()
                    log("typed", text=x["text"])
                    typed.append(t)
                    heard.delta(" " + x["text"].strip(), heard.grew, t)
                    heard.final(x["text"].strip(), t)
                elif x["type"] == "flags":
                    set_flags(o, x)
                    log("flags", **x)
    except websockets.ConnectionClosed:
        print("browser disconnected")
    finally:
        # files first: this also runs when the server is stopped with Ctrl-C
        done.set()
        for t in tasks:
            t.cancel()
        if spk.left() > 0 or spk.busy:
            spk.cut("end")
        spk.closed = True
        end = now()
        tape.write(end, pad=True)
        ttfw(end)
        tape.close()
        res = summary(events, turns.ticks, utts, tape, o)
        (d / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        truth = start.get("truth")
        if truth and Path(truth).exists() and turns.ticks and tape.first is not None:
            # word times in the truth file count from the first caller frame
            tr = json.load(open(truth))
            sh = tape.first
            tr = {"duration": tr["duration"] + sh,
                  "words": [{**w, "start": w["start"] + sh, "end": w["end"] + sh} for w in tr["words"]]}
            res["truth"] = measure(events, turns.ticks, heard.words, spk.track[:int(end * SR_OUT)], tr)
            (d / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        with lock:
            jsonl.close()
        print("call saved", d)
        if stt:
            await stt.close()
        try:
            await ws.send(json.dumps({"type": "saved", "dir": str(d), "summary": res}))
        except websockets.ConnectionClosed:
            pass


active = False


async def handler(ws):
    global active
    stt, llm, model = await asyncio.gather(asyncio.to_thread(up, f"http://127.0.0.1:{ARGS.stt_port}/ready"),
                                           asyncio.to_thread(up, ARGS.llm + "/health"),
                                           asyncio.to_thread(model_name))
    # TTS is loaded in this process before the server listens, so it is up whenever the page can connect
    await ws.send(json.dumps({"type": "hello", "busy": active, "stt": stt, "llm": llm, "tts": True,
                              "model": model}))
    try:
        async for m in ws:
            if not isinstance(m, str) or json.loads(m).get("type") != "start":
                continue
            if active:
                await ws.send(json.dumps({"type": "error", "text": "another call is running"}))
                continue
            if not up(ARGS.llm + "/health"):
                await ws.send(json.dumps({"type": "error", "text": f"LLM not answering on {ARGS.llm}"}))
                continue
            active = True
            try:
                await call(ws, json.loads(m))
            finally:
                active = False
    except websockets.ConnectionClosed:
        pass


def static(conn, req):
    path = req.path.split("?")[0]
    if path == "/ws":
        return None
    parts = path.split("/")
    if len(parts) == 4 and parts[1] == "calls" and parts[3] == "call.wav" and re.fullmatch(r"[\w-]+", parts[2]):
        f = HERE / "calls" / parts[2] / "call.wav"
        if not f.is_file():
            return conn.respond(404, "not found\n")
        body = f.read_bytes()
        return Response(200, "OK", Headers([("Content-Type", "audio/wav"), ("Content-Length", str(len(body))),
                                            ("Content-Disposition", f'attachment; filename="crosstalk_{parts[2]}.wav"'),
                                            ("Cache-Control", "no-store")]), body)
    if path not in FILES:
        return conn.respond(404, "not found\n")
    name, ctype = FILES[path]
    body = (HERE / name).read_bytes()
    return Response(200, "OK", Headers([("Content-Type", ctype), ("Content-Length", str(len(body))),
                                        ("Cache-Control", "no-store")]), body)


async def main():
    async with serve(handler, ARGS.host, ARGS.port, process_request=static, max_size=None, compression=None):
        print(f"open http://127.0.0.1:{ARGS.port}")
        await asyncio.get_running_loop().create_future()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to reach the page from another machine")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--llm", default="http://127.0.0.1:8080")
    p.add_argument("--stt-port", type=int, default=9100)
    p.add_argument("--endpoint-ms", type=int, default=500)
    p.add_argument("--voice", default="anna", help="Pocket TTS voice: a built-in name or a .safetensors state")
    p.add_argument("--tts", choices=["pytorch", "grow"], default="grow",
                   help="grow: one Pocket stream per utterance that takes words as they come; "
                        "pytorch: a new stream per clause")
    p.add_argument("--tts-device", default="cpu", help="cpu, or a CUDA device index or UUID")
    p.add_argument("--echo-filter", action=argparse.BooleanOptionalAction, default=True,
                   help="drop STT words that match what the agent is saying (its voice leaking into the mic)")
    p.add_argument("--persona", help="text file added to the system prompt (default persona.txt here if it exists, "
                                     "'' for none)")
    ARGS = p.parse_args()
    ARGS.stt = f"ws://127.0.0.1:{ARGS.stt_port}/v1/audio/transcriptions/realtime"
    if ARGS.persona is None and (HERE / "persona.txt").exists():
        ARGS.persona = HERE / "persona.txt"
    PERSONA = Path(ARGS.persona).read_text(encoding="utf-8").strip() if ARGS.persona else None
    print("persona:", ARGS.persona or "none")
    if not up(ARGS.llm + "/health"):
        sys.exit(f"LLM server not answering on {ARGS.llm}")
    if not up(f"http://127.0.0.1:{ARGS.stt_port}/ready"):
        print(f"no STT server on {ARGS.stt_port}; calls will take typed lines only")
    print("loading Pocket TTS")
    ENGINE = load_tts(ARGS.tts, ARGS.voice, ARGS.tts_device)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
