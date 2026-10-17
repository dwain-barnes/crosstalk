"""Micro-turn duplex voice loop: caller audio -> streaming STT -> LLM tick -> Pocket TTS.

    python -I loop.py --wav scenarios/pause.wav --out results/pause
    python -I loop.py --mic --seconds 120 --out results/mic

Needs the LLM server and the STT server running (see scripts/). Writes
<out>.wav (left caller, right agent), <out>.jsonl (events) and, when the WAV
has a <name>.json next to it with true word times, <out>.json (measurements).
"""
import argparse
import asyncio
import difflib
import json
import math
import os
import queue
import re
import statistics
import threading
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np
import websockets
from scipy.signal import resample_poly

LISTEN, NOVOICE = "<listen>", "<no voice>"
SR_IN, SR_OUT = 16000, 24000
FRAME = 0.08
# A trailing word with no new piece over this much decoded audio counts as
# complete. Without endpointing the last word before a pause would otherwise
# wait for the next word.
IDLE = 0.32
# later words of an utterance go to the TTS once less than this much audio is left to play. anna takes
# 250-380 ms to its first speech (it generates its quiet lead-in first), so this covers that with room
LOW = 0.45
# grow backend: the first chunk of an utterance plays this much later, and a sentence end starts a new
# sequence once the utterance has this many words. 0.25 s covers most of the stall at the first join on
# the CPU but adds that much to every reply; on the GPU it is left at 0
GROW_DELAY = 0.0
GROW_WORDS = 20
# echo filter: the agent's voice leaks from the caller's headphones into the mic and the STT writes it down.
# Caller words that arrive while the track plays, or this long after it ends, are compared with the words fed
# to the speaker in the last ECHO_SPAN seconds. The span is on feed time, which runs about 0.5 s ahead of
# playback, and the STT adds another 0.3-0.5 s
ECHO_TAIL = 0.8
ECHO_SPAN = 4.5
BACKCHANNEL = re.compile(r"^(oh,? )?(mh?mm?|mm+|uh[- ]huh|right|okay|ok|yeah|yes|i see|sure|got it|go on|lovely)[.!,]?$", re.I)
HAZARD = re.compile(r"(stop|wait|don't|do not|careful|please stop|please do not)\b", re.I)


NUMBERS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}


def prompt_for(tick, persona=None, words=3):
    s = ("You are a voice assistant in a live phone call; both sides can talk at the same time. Every "
         f"{tick:g} seconds you get what the microphone heard in that moment, as a streaming transcript fragment, or "
         f"{NOVOICE}. You reply with up to {NUMBERS[words]} new words to say next, or {LISTEN} to add nothing.")
    # training used the first paragraph alone, so a persona is a shift to measure, not a free change
    return s + "\n\n" + persona if persona else s


def complete(url, body):
    req = urllib.request.Request(url + "/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=60))


def llm(url, prefix, heard, thr, sym=False, lora=None, prev=LISTEN, mid_only=False, n_predict=16):
    prompt = prefix + f"<|im_start|>user\n{heard}<|im_end|>\n<|im_start|>assistant\n"
    body = {"prompt": prompt, "n_predict": n_predict, "temperature": 0, "stop": ["<|im_end|>", "\n"],
            "cache_prompt": True, "n_probs": 8}
    if lora is not None:
        body["lora"] = [{"id": 0, "scale": lora}]
    j = complete(url, body)
    out = j["content"].strip()
    top = j["completion_probabilities"][0]["top_logprobs"]
    ids = [p["id"] for p in top if p["token"].startswith("<")]
    p_listen = sum(math.exp(p["logprob"]) for p in top if p["token"].startswith("<"))
    again = out.startswith("<") and p_listen < thr and bool(ids) and (prev != LISTEN or not mid_only)
    # symmetric rule: a word that won while p_listen was at or over the threshold counts as a listen.
    # held keeps the words that were dropped.
    held = out if sym and not out.startswith("<") and p_listen >= thr else ""
    if again:
        body.update({"logit_bias": [[i, False] for i in ids], "n_probs": 0})
        out = complete(url, body)["content"].strip()
    if held:
        out = LISTEN
    return (LISTEN if out.startswith("<") else out), p_listen, again, held, j.get("timings", {}).get("prompt_n")


class Heard:
    """Running transcript built from add-only STT deltas.

    A word is complete once a space-led piece follows it, a final arrives, or
    it has not grown for IDLE seconds of decoded audio. A word still growing at
    tick time is held for a later tick, so ticks carry whole words. Punctuation
    that lands after its word was sent goes out at the front of the next
    fragment (", so"), which is how the training fragments look.

    The STT only writes a full stop once the next word starts, so a turn ends
    on a bare word. With dot set, a silent tick after a bare word that closed
    at least dot seconds ago sends "." once.
    """

    def __init__(self, log, dot=None):
        self.text = ""
        self.sent = 0
        self.open = None
        self.grew = 0.0
        self.words = []
        self.log = log
        self.seg = 0        # where the current endpointed utterance starts in text
        self.fresh = False
        self.dot = dot
        self.bare = False   # last fragment sent ended on a word
        self.final_at = -1.0

    def close(self, t, how):
        if self.open is None:
            return
        w = self.words[-1]
        w["closed"], w["how"] = t, how
        self.log("word", text=w["text"], first=w["first"], last=w["last"], how=how)
        self.open = None

    def final(self, transcript, t):
        # with endpointing the final carries the punctuation the deltas lacked
        got, cur = transcript.strip(), self.text[self.seg:].strip()
        if got and got.startswith(cur) and len(got) > len(cur):
            tail = got[len(cur):]
            self.text += tail
            if self.words and self.open is not None:
                self.words[-1]["text"] += tail.rstrip()
            self.log("final_tail", tail=tail)
        self.close(t, "final")
        self.final_at = t
        self.seg = len(self.text)
        self.fresh = True

    def delta(self, d, ap, t):
        for piece in re.findall(r"\s*\S+", d):
            if self.fresh and not piece[0].isspace():
                piece = " " + piece
            self.fresh = False
            new = piece[0].isspace() or not self.words
            if new or (self.open is None and re.search(r"\w", piece)):
                self.close(t, "space")
                self.open = len(self.text)
                self.words.append({"text": piece.strip(), "first": t, "last": t, "closed": None,
                                   "cont": not new})
                self.grew = ap
            elif self.open is not None:
                self.words[-1]["text"] += piece
                if re.search(r"\w", piece):
                    self.words[-1]["last"] = t
                    self.grew = ap
            self.text += piece
        if self.open is not None and ap - self.grew >= IDLE:
            self.close(t, "idle")

    def take(self, t):
        end = len(self.text) if self.open is None else max(self.open, self.sent)
        frag = self.text[self.sent:end].strip()
        self.sent = end
        for w in self.words:
            if w["closed"] is not None and "tick" not in w:
                w["tick"] = t
        if not re.search(r"\w|[,.?!]", frag):
            if (self.dot is not None and self.bare and self.open is None
                    and t - self.words[-1]["closed"] >= self.dot):
                self.bare = False
                self.log("dot")
                return "."
            return NOVOICE
        self.bare = bool(re.search(r"\w$", frag))
        return frag


def load_tts(backend, voice="anna", device="cpu"):
    if device != "cpu":
        # an index or a UUID; a UUID stays right if a card drops out and the indices move
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        device = "cuda"
    import tts
    m = tts.load(device)
    v = tts.voice(m, voice)
    if backend == "grow":
        import tts_grow
        g = tts_grow.open(m, v)
        g.feed("Hello there.")
        g.close()
        for _ in g.pull():
            pass
        return tts_grow, m, v
    for _ in tts.stream(m, v, "Hello there.", more=False):
        pass
    return tts, m, v


class Speaker:
    """Pocket TTS on a worker thread, mixed onto an agent track indexed by wall clock.

    In WAV mode the track is the simulated playback; in mic mode the output
    stream reads from it. engine is a (module, model, voice) from load_tts, so
    a long-running caller can load the model once. out, if set, is called with
    (start, chunk) for each chunk placed on the track.

    With the grow backend an utterance is one tts_grow stream: say() opens it on
    the first words and feeds it the rest, done() closes it.
    """

    def __init__(self, seconds, now, log, backend, voice="anna", engine=None, device="cpu"):
        self.tts, self.m, self.v = engine or load_tts(backend, voice, device)
        self.grow = backend == "grow"
        self.g = None           # the open grow stream and its utt
        self.g_utt = None
        self.cur = None         # the grow stream being played
        self.track = np.zeros(int(seconds * SR_OUT), np.float32)
        self.now, self.log = now, log
        self.out = None
        self.fed = []           # (time, text) of every say(), for the echo filter
        self.end = 0.0          # wall time where queued audio runs out
        self.epoch = 0
        self.jobs = queue.Queue()
        self.busy = 0
        self.closed = False
        self.stop = threading.Event()
        self.lock = threading.Lock()
        threading.Thread(target=self.run, daemon=True).start()

    def say(self, utt, text, more, at):
        with self.lock:
            self.fed.append((self.now(), text))
            if not self.grow:
                self.busy += 1
                self.jobs.put((self.epoch, utt, text, more, at, None))
                return
            if self.g is None or self.g_utt != utt:
                if self.g:
                    self.g.close()
                self.g, self.g_utt = self.tts.open(self.m, self.v), utt
                self.busy += 1
                self.jobs.put((self.epoch, utt, text, more, at, self.g))
            self.g.feed(text)
            if not more and len(self.g.text.split()) >= GROW_WORDS:
                # the next sentence starts a fresh sequence: Pocket is trained on sentences up to ~50
                # tokens, and the prefill grows with every frame already spoken
                self.g.close()
                self.g = None

    def done(self, utt):
        with self.lock:
            if self.g and self.g_utt == utt:
                self.g.close()
                self.g = None

    def left(self):
        return max(0.0, self.end - self.now())

    def run(self):
        while not self.closed:
            try:
                epoch, utt, text, more, at, g = self.jobs.get(timeout=0.8)
            except queue.Empty:
                continue
            if epoch == self.epoch:
                self.speak(epoch, utt, text, more, at, g)
            elif g:
                g.stop()
            with self.lock:
                self.busy -= 1

    def speak(self, epoch, utt, text, more, at, g):
        self.stop = threading.Event()
        t_req = self.now()
        rec = None
        tail = np.zeros(0, np.float32)
        cpu, wall = time.thread_time(), time.perf_counter()
        self.cur = g
        src = g.pull() if g else self.tts.stream(self.m, self.v, text, more=more, stop=self.stop)
        for c in src:
            t = self.now()
            if rec is None:
                # drop the quiet lead-in Pocket TTS puts before each chunk. With anna it is 0.5-1 s of
                # room tone up to 0.02 RMS, so look for the first 10 ms that is clearly speech and keep
                # 50 ms before it for soft starts
                c = np.concatenate([tail, c])
                n = len(c) // 240 * 240
                loud = np.flatnonzero(np.sqrt(np.mean(c[:n].reshape(-1, 240) ** 2, axis=1)) > 0.03)
                if loud.size == 0:
                    tail = c[-1200:]
                    continue
                c = c[max(0, loud[0] * 240 - 1200):]
            with self.lock:
                if epoch != self.epoch:
                    break
                # a grow stream stalls 200-300 ms at each new fragment while it reads the text again,
                # so its first chunk waits a little to leave that much queued
                start = max(t + (GROW_DELAY if g and rec is None else 0), self.end)
                i = int(start * SR_OUT)
                if i + len(c) > len(self.track):
                    self.track = np.concatenate([self.track, np.zeros(i + len(c) + 60 * SR_OUT - len(self.track),
                                                                      np.float32)])
                n = max(0, min(len(c), len(self.track) - i))
                self.track[i:i + n] = c[:n]
                self.end = start + len(c) / SR_OUT
                if self.out:
                    self.out(start, c)
            if rec is None:
                # lag: from the tick that produced the words to when they are heard
                a = start + min(240, len(c)) / SR_OUT
                rec = {"utt": utt, "text": text, "more": more, "req": t_req, "gen": t, "play": start,
                       "audio": a, "queued": round(start - t, 3), "lag": round(a - at, 3)}
                self.log("tts", **rec)
        if g:
            g.stop()
            self.log("grow", utt=utt, text=g.text, prefills=[(n, k, round(s * 1000)) for n, k, s in g.prefills],
                     steps=g.steps, gen_cpu=round(g.cpu, 3), dec_cpu=round(time.thread_time() - cpu, 3),
                     wall=round(time.perf_counter() - wall, 3))

    def cut(self, why):
        with self.lock:
            t = self.now()
            dropped = max(0.0, self.end - t)
            self.epoch += 1
            self.stop.set()
            for g in (self.g, self.cur):
                if g:
                    g.stop()
            self.g = None
            while not self.jobs.empty():
                *_, g = self.jobs.get_nowait()
                if g:
                    g.stop()
                self.busy -= 1
            self.track[int(t * SR_OUT):int(self.end * SR_OUT) + 1] = 0
            self.end = min(self.end, t)
        self.log("cut", why=why, dropped=dropped)
        return dropped


def ends_sentence(s):
    return s.rstrip().endswith((".", "?", "!"))


def letters(s):
    return re.sub(r"[^a-z]", "", s.lower())


def is_echo(s, ref):
    """s (letters only) is part of ref, or close to a stretch of it."""
    if len(s) < 4:
        return False
    if s in ref:
        return True
    if len(s) < 6:
        return False
    # fewest edits to turn s into any stretch of ref (the row starts at zero so the stretch can begin anywhere)
    row = [0] * (len(ref) + 1)
    for i, c in enumerate(s, 1):
        new = [i]
        for j, r in enumerate(ref, 1):
            new.append(min(row[j] + 1, new[j - 1] + 1, row[j - 1] + (c != r)))
        row = new
    return min(row) / len(s) <= 0.25


class Echo:
    """Keeps the agent's own voice, picked up by the caller's mic, out of the transcript.

    STT words that arrive while the agent track plays (or ECHO_TAIL after) wait here until the next tick and are
    then judged together, letters only, against the agent's recent words. Words judged echo never reach Heard,
    so the model does not see them and they cannot cut the agent off. A run under four letters is too short to
    judge: it counts as an echo's tail if it follows one closely, waits up to two ticks for more words if it is
    three letters from what the agent said, and otherwise goes through.
    """

    def __init__(self, heard, spk, log):
        self.heard, self.spk, self.log = heard, spk, log
        self.pend = []
        self.last = None        # (time, letters) of the latest echo run
        self.dropped = False    # echo was taken out of the current STT segment

    def delta(self, d, ap, t):
        gated = t < self.spk.end + ECHO_TAIL
        if gated or self.pend:
            self.pend.append({"d": d, "ap": ap, "t": t, "gated": gated})
        else:
            self.heard.delta(d, ap, t)

    def final(self, text, t):
        if self.pend:
            self.pend.append({"f": text, "t": t})
        else:
            self.end_segment(text, t)

    def end_segment(self, text, t):
        # the final repeats what the deltas said, so after dropping some of them it would put them back
        self.heard.final("" if self.dropped else text, t)
        self.dropped = False

    def sift(self, t):
        pend, self.pend = self.pend, []
        run = [x for x in pend if "d" in x and x["gated"]]
        gone = []
        if run:
            ref = letters(" ".join(w for ft, w in self.spk.fed if run[0]["t"] - ECHO_SPAN <= ft <= run[-1]["t"]))
            # the longest start of the run that matches, then what is left; a word the STT mangles
            # ("death" for "desk") only has to be a tail of something that matched
            while run:
                for k in range(len(run), 0, -1):
                    s = letters("".join(x["d"] for x in run[:k]))
                    near = self.last is not None and run[0]["t"] - self.last[0] <= 1.5
                    if len(s) >= 4 and (is_echo(s, ref) or near and is_echo(self.last[1] + s, ref)):
                        self.last = (run[k - 1]["t"], (self.last[1] if near else "") + s)
                        gone += run[:k]
                        run = run[k:]
                        break
                else:
                    break
            if run:
                s = letters("".join(x["d"] for x in run))
                if len(s) < 4 and self.last is not None and run[0]["t"] - self.last[0] <= 0.6:
                    gone += run
                elif not gone and len(s) == 3 and s in ref and not any(x.get("wait", 0) > 1 for x in run):
                    # could be the start of something she said
                    for x in run:
                        x["wait"] = x.get("wait", 0) + 1
                    self.pend = pend
                    return
            if gone:
                self.log("echo", text="".join(x["d"] for x in gone).strip(), aps=[x["ap"] for x in gone])
                self.dropped = True
        for x in pend:
            if "f" in x:
                self.end_segment(x["f"], x["t"])
            elif not any(x is g for g in gone):
                self.heard.delta(x["d"], x["ap"], x["t"])


class Turns:
    """The agent's side of the call: one LLM request per tick, and the rules for which words reach the speaker.

    Options are read from a on every tick, so they can change mid-call.
    """

    def __init__(self, a, spk, heard, log, now):
        self.a, self.spk, self.heard, self.log, self.now = a, spk, heard, log, now
        self.system = f"<|im_start|>system\n{prompt_for(a.prompt_tick, a.persona, a.max_words)}<|im_end|>\n"
        self.prefix = self.system
        self.prev = LISTEN
        self.ticks = []
        self.echo = Echo(heard, spk, log)
        # held: a short start kept back because the caller had moved on; played if it grows past two words
        # wait: when an answer started waiting for the caller's final (--answer-after-final)
        self.state = {"speaking": False, "utt": 0, "buf": "", "at": 0.0, "held": None,
                      "text": "", "go": False, "wait": None, "nw": 0}

    def hear(self, d, ap, t):
        (self.echo if self.a.echo_filter else self.heard).delta(d, ap, t)

    def hear_final(self, text, t):
        (self.echo if self.a.echo_filter else self.heard).final(text, t)

    def warm(self):
        # warm the prefix cache so the first tick is not a cold prompt
        a = self.a
        llm(a.llm, self.system, NOVOICE, a.threshold, a.sym_threshold, a.lora_scale)

    def flush(self, more, upto=None):
        state = self.state
        if state["wait"] is not None:
            return
        text = state["buf"][:upto].strip()
        if text:
            self.spk.say(state["utt"], text, more, state["at"])
            state["buf"] = state["buf"][len(state["buf"][:upto]):].strip()
        if not state["speaking"] and not state["buf"]:
            self.spk.done(state["utt"])

    def room(self):
        return self.a.cap is None or self.spk.left() < self.a.cap

    def nwords(self):
        return sum(not w["cont"] for w in self.heard.words)

    def ready(self):
        heard = self.heard
        if not heard.words:
            return True
        w = heard.words[-1]
        return heard.final_at > w["first"] or self.now() - w["last"] >= self.a.final_wait

    def hold(self):
        # an answer waits for the end of the caller's turn; backchannels and warnings do not
        state = self.state
        if not self.a.answer_after_final or state["go"] or len(state["text"].split()) <= 2:
            return False
        if state["wait"] is None:
            if HAZARD.match(state["text"]) or self.ready():
                state["go"] = True
                return False
            state["wait"] = self.now()
        return True

    def check(self):
        state = self.state
        if state["wait"] is None:
            return
        took = round(self.now() - state["wait"], 3)
        if self.ready():
            self.log("held_answer", text=state["buf"], hold=took)
            state.update(wait=None, go=True)
            self.flush(state["speaking"] and not ends_sentence(state["buf"]))
        elif self.nwords() > state["nw"]:
            # the caller went on: it stays in the model's history only, and what the model says next
            # waits again
            self.log("dropped_answer", text=state["buf"], hold=took)
            state.update(wait=None, buf="", nw=self.nwords())

    async def tick(self, n, t):
        a, state, heard, spk, log = self.a, self.state, self.heard, self.spk, self.log
        if a.echo_filter:
            self.echo.sift(t)
        h = heard.take(t)
        said, p, again, held, pn = await asyncio.to_thread(llm, a.llm, self.prefix, h, a.threshold,
                                                            a.sym_threshold, a.lora_scale, self.prev,
                                                            a.reask_mid_answer_only, 32 if a.max_words > 3 else 16)
        self.prev = said
        ms = (self.now() - t) * 1000
        self.prefix += f"<|im_start|>user\n{h}<|im_end|>\n<|im_start|>assistant\n{said}<|im_end|>\n"
        rec = {"n": n, "at": t, "heard": h, "said": said, "p_listen": round(p, 3), "again": again,
               "held": held, "ms": round(ms, 1), "prompt_n": pn}
        log("tick", **rec)
        self.ticks.append(rec)
        self.check()

        if said == LISTEN and state["held"]:
            # dropped from the audio only, the model's history keeps it
            log("dropped_bc", text=state["held"]["text"], why=state["held"]["why"])
            state["held"] = None
        if said != LISTEN:
            # the server sometimes decodes a curly apostrophe as U+FFFD
            text = said.replace("�", "'")
            rec["utt"] = state["utt"] + (not state["speaking"])
            if not state["speaking"]:
                self.flush(False)
                state.update(speaking=True, utt=state["utt"] + 1, text=text, go=False)
                if state["wait"] is None:
                    state["nw"] = self.nwords()
                # a short reply reaches the TTS a tick after the phrase it answers; skip it if the
                # caller has already gone on, unless it is a plain acknowledgement, which is what
                # a listener says over someone anyway
                new = sum(bool(re.search(r"\w", w)) for w in heard.text[heard.sent:].split())
                if a.bc_gate and len(text.split()) <= 2 and new >= a.bc_words and not BACKCHANNEL.match(text.strip()):
                    why = "mid-word" if heard.open is not None else "new word"
                    state["held"] = {"text": text, "at": t, "why": why}
                    log("bc_hold", text=text, why=why)
                elif self.hold():
                    if not state["buf"]:
                        state["at"] = t
                    state["buf"] = (state["buf"] + " " + text).strip()
                else:
                    spk.say(state["utt"], text, not ends_sentence(text), t)
            elif state["held"]:
                state["text"] += " " + text
                h2 = state["held"]
                h2["text"] += " " + text
                if len(h2["text"].split()) > 2:
                    state["held"] = None
                    if self.hold():
                        state.update(buf=h2["text"], at=h2["at"])
                    else:
                        spk.say(state["utt"], h2["text"], not ends_sentence(h2["text"]), h2["at"])
            else:
                state["text"] += " " + text
                if not state["buf"]:
                    state["at"] = t
                state["buf"] = (state["buf"] + " " + text).strip()
                self.hold()
                # every stream() call restarts the voice's pitch and rhythm, so later words wait until the
                # track is about to run out and whatever has queued up by then goes in one call
                self.pump()
        elif h != NOVOICE and re.search(r"\w", h) and (spk.left() > 0 or spk.busy or state["buf"]) \
                and not BACKCHANNEL.match(state["text"].strip()):
            # the caller is talking and the model chose to listen: barge-in. An acknowledgement is
            # meant to land while the caller talks, so it plays out instead of being cut
            state.update(speaking=False, buf="", wait=None)
            spk.cut(f"tick {n}")
        elif state["speaking"]:
            state["speaking"] = False
            if self.room():
                self.flush(False)
            elif not state["buf"]:
                self.spk.done(state["utt"])
        return rec

    def pump(self):
        state, spk = self.state, self.spk
        self.check()
        if spk.grow and state["buf"]:
            # one growing stream per utterance: words go in as they come, there is no splice to place
            self.flush(state["speaking"] and not ends_sentence(state["buf"]))
            return
        if not state["buf"] or spk.busy:
            return
        if spk.left() < LOW:
            # mid-answer, stop at the last clause end so the splice lands where speech pauses anyway;
            # the rest waits for its clause to finish or for the next low point
            upto = None
            if state["speaking"]:
                upto = max(state["buf"].rfind(x) for x in ",.?!;:") + 1 or None
            self.flush(state["speaking"] and not ends_sentence(state["buf"][:upto]), upto)
        elif self.a.cap is not None and spk.left() < self.a.cap:
            # held words: release up to the last clause boundary
            cut = max(state["buf"].rfind(x) for x in ",.?!;:")
            if cut >= 0:
                self.flush(not ends_sentence(state["buf"][:cut + 1]), cut + 1)

    def finish(self):
        self.state["wait"] = None
        self.flush(False)
        self.spk.done(self.state["utt"])


async def call(a):
    events = []
    lock = threading.Lock()
    t0 = None

    def now():
        return time.perf_counter() - t0

    def log(ev, **kw):
        with lock:
            events.append({"t": round(now(), 4), "ev": ev, **kw})

    caller = None
    if a.wav:
        with wave.open(a.wav) as w:
            assert w.getframerate() == SR_IN and w.getnchannels() == 1 and w.getsampwidth() == 2
            caller = w.readframes(w.getnframes())
        seconds = len(caller) / (2 * SR_IN) + 30
    else:
        seconds = a.seconds + 30

    t0 = time.perf_counter()
    spk = Speaker(seconds, now, log, a.tts, a.voice, device=a.tts_device)
    heard = Heard(log, a.tick if a.dot else None)
    turns = Turns(a, spk, heard, log, now)
    turns.warm()
    mic_in = []
    done = asyncio.Event()

    async with websockets.connect(a.stt, max_size=None) as ws:
        json.loads(await ws.recv())
        session = {"language": "en-US"}
        if a.endpoint_ms:
            session["endpointing_ms"] = a.endpoint_ms
        await ws.send(json.dumps({"type": "session.update", "session": session}))
        json.loads(await ws.recv())

        async def rx():
            async for m in ws:
                e = json.loads(m)
                if e["type"].endswith(".delta"):
                    if e["delta"]:
                        log("stt", delta=e["delta"], ap=e["audio_processed"])
                    turns.hear(e["delta"], e["audio_processed"], now())
                elif e["type"].endswith(".completed"):
                    log("final", text=e["transcript"])
                    turns.hear_final(e["transcript"], now())
                elif e["type"] == "input_audio_buffer.committed":
                    return

        async def send_wav():
            step = int(SR_IN * FRAME) * 2
            for k, i in enumerate(range(0, len(caller), step)):
                await asyncio.sleep(max(0, (k + 1) * FRAME - now()))
                await ws.send(caller[i:i + step])

        async def send_mic():
            import sounddevice as sd
            loop = asyncio.get_running_loop()
            q = asyncio.Queue()
            pos = [0]

            def on_in(data, frames, ti, status):
                loop.call_soon_threadsafe(q.put_nowait, bytes(data))

            def on_out(out, frames, ti, status):
                i = pos[0]
                chunk = spk.track[i:i + frames]
                out[:len(chunk), 0] = chunk
                out[len(chunk):, 0] = 0
                pos[0] += frames

            with sd.RawInputStream(samplerate=SR_IN, channels=1, dtype="int16", blocksize=int(SR_IN * FRAME),
                                   callback=on_in), \
                    sd.OutputStream(samplerate=SR_OUT, channels=1, dtype="float32", callback=on_out):
                while now() < a.seconds:
                    b = await q.get()
                    mic_in.append(b)
                    await ws.send(b)

        async def ticker():
            n = 1
            while not done.is_set():
                await asyncio.sleep(max(0, n * a.tick - now()))
                if done.is_set():
                    break
                await turns.tick(n, now())
                n = max(n + 1, int(now() / a.tick) + 1)

        async def pump():
            while not done.is_set():
                await asyncio.sleep(0.04)
                turns.pump()

        t0 = time.perf_counter()
        tasks = [asyncio.create_task(x()) for x in (rx, ticker, pump)]
        await (send_wav() if a.wav else send_mic())
        done.set()
        await tasks[1]
        await tasks[2]
        turns.finish()
        while spk.busy or spk.left() > 0:
            await asyncio.sleep(0.1)
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        await asyncio.wait_for(tasks[0], 10)

    total = now()
    if mic_in:
        caller = b"".join(mic_in)
    pcm = np.frombuffer(caller, np.int16).astype(np.float32) / 32768
    left = resample_poly(pcm, 3, 2).astype(np.float32)
    right = spk.track[:int(max(total, len(left) / SR_OUT) * SR_OUT)]
    n = max(len(left), len(right))
    st = np.zeros((n, 2), np.float32)
    st[:len(left), 0] = left
    st[:len(right), 1] = right
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out.with_suffix(".wav")), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR_OUT)
        w.writeframes((np.clip(st, -1, 1) * 32767).astype(np.int16).tobytes())
    with open(out.with_suffix(".jsonl"), "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    return events, turns.ticks, heard.words, right


def norm(s):
    return re.sub(r"[^a-z0-9']", "", s.lower())


def measure(events, ticks, words, agent, truth):
    tw = truth["words"]
    # merge pieces that reopened a word after an idle close
    stt = []
    for w in words:
        if w.get("cont") and stt:
            stt[-1] = {**w, "text": stt[-1]["text"] + w["text"], "first": stt[-1]["first"]}
        else:
            stt.append(w)
    sm = difflib.SequenceMatcher(a=[norm(w["text"]) for w in tw], b=[norm(w["text"]) for w in stt], autojunk=False)
    match = {}
    for i, j, n in sm.get_matching_blocks():
        for k in range(n):
            match[i + k] = stt[j + k]

    tts_ev = [e for e in events if e["ev"] == "tts"]
    cuts = [e for e in events if e["ev"] == "cut"]
    first = {}
    for e in tts_ev:
        first.setdefault(e["utt"], e)
    utts = []
    for u, e in sorted(first.items()):
        tk = next(t for t in ticks if t.get("utt") == u)
        said = " ".join(t["said"] for t in ticks if t.get("utt") == u)
        utts.append({"utt": u, "audio": e["audio"], "tick": tk, "said": said, "tts": e})

    # caller speech: word intervals with short gaps merged
    segs = []
    for w in tw:
        if segs and w["start"] - segs[-1][1] < 0.3:
            segs[-1][1] = max(segs[-1][1], w["end"])
        else:
            segs.append([w["start"], w["end"]])
    hop = int(0.02 * SR_OUT)
    rms = np.sqrt(np.mean(agent[:len(agent) // hop * hop].reshape(-1, hop) ** 2, axis=1))
    talk = rms > 0.01
    tt = np.arange(len(talk)) * 0.02
    caller_on = np.zeros(len(talk), bool)
    for s, e in segs:
        caller_on |= (tt >= s) & (tt < e)
    over = float(np.sum(talk & caller_on) * 0.02)

    for u in utts:
        fa = u["audio"]
        prev = [i for i, w in enumerate(tw) if w["end"] <= fa]
        inside = any(w["start"] <= fa < w["end"] for w in tw)
        nxt = [w["start"] for w in tw if w["start"] > fa]
        if not prev:
            u["kind"] = "before caller"
            continue
        i = prev[-1]
        gap = (nxt[0] if nxt else truth["duration"]) - tw[i]["end"]
        u["after_word"], u["ttfw"] = tw[i]["text"], fa - tw[i]["end"]
        u["kind"] = "over speech" if inside or u["ttfw"] < 0 else (
            "answer" if gap >= 2 else "in pause" if gap >= 0.8 else "over speech")
        s = match.get(i)
        if u["kind"] == "answer" and s and s.get("closed") is not None and "tick" in s:
            t = u["tick"]
            u["parts"] = {"stt": s["closed"] - tw[i]["end"], "tick_wait": s["tick"] - s["closed"],
                          "extra_ticks": t["at"] - s["tick"], "llm": t["ms"] / 1000,
                          "tts": fa - (t["at"] + t["ms"] / 1000), "how": s["how"]}
            if u["parts"]["extra_ticks"] < 0:
                # spoke before the last word reached it: a filler that happened to land late
                u["kind"] = "early"

    # end-of-turn gaps the agent never answered
    missed = []
    for k in range(len(segs)):
        end = segs[k][1]
        nxt = segs[k + 1][0] if k + 1 < len(segs) else truth["duration"]
        if nxt - end >= 2 and not any(end <= u["audio"] < nxt for u in utts):
            missed.append(round(end, 2))

    res = {"answers": [], "other": [], "missed_gaps": missed, "talk_over_s": round(over, 2)}
    for u in utts:
        r = {"at": round(u["audio"], 2), "kind": u["kind"], "said": u["said"],
             "ttfw": round(u.get("ttfw", 0), 3), "after": u.get("after_word")}
        if "parts" in u:
            r["parts"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in u["parts"].items()}
        (res["answers"] if u["kind"] == "answer" else res["other"]).append(r)

    if tts_ev:
        res["tts"] = {"jobs": len(tts_ev), "queued_max": max(e["queued"] for e in tts_ev),
                      "lag_max": max(e["lag"] for e in tts_ev),
                      "lag_median": statistics.median(e["lag"] for e in tts_ev)}
        res["tts"]["synth_ms_median"] = round(statistics.median(e["gen"] - e["req"] for e in tts_ev) * 1000)
    res["cuts"] = len(cuts)
    res["dropped_bc"] = [{"t": e["t"], "text": e["text"], "why": e["why"]} for e in events if e["ev"] == "dropped_bc"]
    for k in ("held_answer", "dropped_answer"):
        res[k] = [{"t": e["t"], "text": e["text"], "hold": e["hold"]} for e in events if e["ev"] == k]
    lat = sorted(t["ms"] for t in ticks)
    res["ticks"] = {"n": len(ticks), "speaking": sum(t["said"] != LISTEN for t in ticks),
                    "llm_ms_median": statistics.median(lat), "llm_ms_p95": lat[math.ceil(0.95 * len(lat)) - 1],
                    "llm_ms_max": lat[-1], "re_requests": sum(t["again"] for t in ticks),
                    "held": sum(bool(t.get("held")) for t in ticks),
                    "prompt_n_median": statistics.median([t["prompt_n"] for t in ticks if t["prompt_n"]] or [0])}

    def word_at(text):
        return next((w for w in tw if norm(w["text"]) == text), None)

    hz = word_at("touch")
    if hz:
        u = next((u for u in utts if u["audio"] > hz["start"] - 4), None)
        res["hazard"] = {"touch_start": round(hz["start"], 2), "touch_end": round(hz["end"], 2),
                         "first_audio_vs_touch_end": round(u["audio"] - hz["end"], 2) if u else None,
                         "said": u["said"] if u else None}
    bw = word_at("wait")
    if bw:
        playing = bool(talk[int(bw["start"] / 0.02):int(bw["start"] / 0.02) + 5].any())
        c = next((c for c in cuts if c["t"] > bw["start"]), None)
        after = [u for u in utts if u["audio"] > (c["t"] if c else bw["end"])]
        i = tw.index(bw)
        heard_at = match[i].get("tick") if i in match else None
        res["barge"] = {"wait_start": round(bw["start"], 2), "agent_talking": playing,
                        "wait_sent_tick": round(heard_at, 2) if heard_at else None,
                        "cut_at": round(c["t"], 2) if c else None,
                        "cut_after_wait_s": round(c["t"] - bw["start"], 2) if c else None,
                        "dropped_s": round(c["dropped"], 2) if c else None,
                        "next": {"at": round(after[0]["audio"], 2), "said": after[0]["said"]} if after else None}
    return res


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--wav", help="16 kHz mono WAV played as the caller in real time")
    g.add_argument("--mic", action="store_true")
    p.add_argument("--seconds", type=float, default=120, help="call length in mic mode")
    p.add_argument("--tick", type=float, default=0.6)
    p.add_argument("--prompt-tick", type=float, default=0.6, help="tick length written in the system prompt")
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--sym-threshold", action="store_true",
                   help="also listen when the model picked a word but p_listen is at or over the threshold")
    p.add_argument("--reask-mid-answer-only", action=argparse.BooleanOptionalAction, default=True,
                   help="only re-ask when the model was speaking on the previous tick")
    p.add_argument("--bc-gate", action=argparse.BooleanOptionalAction, default=True,
                   help="drop a reply of two words or less when the caller has said more since its tick")
    p.add_argument("--bc-words", type=int, default=1,
                   help="new caller words since the reply's tick that make the backchannel gate hold it")
    p.add_argument("--echo-filter", action=argparse.BooleanOptionalAction, default=True,
                   help="drop STT words that match what the agent is saying (its voice leaking into the mic)")
    p.add_argument("--answer-after-final", action=argparse.BooleanOptionalAction, default=True,
                   help="start a reply of more than two words only after the STT final for the caller's turn")
    p.add_argument("--final-wait", type=float, default=1.0,
                   help="with --answer-after-final, caller silence that counts as a final when none arrives")
    p.add_argument("--lora-scale", type=float, help="per-request scale for the server's first --lora adapter")
    p.add_argument("--dot", action="store_true", help='send "." after a silent tick that follows a bare word')
    p.add_argument("--endpoint-ms", type=int, default=500,
                   help="STT endpointing silence (the server needs endpointing enabled), 0 for none")
    p.add_argument("--cap", type=float, help="hold the model's words while more than this much audio is queued")
    p.add_argument("--max-words", type=int, default=3, help="words per tick the system prompt allows (2 to 8)")
    p.add_argument("--persona", help="text file added to the system prompt as a second paragraph")
    p.add_argument("--tts", choices=["pytorch", "grow"], default="grow",
                   help="grow: one Pocket stream per utterance that takes words as they come (tts_grow.py); "
                        "pytorch: a new stream per clause")
    p.add_argument("--tts-device", default="cpu", help="cpu, or a CUDA device index or UUID")
    p.add_argument("--voice", default="anna", help="Pocket TTS voice: a built-in name or a .safetensors state")
    p.add_argument("--llm", default="http://127.0.0.1:8080")
    p.add_argument("--stt", default="ws://127.0.0.1:9100/v1/audio/transcriptions/realtime")
    p.add_argument("--out", default="results/call")
    a = p.parse_args()
    if a.persona:
        a.persona = Path(a.persona).read_text(encoding="utf-8").strip()

    events, ticks, words, agent = asyncio.run(call(a))
    for t in ticks:
        if t["heard"] != NOVOICE or t["said"] != LISTEN:
            print(f"{t['at']:6.2f}  {t['heard']:<28} {t['said']:<26} p={t['p_listen']:.2f} {t['ms']:4.0f} ms")
    truth = Path(a.wav).with_suffix(".json") if a.wav else None
    if truth and truth.exists():
        res = measure(events, ticks, words, agent, json.load(open(truth)))
        res["args"] = vars(a)
        Path(a.out).with_suffix(".json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(json.dumps({k: v for k, v in res.items() if k != "args"}, indent=1))


if __name__ == "__main__":
    main()
