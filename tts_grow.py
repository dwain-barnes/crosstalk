"""Speak an utterance whose text is still growing, as one continuous stream.

Pocket TTS reads the whole text as a prefix before the first audio frame, so it
cannot take more words once it has started. Instead, each time words arrive we
start again from the voice prompt with the longer text, force-feed the latents
already released, and keep sampling from there. The model conditions on its own
earlier audio, so pitch and rhythm carry on. Mimi only ever sees released
latents, in order, so one decoder stream lasts the whole utterance.

The model winds down at the end of whatever text it has, so when the text stops
mid-phrase the last `hold` frames are kept back until more words arrive (then
they are thrown away and regenerated) or close() is called. Text that ends on a
comma or a full stop ends where the voice would pause anyway, so it all goes out.

    g = tts_grow.open(m, voice)
    g.feed("That sounds")           # from the LLM thread, as often as words come
    for pcm in g.pull():            # float32 at 24 kHz, ends once close() is flushed
        play(pcm)
"""
import copy
import queue
import threading
import time

import torch
from pocket_tts.models.text_chunking import prepare_text_prompt
from pocket_tts.modules.stateful_module import increment_steps, init_states


class Grow:
    def __init__(self, m, voice, hold=4, fresh=8):
        self.m, self.voice, self.hold, self.fresh = m, voice, hold, fresh
        self.text = ""
        self.closed = False
        self.stopped = False
        self.dirty = False
        self.wake = threading.Event()
        self.lock = threading.Lock()
        self.out = queue.Queue()
        self.pad = None
        self.lat = []           # latents of the current attempt; the first `sent` are released
        self.sent = 0
        self.eos = None
        self.joins = []         # frame index of each re-prefix after the first
        self.prefills = []      # (text tokens, forced latents, seconds)
        self.steps = 0
        self.cpu = 0.0          # CPU seconds of the generation thread, set when it ends
        threading.Thread(target=self.run, daemon=True).start()

    def feed(self, text):
        with self.lock:
            self.text = f"{self.text} {text}".strip()
            self.dirty = True
        self.wake.set()

    def close(self):
        with self.lock:
            self.closed = True
            self.dirty = True
        self.wake.set()

    def stop(self):
        # barge-in: stop generating and drop whatever has not been pulled yet
        self.stopped = True
        self.wake.set()

    def pull(self):
        m = self.m
        steps = int(m.mimi.encoder_frame_rate / m.mimi.frame_rate)
        mimi = init_states(m.mimi, batch_size=1, sequence_length=60 * 13 * steps)
        done = False
        while not done:
            lat = [self.out.get()]
            if self.stopped:
                return
            # decode whatever else is already waiting in the same call, it is cheaper per frame
            while not self.out.empty():
                lat.append(self.out.get())
            if lat[-1] is None:
                done = True
                lat.pop()
            if not lat:
                break
            with torch.no_grad():
                x = torch.cat(lat, dim=1) * m.flow_lm.emb_std + m.flow_lm.emb_mean
                pcm = m.mimi.decode_from_latent(x, mimi)
                increment_steps(m.mimi, mimi, increment=steps * len(lat))
            yield pcm[0, 0].float().cpu().numpy()

    def prefill(self, text, closed):
        m = self.m
        if self.pad is None:
            # decide the short-input padding once, so the text prefix stays the same as it grows
            self.pad = m.pad_with_spaces_for_short_inputs and len(text.split()) < 5
        text, after = prepare_text_prompt(text, self.pad, m.remove_semicolons, closed,
                                          m.capitalize_first_letter, m.replace_characters)
        self.after = m.model_recommended_frames_after_eos or after + 2
        t0 = time.perf_counter()
        st = copy.deepcopy(self.voice)
        tok = m.flow_lm.conditioner.prepare(text)
        self.max_len = m._estimate_max_gen_len(tok.shape[1]) + self.sent
        m._expand_kv_cache(st, m._flow_lm_current_end(st) + tok.shape[1] + self.max_len + 1)
        bos = torch.full((1, 1, m.flow_lm.ldim), float("nan"), dtype=m.flow_lm.dtype, device=tok.device)
        # BOS and every released latent go in as one step; the model samples from the last position
        seq = torch.cat([bos] + self.lat[:self.sent], dim=1)
        nxt, eos = m._run_flow_lm_and_increment_step(st, text_tokens=tok, backbone_input_latents=seq)
        self.prefills.append((tok.shape[1], self.sent, time.perf_counter() - t0))
        if self.lat:
            self.joins.append(self.sent)
        self.st = st
        self.soft = text.rstrip()[-1] in ",;:.?!"
        self.lat = self.lat[:self.sent]
        self.start = self.sent
        self.eos = None
        self.add(nxt, eos)

    def add(self, nxt, eos):
        n = len(self.lat)
        # after a rollback into a pause the EOS head can fire before the new words are said ("that!" then
        # "Please"), which left nothing to play until the next fragment. Text that ends mid-phrase gets
        # `fresh` frames first
        fresh = 0 if self.soft else self.fresh
        if eos.item() and self.eos is None and n >= max(self.m._MIN_FRAMES_BEFORE_EOS, self.start + fresh):
            self.eos = n
        self.lat.append(nxt)

    def release(self, n):
        for x in self.lat[self.sent:n]:
            self.out.put(x)
        self.sent = max(self.sent, n)

    @torch.no_grad
    def run(self):
        started = False
        while True:
            with self.lock:
                dirty, text, closed = self.dirty, self.text, self.closed
                self.dirty = False
                self.wake.clear()
            if self.stopped:
                break
            if dirty and text:
                self.prefill(text, closed)
                started = True
            if not started and closed:
                break
            if not started:
                self.wake.wait()
                continue
            if self.eos is None:
                self.release(len(self.lat) - self.hold)
            elif not closed:
                self.release(self.eos - (0 if self.soft else self.hold))
            if closed and (len(self.lat) >= self.max_len
                           or self.eos is not None and len(self.lat) >= self.eos + self.after):
                self.release(self.eos + self.after if self.eos is not None else len(self.lat))
                break
            # with more text to come there is no point running past EOS
            if self.eos is not None and not closed or len(self.lat) >= self.max_len:
                self.wake.wait()
                continue
            nxt, eos = self.m._run_flow_lm_and_increment_step(self.st, backbone_input_latents=self.lat[-1])
            self.add(nxt, eos)
            self.steps += 1
        self.cpu = time.thread_time()
        self.out.put(None)


def open(m, voice, hold=4, fresh=8):
    return Grow(m, voice, hold, fresh)
