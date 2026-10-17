const $ = (id) => document.getElementById(id);
const LEAD = 0.1;      // seconds of jitter buffer before agent audio plays
const VOICED = 0.01;   // server's speech threshold, shown on the meter

let ws = null;
let inCall = false;
let saving = false;
let mic = null;        // { ctx, stream, node }
let play = null;       // AudioContext for the agent
let next = 0;          // when the last scheduled chunk ends
let off = null;        // play.currentTime minus server clock, smallest seen
let sources = [];
let cuts = 0;
let callTime = 0;
let cutUntil = 0;

const stage = $("stage");
const halo = stage.querySelector(".halo");
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)");

function db(x) { return 20 * Math.log10(Math.max(x, 1e-6)); }
function pct(x) { return Math.max(0, Math.min(100, (db(x) + 60) / 60 * 100)); }
function say(text) { $("msg").textContent = text || ""; }
function clock(t) {
  const s = Math.max(0, Math.floor(t));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}
function flags() {
  return { reask: $("reask").checked, bc_gate: $("bc_gate").checked, after_final: $("after_final").checked,
           threshold: parseFloat($("threshold").value) };
}

for (const m of document.querySelectorAll(".meter .thr")) m.style.left = pct(VOICED) + "%";

function setTheme(t) {
  const dark = t ? t === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  if (t) document.documentElement.dataset.theme = t;
  $("theme").setAttribute("aria-pressed", dark);
  $("theme").setAttribute("aria-label", dark ? "Light theme" : "Dark theme");
}
setTheme(document.documentElement.dataset.theme);
$("theme").onclick = () => {
  const t = $("theme").getAttribute("aria-pressed") === "true" ? "light" : "dark";
  setTheme(t);
  try { localStorage.setItem("theme", t); } catch {}
};

async function listDevices() {
  const all = await navigator.mediaDevices.enumerateDevices();
  for (const [sel, kind] of [[$("mic"), "audioinput"], [$("out"), "audiooutput"]]) {
    const keep = sel.value;
    sel.innerHTML = "";
    for (const d of all.filter((d) => d.kind === kind)) {
      sel.add(new Option(d.label || `${kind === "audioinput" ? "Microphone" : "Output"} ${sel.length + 1}`, d.deviceId));
    }
    if ([...sel.options].some((o) => o.value === keep)) sel.value = keep;
  }
  $("out").disabled = !("setSinkId" in AudioContext.prototype);
}

function showLevel(rms, peak) {
  for (const m of [$("meter"), $("meter2")]) {
    m.querySelector(".fill").style.width = pct(rms) + "%";
    m.querySelector(".peak").style.left = pct(peak) + "%";
  }
  $("db").textContent = `${db(rms).toFixed(0)} dB`;
  halo.style.setProperty("--lvl", (pct(rms) / 100) ** 2);
}

async function openMic() {
  if (mic) {
    mic.stream.getTracks().forEach((t) => t.stop());
    await mic.ctx.close();
    mic = null;
  }
  const id = $("mic").value;
  const stream = await navigator.mediaDevices.getUserMedia({ audio: {
    deviceId: id ? { exact: id } : undefined, channelCount: 1,
    echoCancellation: $("ec").checked, noiseSuppression: $("ns").checked, autoGainControl: $("agc").checked } });
  const ctx = new AudioContext({ sampleRate: 16000 });
  if (ctx.sampleRate !== 16000) throw new Error("this browser would not open a 16 kHz audio context; use Edge or Chrome");
  await ctx.audioWorklet.addModule("mic.js");
  const node = new AudioWorkletNode(ctx, "mic");
  ctx.createMediaStreamSource(stream).connect(node);
  node.connect(ctx.destination);   // the worklet outputs silence; this keeps it pulled
  node.port.postMessage({ gain: 10 ** ($("gain").value / 20) });
  node.port.onmessage = (e) => {
    const { pcm, rms, peak } = e.data;
    showLevel(rms, peak);
    if (inCall && ws && ws.readyState === WebSocket.OPEN) ws.send(pcm);
  };
  mic = { ctx, stream, node };
  await listDevices();
  if (!id) $("mic").value = stream.getAudioTracks()[0].getSettings().deviceId || "";
}

async function openPlayback() {
  if (!play) play = new AudioContext();
  await play.resume();
  const id = $("out").value;
  if (play.setSinkId && id && play.sinkId !== id) await play.setSinkId(id);
}

function playChunk(buf) {
  const dv = new DataView(buf);
  const start = dv.getFloat64(0, true);
  const sent = dv.getFloat64(8, true);
  const pcm = new Int16Array(buf, 16);
  const ab = play.createBuffer(1, pcm.length, 24000);
  const ch = ab.getChannelData(0);
  for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768;
  const o = play.currentTime - sent;
  if (off === null || o < off) off = o;
  // keep the server's spacing between chunks, never overlap, never schedule in the past
  const when = Math.max(start + off + LEAD, next, play.currentTime);
  const s = play.createBufferSource();
  s.buffer = ab;
  s.connect(play.destination);
  s.start(when);
  next = when + ab.duration;
  sources.push(s);
  s.onended = () => { sources = sources.filter((x) => x !== s); };
}

function flush() {
  for (const s of sources) s.stop();
  sources = [];
  next = 0;
}

function queued() { return play ? Math.max(0, next - play.currentTime) : 0; }

function showState() {
  const open = ws && ws.readyState === WebSocket.OPEN;
  const st = !inCall ? "idle" : queued() > 0.02 ? "speaking" : "listening";
  if (stage.dataset.state !== st) stage.dataset.state = st;
  let label;
  if (saving) label = "Saving the call";
  else if (!open) label = "Connecting";
  else if (performance.now() < cutUntil) label = "You cut in";
  else label = { idle: "Ready", listening: "Listening", speaking: "Crosstalk is speaking" }[st];
  if ($("state").textContent !== label) $("state").textContent = label;
  if (play) $("s_q").textContent = `${queued().toFixed(2)} s`;
}
setInterval(showState, 100);

function flashCut() {
  cutUntil = performance.now() + 900;
  stage.classList.remove("cut");
  void stage.offsetWidth;   // restart the animation
  stage.classList.add("cut");
  setTimeout(() => stage.classList.remove("cut"), 900);
}

function pill(id, up) {
  const p = $(id);
  p.classList.toggle("up", up === true);
  p.classList.toggle("down", up === false);
  p.querySelector("[data-state]").textContent = up === true ? "up" : up === false ? "down" : "unknown";
}

function modelName(s) {
  return s.replace(/^crosstalk-/, "").replace(/^run(\d)/, "run $1");
}

const log = $("log");
const live = $("live");
let seg = [];          // caller entries of the current STT segment
let callerEl = null;   // where the next caller delta goes
let agentEl = null;
let agentUtt = -1;
const agents = {};     // utt -> element
let heldText = null;
const deltas = new Map();   // STT audio position -> its span, for marking echo afterwards
let unread = false;
let speak = null;      // agent reply waiting to be read out by screen readers

function nearBottom() { return log.scrollHeight - log.scrollTop - log.clientHeight < 60; }

function stick(near) {
  if (near) log.scrollTop = log.scrollHeight;
  else if (!unread) { unread = true; $("more").hidden = false; }
}

log.addEventListener("scroll", () => {
  if (unread && nearBottom()) { unread = false; $("more").hidden = true; }
});
$("more").onclick = () => {
  log.scrollTo({ top: log.scrollHeight, behavior: reduceMotion.matches ? "auto" : "smooth" });
  $("more").hidden = true;
  unread = false;
};

// grow the element and keep the view pinned to the bottom if it was already there
function grow(fn) {
  const near = nearBottom();
  fn();
  stick(near);
}

function entry(cls, t) {
  $("empty").hidden = true;
  const row = document.createElement("div");
  row.className = "msgrow " + cls;
  const b = document.createElement("div");
  b.className = "bubble";
  const tm = document.createElement("time");
  tm.textContent = clock(t);
  row.append(b, tm);
  row.body = b;
  grow(() => log.append(row));
  return row;
}

function tag(el, text, cls, title) {
  const s = document.createElement("span");
  s.className = "chip " + (cls || "");
  s.textContent = text;
  if (title) s.title = title;
  grow(() => el.chips.append(s));
  return s;
}

function callerDelta(d, t, ap) {
  if (!callerEl) {
    callerEl = entry("caller", t);
    seg.push(callerEl);
  }
  let p = callerEl.body.querySelector(".partial");
  if (!p) {
    p = document.createElement("span");
    p.className = "partial";
    callerEl.body.append(p);
  }
  const w = document.createElement("span");
  w.textContent = d;
  deltas.set(ap, w);
  grow(() => p.append(w));
}

function callerFinal(text, t) {
  const parts = seg.flatMap((el) => [...el.body.querySelectorAll(".partial")]);
  const had = parts.map((p) => p.textContent).join("").trim();
  parts.forEach((p) => p.classList.remove("partial"));
  if (!parts.length && text.trim()) {
    callerDelta(text, t);
    callerEl.body.querySelector(".partial").classList.remove("partial");
  } else if (text.startsWith(had) && text.length > had.length && parts.length) {
    parts[parts.length - 1].append(text.slice(had.length));
  }
  seg = [];
  callerEl = null;
}

function announce() {
  if (speak && speak.words.textContent.trim()) live.textContent = speak.words.textContent.trim();
  speak = null;
}

function onTick(m) {
  $("s_n").textContent = m.n;
  $("s_ms").textContent = `${Math.round(m.ms)} ms`;
  if (m.heard !== "<no voice>" || m.said !== "<listen>") {
    const at = m.at.toFixed(1).padStart(5);
    const line = `${at}  ${m.heard.padEnd(30)} ${m.said.padEnd(24)} p=${m.p_listen.toFixed(2)} ${Math.round(m.ms)} ms${m.again ? " re-asked" : ""}`;
    const tk = $("ticks");
    const near = tk.scrollHeight - tk.scrollTop - tk.clientHeight < 30;
    tk.textContent += line + "\n";
    if (near) tk.scrollTop = tk.scrollHeight;
  }
  if (m.said === "<listen>") { announce(); return; }
  if (m.utt !== agentUtt) {
    announce();
    agentUtt = m.utt;
    agentEl = entry("agent", m.at);
    agentEl.words = document.createElement("div");
    agentEl.chips = document.createElement("div");
    agentEl.chips.className = "chips";
    agentEl.body.append(agentEl.words, agentEl.chips);
    agents[m.utt] = agentEl;
    callerEl = null;   // later caller words go below this reply
    speak = agentEl;
  }
  const words = document.createElement("span");
  words.textContent = (agentEl.words.children.length ? " " : "") + m.said;
  if (heldText !== null && m.said === heldText) {
    words.className = "held";
    words.title = "Held by the backchannel gate";
    heldText = null;
  }
  grow(() => agentEl.words.append(words));
}

function showSummary(m) {
  const s = m.summary || {};
  const u = s.utterances || [];
  const after = u.filter((x) => x.kind === "after caller" && x.ttfw != null).map((x) => x.ttfw);
  const card = $("summary_tpl").content.firstElementChild.cloneNode(true);
  const name = m.dir.split(/[\\/]/).pop();
  const f = (k) => card.querySelector(`[data-k="${k}"]`);
  f("dur").textContent = clock(callTime);
  f("replies").textContent = u.length;
  f("ttfw").textContent = after.length ? `${(after.reduce((a, b) => a + b, 0) / after.length).toFixed(2)} s` : "-";
  f("ttfw").title = `${after.length} of ${u.length} replies came after you had stopped`;
  f("over").textContent = `${(s.talk_over_s ?? 0).toFixed(1)} s`;
  f("dir").textContent = name;
  f("dir").title = m.dir;
  f("wav").href = `calls/${encodeURIComponent(name)}/call.wav`;
  grow(() => log.append(card));
}

function onEvent(m) {
  switch (m.type) {
    case "hello":
      pill("p_stt", m.stt);
      pill("p_llm", m.llm);
      pill("p_tts", m.tts ?? true);
      if (m.model) { $("model").textContent = modelName(m.model); $("model").title = m.model; $("model").hidden = false; }
      $("call").disabled = false;
      say(m.busy ? "Another call is running on the server." : "");
      break;
    case "started":
      $("sub").textContent = "Recording to " + m.dir.split(/[\\/]/).pop();
      $("sub").title = m.dir;
      pill("p_stt", m.stt);
      if (!m.stt) say("STT not connected: typed lines only.");
      break;
    case "stt": callerDelta(m.delta, m.t, m.ap); break;
    case "final": callerFinal(m.text, m.t); break;
    case "typed": {
      callerFinal("", m.t);
      const el = entry("caller typed", m.t);
      el.body.append(m.text);
      el.title = "Typed";
      break;
    }
    case "tick": onTick(m); break;
    case "bc_hold": heldText = m.text; break;
    case "dropped_bc": {
      const el = agents[agentUtt];
      if (el) {
        el.body.querySelectorAll(".held").forEach((s) => s.classList.add("dropped"));
        tag(el, "dropped", "drop", `Backchannel dropped by the gate (${m.why})`);
      }
      break;
    }
    case "echo": {
      // the words were shown as they arrived; now that the tick has judged them, grey them out
      let row = null;
      for (const ap of m.aps) {
        const w = deltas.get(ap);
        if (!w) continue;
        w.classList.add("echoed");
        row = w.closest(".msgrow");
      }
      if (row && !row.querySelector(".chip.echo")) {
        if (!row.chips) {
          row.chips = document.createElement("div");
          row.chips.className = "chips";
          row.body.append(row.chips);
        }
        tag(row, "echo", "echo", `Taken for the agent's own voice, not sent to the model: ${m.text}`);
      }
      break;
    }
    case "held_answer":
      if (agentEl) tag(agentEl, `held ${m.hold.toFixed(2)} s`, "warn", `Waited ${m.hold.toFixed(2)} s for the final transcript`);
      break;
    case "dropped_answer":
      if (agentEl) tag(agentEl, "answer dropped", "drop", `Answer dropped: ${m.text}`);
      break;
    case "cut":
      flush();
      cuts++;
      $("s_cuts").textContent = cuts;
      flashCut();
      if (agentEl) tag(agentEl, "cut", "cut", `Cut at ${m.why}, ${m.dropped.toFixed(1)} s of audio dropped`);
      announce();
      break;
    case "ttfw": {
      const el = agents[m.utt];
      if (!el) break;
      if (m.kind === "after caller") tag(el, `${m.ttfw.toFixed(1)} s`, "lat", `First word ${m.ttfw.toFixed(2)} s after you stopped`);
      else tag(el, m.kind, m.kind === "over speech" ? "warn" : "");
      break;
    }
    case "stat":
      callTime = m.t;
      $("s_t").textContent = clock(m.t);
      $("s_over").textContent = `${m.over.toFixed(2)} s`;
      pill("p_stt", m.stt);
      break;
    case "saved":
      say("");
      announce();
      showSummary(m);
      $("sub").textContent = "";
      stopped();
      break;
    case "error": say(m.text); break;
  }
}

function connect() {
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onmessage = (e) => (e.data instanceof ArrayBuffer ? playChunk(e.data) : onEvent(JSON.parse(e.data)));
  ws.onclose = () => {
    if (inCall || saving) say("Lost the server mid-call. The recording up to now is saved on the server.");
    $("call").disabled = true;
    for (const id of ["p_stt", "p_llm", "p_tts"]) pill(id, null);
    stopped();
    setTimeout(connect, 2000);
  };
}

function setLine(on) {
  $("line").disabled = !on;
  $("send").disabled = !on;
  $("line").placeholder = on ? "Type as if spoken" : "Start a call to type a line";
}

function stopped() {
  inCall = false;
  saving = false;
  flush();
  stage.classList.remove("on");
  $("call_label").textContent = "Start";
  $("call").setAttribute("aria-label", "Start call");
  $("call").disabled = !(ws && ws.readyState === WebSocket.OPEN);
  setLine(false);
}

$("call").onclick = async () => {
  if (inCall) {
    ws.send(JSON.stringify({ type: "end" }));
    saving = true;
    $("call").disabled = true;
    $("call_label").textContent = "Saving";
    return;
  }
  say("");
  try {
    if (!mic) await openMic();
  } catch (e) {
    say(`No microphone (${e.message}). Typed lines only.`);
  }
  try {
    await openPlayback();
  } catch (e) {
    say("Output: " + e.message);
  }
  log.replaceChildren($("empty"));
  $("ticks").textContent = "";
  seg = []; callerEl = null; agentEl = null; agentUtt = -1; heldText = null; cuts = 0; off = null; speak = null;
  deltas.clear();
  callTime = 0;
  $("s_cuts").textContent = "0";
  $("s_t").textContent = "0:00";
  ws.send(JSON.stringify({ type: "start", ...flags() }));
  inCall = true;
  stage.classList.add("on");
  $("call_label").textContent = "End";
  $("call").setAttribute("aria-label", "End call");
  setLine(true);
  $("line").focus();
};

$("test").onclick = async () => {
  try { await openMic(); say(""); } catch (e) { say("Microphone: " + e.message); }
};

$("open_settings").onclick = () => $("settings").showModal();
$("settings").addEventListener("click", (e) => { if (e.target === $("settings")) $("settings").close(); });

$("mic").onchange = () => openMic().catch((e) => say("Microphone: " + e.message));
for (const id of ["agc", "ns", "ec"]) $(id).onchange = () => { if (mic) openMic().catch((e) => say(e.message)); };
$("out").onchange = () => { if (play) openPlayback().catch((e) => say("Output: " + e.message)); };
$("gain").oninput = () => {
  $("gainv").textContent = `+${$("gain").value} dB`;
  if (mic) mic.node.port.postMessage({ gain: 10 ** ($("gain").value / 20) });
};
$("threshold").oninput = () => { $("thrv").textContent = parseFloat($("threshold").value).toFixed(2); };
for (const id of ["reask", "bc_gate", "after_final", "threshold"]) {
  $(id).onchange = () => { if (inCall) ws.send(JSON.stringify({ type: "flags", ...flags() })); };
}

$("say").onsubmit = (e) => {
  e.preventDefault();
  const text = $("line").value.trim();
  if (!text || !inCall) return;
  ws.send(JSON.stringify({ type: "text", text }));
  $("line").value = "";
};

navigator.mediaDevices.addEventListener("devicechange", () => listDevices());
listDevices();
setLine(false);
connect();
