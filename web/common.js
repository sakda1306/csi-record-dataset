// Shared by monitor.html and remote.html: config, live stream, commands, timed sounds.
"use strict";

const App = { cfg: null, status: null, offsets: [], lastMsg: 0 };

async function loadConfig() {
  const r = await fetch("/api/config", { cache: "no-store" });
  App.cfg = await r.json();
  return App.cfg;
}

// Server clock estimate: server_time - local time, the largest recent value
// (the sample with the least network delay).
function updateOffset(serverTime) {
  App.offsets.push(serverTime - Date.now() / 1000);
  if (App.offsets.length > 40) App.offsets.shift();
  App.offset = Math.max(...App.offsets);
}
function serverNow() { return Date.now() / 1000 + (App.offset || 0); }
function isConnected() { return Date.now() - App.lastMsg < 2500; }

function connect(view, onStatus, onRows) {
  const es = new EventSource("/api/stream?view=" + view);
  es.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    App.lastMsg = Date.now();
    updateOffset(m.status.server_time);
    App.status = m.status;
    if (m.rows && onRows) onRows(m.rows);
    onStatus(m.status);
  };
  return es;
}

function getPin() { try { return localStorage.getItem("csirec.pin") || ""; } catch { return ""; } }
function setPin(p) { try { localStorage.setItem("csirec.pin", p); } catch { /* private mode */ } }

async function api(action, args = {}) {
  const r = await fetch("/api/cmd", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pin: getPin(), action, args }),
  });
  let body = {};
  try { body = await r.json(); } catch { /* empty */ }
  if (!r.ok || !body.ok) {
    const err = new Error(body.error || "HTTP " + r.status);
    err.status = r.status;
    throw err;
  }
  return body;
}

// ------------------------------------------------------------------ sound
let actx = null;
function unlockAudio() {
  try {
    if (!actx) actx = new (window.AudioContext || window.webkitAudioContext)();
    if (actx.state === "suspended") actx.resume();
  } catch { /* no audio */ }
}
function audioReady() { return actx && actx.state === "running"; }

function tone(freq, ms, whenS, vol = 0.35) {
  if (!actx || !App.cfg?.cue?.sound) return null;
  const t = actx.currentTime + Math.max(0, whenS);
  const osc = actx.createOscillator();
  const g = actx.createGain();
  osc.type = "square";
  osc.frequency.value = freq;
  g.gain.setValueAtTime(0, t);
  g.gain.linearRampToValueAtTime(vol, t + 0.005);
  g.gain.setValueAtTime(vol, t + ms / 1000 - 0.01);
  g.gain.linearRampToValueAtTime(0, t + ms / 1000);
  osc.connect(g).connect(actx.destination);
  osc.start(t);
  osc.stop(t + ms / 1000 + 0.02);
  return osc;
}

const SOUNDS = {
  tick: (w) => [tone(880, 110, w, 0.25)],
  start: (w) => [tone(1320, 260, w)],
  cue: (w) => [tone(1900, 90, w), tone(1900, 90, w + 0.13), tone(1900, 160, w + 0.26)],
  end: (w) => [tone(660, 150, w), tone(440, 260, w + 0.18)],
};

function speak(text) {
  if (!App.cfg?.cue?.speech || !("speechSynthesis" in window)) return;
  try {
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.lang = "th-TH";
    u.rate = 1.05;
    speechSynthesis.speak(u);
  } catch { /* no speech */ }
}

// ------------------------------------------------------------------ take timeline
// Plays the countdown ticks, start, cue and end sounds at the server's planned times,
// and calls onEvent(name) at those times for the visual side (flash, vibrate).
const Sched = { seq: null, done: new Set(), timers: [], oscs: [] };

function clearSched() {
  Sched.timers.forEach(clearTimeout);
  Sched.oscs.forEach((o) => { try { o.stop(); } catch { /* already stopped */ } });
  Object.assign(Sched, { seq: null, done: new Set(), timers: [], oscs: [] });
}

function runTimeline(st, onEvent) {
  const tk = st.take;
  const live = tk && (st.phase === "countdown" || st.phase === "recording");
  if (!live) { if (Sched.seq !== null) clearSched(); return; }
  if (tk.seq !== Sched.seq) { clearSched(); Sched.seq = tk.seq; onEvent("new", tk); }
  const now = serverNow();
  const events = [["tick3", tk.t_start - 3], ["tick2", tk.t_start - 2], ["tick1", tk.t_start - 1],
    ["start", tk.t_start], ["end", tk.t_end]];
  if (tk.cue_at) events.push(["cue", tk.cue_at]);
  for (const [name, t] of events) {
    if (Sched.done.has(name)) continue;
    const dt = t - now;
    if (dt < -0.3) { Sched.done.add(name); continue; } // missed it
    if (dt > 3.5) continue; // schedule closer to the time, with a fresher clock estimate
    Sched.done.add(name);
    const kind = name.startsWith("tick") ? "tick" : name;
    Sched.oscs.push(...(SOUNDS[kind](Math.max(0, dt)) || []).filter(Boolean));
    Sched.timers.push(setTimeout(() => onEvent(name, tk), Math.max(0, dt * 1000)));
  }
}

// ------------------------------------------------------------------ helpers
const $ = (sel, root = document) => root.querySelector(sel);
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function activityById(id) { return App.cfg.activities.find((a) => a.id === id); }
function fmt(v, d = 0, unit = "") { return v === null || v === undefined ? "–" : Number(v).toFixed(d) + unit; }

const PHASE_TH = { idle: "พร้อม", countdown: "เตรียมตัว", recording: "กำลังบันทึก", review: "รอตัดสิน", rest: "พัก" };
