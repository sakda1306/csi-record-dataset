// Colour scaling for the CSI amplitude waterfall on the monitor page.
// rows = [{t, a: Uint8Array(nsc)}]. Amplitudes are whole numbers 0..255, so percentiles come
// from histograms instead of sorting.

// Turbo colour map (polynomial fit, Google 2019): dark blue -> green -> yellow -> dark red.
function turbo(t) {
  t = Math.min(1, Math.max(0, t));
  const r = 0.13572138 + t * (4.6153926 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943))));
  const g = 0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604))));
  const b = 0.1066733 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973))));
  const c = (v) => Math.round(Math.min(1, Math.max(0, v)) * 255);
  return [c(r), c(g), c(b)];
}

function percentile(hist, off, count, p) {
  const target = p * count;
  let acc = 0;
  for (let v = 0; v < 256; v++) {
    acc += hist[off + v];
    if (acc >= target && acc > 0) return v;
  }
  return 255;
}

// Amplitude mode: the colours span the 2nd..98th percentile, so one spike can't flatten the rest.
function ampRange(rows, nsc) {
  const hist = new Uint32Array(256);
  for (const r of rows) for (let i = 0; i < nsc; i++) hist[r.a[i]]++;
  const n = rows.length * nsc;
  if (!n) return [0, 1];
  const lo = percentile(hist, 0, n, 0.02);
  const hi = Math.max(lo + 1, percentile(hist, 0, n, 0.98));
  return [lo, hi];
}

// Change mode: each subcarrier against its own median, so movement lights up and a still room
// stays dark. On the real board a still room still moves |a - median| by 1-2 most of the time, so
// that much (noise = 75th percentile of |a - median|) counts as nothing, and the colours span at
// least 6 above it (more when the 98th percentile is higher, i.e. somebody moved).
function changeScale(rows, nsc) {
  const med = new Uint8Array(nsc);
  if (!rows.length) return { med, noise: 0, range: 6 };
  const sub = new Uint32Array(nsc * 256);
  for (const r of rows) for (let i = 0; i < nsc; i++) sub[i * 256 + r.a[i]]++;
  for (let i = 0; i < nsc; i++) med[i] = percentile(sub, i * 256, rows.length, 0.5);
  const dh = new Uint32Array(256);
  for (const r of rows) for (let i = 0; i < nsc; i++) dh[Math.abs(r.a[i] - med[i])]++;
  const n = rows.length * nsc;
  const noise = percentile(dh, 0, n, 0.75);
  return { med, noise, range: Math.max(6, percentile(dh, 0, n, 0.98) - noise) };
}

// 0..1 colour position for a deviation |a - median| under a changeScale result.
function changeT(absDev, s) {
  return Math.min(1, Math.max(0, (absDev - s.noise) / s.range));
}

if (typeof module !== "undefined") module.exports = { turbo, ampRange, changeScale, changeT };
