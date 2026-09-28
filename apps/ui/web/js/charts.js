// charts.js — small SVG charts (line with optional log y, horizontal bars). No dependencies.
const NS = "http://www.w3.org/2000/svg";
const s = (tag, attrs = {}, ...kids) => {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v != null) el.setAttribute(k, v);
  for (const k of kids) if (k != null) el.append(typeof k === "string" ? document.createTextNode(k) : k);
  return el;
};

function niceTicks(lo, hi, n = 5) {
  if (!(hi > lo)) return [lo];
  const step0 = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((x) => x >= step0) || step0;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-12 * Math.abs(hi); v += step) out.push(v);
  return out;
}
const fmtTick = (v) => (v === 0 ? "0" : Math.abs(v) >= 1e4 || Math.abs(v) < 1e-2 ? v.toExponential(0) : String(+v.toPrecision(3)));

/** series: [{name, color, points:[[x,y],...], dash}] ; opts: {logY, xLabel, yLabel, height} */
export function lineChart(series, opts = {}) {
  const W = 640, H = opts.height || 240, L = 58, R = 14, T = 10, B = 28;
  const svg = s("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none", role: "img" });
  const pts = series.flatMap((se) => se.points.filter(([, y]) => y != null && Number.isFinite(y) && (!opts.logY || y > 0)));
  if (!pts.length) {
    svg.append(s("text", { x: W / 2, y: H / 2, "text-anchor": "middle" }, opts.empty || "waiting for data"));
    return svg;
  }
  const xs = pts.map((p) => p[0]);
  // robust: the y-range ignores the first iterations (early objectives can be orders of magnitude off)
  const yPts = opts.robust && pts.length > 12 ? series.flatMap((se) => se.points.slice(Math.floor(se.points.length * 0.12)))
    .filter(([, y]) => y != null && Number.isFinite(y)) : pts;
  const ys = yPts.map((p) => (opts.logY ? Math.log10(p[1]) : p[1]));
  let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  if (x1 === x0) x1 = x0 + 1;
  if (y1 === y0) { y0 -= 1; y1 += 1; }
  if (opts.logY) { y0 = Math.floor(y0); y1 = Math.ceil(y1); } else { const pad = (y1 - y0) * 0.06; y0 -= pad; y1 += pad; }
  const lo = () => (opts.logY ? 10 ** y0 : y0), hi = () => (opts.logY ? 10 ** y1 : y1);
  const X = (x) => L + ((x - x0) / (x1 - x0)) * (W - L - R);
  const Y = (y) => T + (1 - ((opts.logY ? Math.log10(y) : y) - y0) / (y1 - y0)) * (H - T - B);
  const g = s("g", { class: "axis" });
  const yt = opts.logY ? Array.from({ length: y1 - y0 + 1 }, (_, i) => 10 ** (y0 + i)).filter((_, i, a) => a.length <= 9 || i % 2 === 0)
    : niceTicks(y0, y1, 4);
  for (const v of yt) {
    const y = Y(v);
    g.append(s("line", { x1: L, x2: W - R, y1: y, y2: y, class: "gridl" }));
    g.append(s("text", { x: L - 6, y: y + 3, "text-anchor": "end" }, opts.logY ? `1e${Math.round(Math.log10(v))}` : fmtTick(v)));
  }
  for (const v of niceTicks(x0, x1, 6)) g.append(s("text", { x: X(v), y: H - 9, "text-anchor": "middle" }, fmtTick(v)));
  g.append(s("line", { x1: L, x2: W - R, y1: H - B, y2: H - B }));
  if (opts.xLabel) g.append(s("text", { x: W - R, y: H - 9, "text-anchor": "end" }, opts.xLabel));
  if (opts.threshold != null && (!opts.logY || opts.threshold > 0)) {
    const y = Y(opts.threshold);
    if (y > T && y < H - B) {
      g.append(s("line", { x1: L, x2: W - R, y1: y, y2: y, stroke: "#ffb547", "stroke-dasharray": "4 4", "stroke-opacity": 0.6 }));
      g.append(s("text", { x: W - R - 2, y: y - 4, "text-anchor": "end", fill: "#ffb547" }, opts.thresholdLabel || ""));
    }
  }
  svg.append(g);
  for (const se of series) {
    let p = se.points.filter(([, y]) => y != null && Number.isFinite(y) && (!opts.logY || y > 0));
    if (opts.robust) p = p.map(([x, y]) => [x, Math.min(Math.max(y, lo()), hi())]);
    if (!p.length) continue;
    const d = p.map(([x, y], i) => `${i ? "L" : "M"}${X(x).toFixed(1)},${Y(y).toFixed(1)}`).join("");
    svg.append(s("path", { d, fill: "none", stroke: se.color, "stroke-width": 1.8, "stroke-dasharray": se.dash, "vector-effect": "non-scaling-stroke" }));
    if (opts.markers) for (const [x, y] of p) svg.append(s("circle", { cx: X(x), cy: Y(y), r: 3.2, fill: se.color }));
    else { const [lx, ly] = p[p.length - 1]; svg.append(s("circle", { cx: X(lx), cy: Y(ly), r: 2.8, fill: se.color })); }
  }
  return svg;
}

/** rows: [{label, value, color, text, sub}] horizontal bars, log scale optional */
export function barChart(rows, opts = {}) {
  const W = 640, rowH = opts.rowH || 26, L = opts.labelW || 190, R = 110, T = 6;
  const H = T * 2 + rows.length * rowH + (opts.refLine ? 14 : 0);
  const svg = s("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, style: `height:${H}px`, role: "img" });
  const vals = rows.map((r) => r.value).filter((v) => v > 0);
  const vmax = Math.max(...vals, opts.refLine || 0), vmin = Math.min(...vals, opts.refLine || Infinity);
  const lg = opts.log;
  const lo = lg ? Math.floor(Math.log10(vmin)) : 0, hi = lg ? Math.ceil(Math.log10(vmax)) : vmax;
  const X = (v) => L + (((lg ? Math.log10(v) : v) - lo) / (hi - lo || 1)) * (W - L - R);
  rows.forEach((r, i) => {
    const y = T + i * rowH;
    svg.append(s("text", { x: L - 8, y: y + rowH / 2 + 3, "text-anchor": "end", fill: "#93a1b0" }, r.label));
    if (r.value > 0) {
      svg.append(s("rect", { x: L, y: y + 5, width: Math.max(2, X(r.value) - L), height: rowH - 11, rx: 2, fill: r.color || "#5ce1e6", "fill-opacity": 0.85 }));
      svg.append(s("text", { x: X(r.value) + 6, y: y + rowH / 2 + 3, fill: "#d5dde6" }, r.text ?? String(r.value)));
    } else svg.append(s("text", { x: L + 6, y: y + rowH / 2 + 3, fill: "#5f6d7c" }, r.text ?? "—"));
  });
  if (opts.refLine) {
    const x = X(opts.refLine);
    svg.append(s("line", { x1: x, x2: x, y1: T, y2: H - 14, stroke: "#ffb547", "stroke-dasharray": "3 3" }));
    svg.append(s("text", { x, y: H - 3, "text-anchor": "middle", fill: "#ffb547" }, opts.refLabel || ""));
  }
  return svg;
}

/** Dolan–Moré performance profile. series: [{name, color, dash, ratios:[r ≥ 1 or Infinity per problem]}].
 * ρ(τ) = share of problems solved within τ × the best solver's time; x axis log₂ τ from 1 to tauMax. */
export function profileChart(series, opts = {}) {
  const W = 640, H = opts.height || 260, L = 46, R = 14, T = 10, B = 30;
  const tauMax = opts.tauMax || 2 ** 10;
  const svg = s("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none", role: "img" });
  const lx = (t) => L + (Math.log2(Math.max(1, Math.min(t, tauMax))) / Math.log2(tauMax)) * (W - L - R);
  const ly = (v) => T + (1 - v) * (H - T - B);
  const g = s("g", { class: "axis" });
  for (const v of [0, 0.25, 0.5, 0.75, 1]) {
    g.append(s("line", { x1: L, x2: W - R, y1: ly(v), y2: ly(v), class: "gridl" }));
    g.append(s("text", { x: L - 6, y: ly(v) + 3, "text-anchor": "end" }, `${Math.round(v * 100)}%`));
  }
  for (let e = 0; 2 ** e <= tauMax; e += Math.log2(tauMax) > 8 ? 2 : 1)
    g.append(s("text", { x: lx(2 ** e), y: H - 11, "text-anchor": "middle" }, 2 ** e >= 1024 ? `2^${e}` : String(2 ** e)));
  g.append(s("text", { x: W - R, y: H - 1, "text-anchor": "end" }, opts.xLabel || "τ — within this factor of the fastest"));
  g.append(s("line", { x1: L, x2: W - R, y1: H - B, y2: H - B }));
  svg.append(g);
  for (const se of series) {
    const n = se.ratios.length;
    if (!n) continue;
    const fin = se.ratios.filter((r) => Number.isFinite(r) && r <= tauMax).sort((a, b) => a - b);
    let d = `M${lx(1).toFixed(1)},${ly(0).toFixed(1)}`, k = 0;
    for (const r of fin) { k++; d += `H${lx(r).toFixed(1)}V${ly(k / n).toFixed(1)}`; }
    d += `H${(W - R).toFixed(1)}`;
    svg.append(s("path", { d, fill: "none", stroke: se.color, "stroke-width": 2, "stroke-dasharray": se.dash, "vector-effect": "non-scaling-stroke" }));
  }
  return svg;
}

/** Scatter on log–log axes with the y = x diagonal. points: [{x, y, color, title}]. */
export function scatterChart(points, opts = {}) {
  const W = 640, H = opts.height || 300, L = 56, R = 14, T = 10, B = 32;
  const svg = s("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none", role: "img" });
  const ok = points.filter((p) => p.x > 0 && p.y > 0 && Number.isFinite(p.x) && Number.isFinite(p.y));
  if (!ok.length) { svg.append(s("text", { x: W / 2, y: H / 2, "text-anchor": "middle" }, opts.empty || "no data")); return svg; }
  const lo = Math.floor(Math.log10(Math.min(...ok.flatMap((p) => [p.x, p.y])))), hi = Math.ceil(Math.log10(Math.max(...ok.flatMap((p) => [p.x, p.y]))));
  const X = (v) => L + ((Math.log10(v) - lo) / Math.max(1, hi - lo)) * (W - L - R);
  const Y = (v) => T + (1 - (Math.log10(v) - lo) / Math.max(1, hi - lo)) * (H - T - B);
  const g = s("g", { class: "axis" });
  for (let e = lo; e <= hi; e++) {
    g.append(s("line", { x1: L, x2: W - R, y1: Y(10 ** e), y2: Y(10 ** e), class: "gridl" }));
    g.append(s("line", { x1: X(10 ** e), x2: X(10 ** e), y1: T, y2: H - B, class: "gridl" }));
    g.append(s("text", { x: L - 6, y: Y(10 ** e) + 3, "text-anchor": "end" }, `1e${e}`));
    g.append(s("text", { x: X(10 ** e), y: H - 13, "text-anchor": "middle" }, `1e${e}`));
  }
  g.append(s("text", { x: W - R, y: H - 1, "text-anchor": "end" }, opts.xLabel || "x"));
  g.append(s("text", { x: L + 4, y: T + 10 }, opts.yLabel || "y"));
  svg.append(g);
  svg.append(s("line", { x1: X(10 ** lo), y1: Y(10 ** lo), x2: X(10 ** hi), y2: Y(10 ** hi), stroke: "#5f6d7c", "stroke-dasharray": "5 4", "vector-effect": "non-scaling-stroke" }));
  if (opts.diagLabel) svg.append(s("text", { x: X(10 ** hi) - 4, y: Y(10 ** hi) + 14, "text-anchor": "end", fill: "#93a1b0", "font-size": 10 }, opts.diagLabel));
  for (const p of ok) {
    const c = s("circle", { cx: X(p.x), cy: Y(p.y), r: 3.6, fill: p.color || "#5ce1e6", "fill-opacity": 0.8 });
    if (p.title) c.append(s("title", {}, p.title));
    svg.append(c);
  }
  return svg;
}
