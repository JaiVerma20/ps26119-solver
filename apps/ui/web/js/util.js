// util.js — tiny DOM + formatting helpers (no framework).

// h("div.panel#id", {attrs}, ...children) — children: strings, nodes, arrays, null
export function h(tag, attrs, ...children) {
  // "div.a.b#id" — empty class segments ("div.kpi." from a conditional class) and spaces are tolerated
  const name = tag.match(/^[a-z0-9]+/i)?.[0] || "div";
  const el = document.createElement(name);
  for (const part of tag.slice(name.length).match(/[.#][^.#\s]*/g) || []) {
    if (part.length < 2) continue;
    part[0] === "." ? el.classList.add(part.slice(1)) : (el.id = part.slice(1));
  }
  if (attrs && (typeof attrs !== "object" || attrs instanceof Node || Array.isArray(attrs))) { children.unshift(attrs); attrs = null; }
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "html") el.innerHTML = v;
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  append(el, children);
  return el;
}
function append(el, kids) {
  for (const c of kids) {
    if (c == null || c === false) continue;
    if (Array.isArray(c)) append(el, c);
    else el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
}
export const $ = (sel, root = document) => root.querySelector(sel);

// numbers --------------------------------------------------------------------------------
export function fnum(v, digits = 6) {
  if (v == null || Number.isNaN(v)) return "—";
  if (!Number.isFinite(v)) return v > 0 ? "+∞" : "−∞";
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e9)) return v.toExponential(3).replace("e", "e");
  return Number(v.toPrecision(digits)).toLocaleString("en-US", { maximumFractionDigits: 8 });
}
export const fexp = (v) => (v == null || Number.isNaN(v) ? "—" : !Number.isFinite(v) ? (v > 0 ? "+∞" : "−∞") : v === 0 ? "0" : v.toExponential(1));
export const fint = (v) => (v == null ? "—" : Math.round(v).toLocaleString("en-US"));
export function fsec(s) {
  if (s == null || Number.isNaN(s)) return "—";
  if (s < 1e-3) return `${(s * 1e6).toFixed(0)} µs`;
  if (s < 1) return `${(s * 1e3).toFixed(s < 0.01 ? 2 : 1)} ms`;
  if (s < 120) return `${s.toFixed(s < 10 ? 2 : 1)} s`;
  return `${(s / 60).toFixed(1)} min`;
}
export const fratio = (r) => (r == null ? "—" : `${r >= 10 ? r.toFixed(1) : r.toFixed(2)}×`);
export const fbytes = (b) => (b < 1024 ? `${b} B` : b < 1 << 20 ? `${(b / 1024).toFixed(1)} KB` : `${(b / (1 << 20)).toFixed(1)} MB`);

// status → badge -------------------------------------------------------------------------
export function statusBadge(status) {
  const cls = { Optimal: "ok", Infeasible: "acc", Unbounded: "acc", TimeLimit: "warn", IterationLimit: "warn",
    NumericalError: "bad", NotSolved: "bad" }[status] || "mut";
  return h(`span.badge.${cls}`, status || "—");
}
export function passBadge(v, labels = ["PASS", "FAIL"]) {
  if (v === true || v === "PASS") return h("span.badge.ok", labels[0]);
  if (v === false || v === "FAIL") return h("span.badge.bad", labels[1]);
  return h("span.badge.mut", "—");
}

// provenance chip: every benchmark number shows where it comes from
export function srcChip(src) {
  if (!src) return null;
  const tip = `${src.file}\ncommit ${src.git_hash} · ${src.machine} · ${src.cpu || ""}${src.gpu && src.gpu !== "none" ? " · " + src.gpu : ""}\n${src.date || ""}`;
  return h("span.src-chip", { title: tip }, "source ", h("b", src.git_hash), " · ", src.file);
}

export function toast(msg, ms = 4000) {
  const t = h("div.toast", msg);
  document.body.append(t);
  setTimeout(() => t.remove(), ms);
}
