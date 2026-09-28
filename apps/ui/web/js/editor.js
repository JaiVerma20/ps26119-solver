// editor.js — write your own model on the Solve page: an algebraic text editor (CPLEX-LP / lp_solve
// style, parsed by apps/ui/backend/lptext.py — our own code) and a table grid for small problems that
// writes the same text. The saved model is an ordinary .lpm file in the uploads: the solver, the
// verifier and every reference solver read exactly the same numbers.
import { api } from "./api.js";
import { h, toast } from "./util.js";

const OPS = ["<=", ">=", "="];

function gridToText(g) {
  const term = (a, v, first) => {
    const x = Number(a);
    if (!Number.isFinite(x) || x === 0) return "";
    const sign = x < 0 ? "- " : first ? "" : "+ ";
    const mag = Math.abs(x) === 1 ? "" : `${Math.abs(x)} `;
    return `${sign}${mag}${v}`;
  };
  const expr = (coefs) => {
    const parts = [];
    coefs.forEach((a, j) => { const t = term(a, g.vars[j].name, parts.length === 0); if (t) parts.push(t); });
    return parts.join(" ") || "0";
  };
  const lines = [`\\ written by the table editor`, g.sense === "max" ? "maximize" : "minimize", `  obj: ${expr(g.obj)}`, "subject to"];
  g.rows.forEach((r) => lines.push(`  ${r.name}: ${expr(r.coefs)} ${r.op} ${Number(r.rhs) || 0}`));
  const bounds = [];
  g.vars.forEach((v) => {
    const lo = v.lo === "" ? "0" : v.lo, up = v.up === "" ? "inf" : v.up;
    if (lo === "0" && up === "inf") return;
    bounds.push(`  ${lo === "-inf" ? "-inf" : lo} <= ${v.name} <= ${up}`);
  });
  if (bounds.length) lines.push("bounds", ...bounds);
  const ints = g.vars.filter((v) => v.int).map((v) => v.name);
  if (ints.length) lines.push("integer", `  ${ints.join(" ")}`);
  lines.push("end");
  return lines.join("\n") + "\n";
}

const DEFAULT_GRID = () => ({
  sense: "max",
  vars: [{ name: "x1", lo: "", up: "", int: false }, { name: "x2", lo: "", up: "", int: false }],
  obj: [3, 5],
  rows: [{ name: "c1", coefs: [1, 0], op: "<=", rhs: 4 }, { name: "c2", coefs: [0, 2], op: "<=", rhs: 12 },
    { name: "c3", coefs: [3, 2], op: "<=", rhs: 18 }],
});

/** The editor panel. onSaved(path, run) is called after a successful save (run = also solve it). */
export function modelEditor({ onSaved, onClose }) {
  let mode = "text", grid = DEFAULT_GRID(), examples = {};
  let saved = "";
  try { saved = localStorage.getItem("ps.editor.text") || ""; } catch { /* storage may be unavailable */ }
  const nameIn = h("input.input.mono", { value: "my_model", style: { width: "180px" }, title: "saved as apps/ui/.runs/uploads/<name>.lpm" });
  const ta = h("textarea.input.mono.editor-ta", { spellcheck: "false", rows: 16 });
  ta.value = saved;
  ta.addEventListener("input", () => { try { localStorage.setItem("ps.editor.text", ta.value); } catch { /* ignore */ } status.replaceChildren(); });
  const status = h("div.editor-status");
  const exampleSel = h("select.input", { style: { width: "200px" }, onchange: (e) => {
    if (!e.target.value) return;
    ta.value = examples[e.target.value];
    nameIn.value = e.target.value;
    e.target.value = "";
    setMode("text");
  } }, h("option", { value: "" }, "load an example…"));
  api.modelExamples().then((r) => {
    examples = r.examples;
    for (const k of Object.keys(examples)) exampleSel.append(h("option", { value: k }, k));
    if (!ta.value.trim()) ta.value = examples.wyndor || "";
  }).catch(() => {});

  const modeSeg = h("div.seg", { style: { width: "220px" } });
  const body = h("div");
  const gridBox = h("div.scroll", { style: { maxHeight: "420px" } });
  const preview = h("pre.console", { style: { height: "auto", maxHeight: "160px", fontSize: "11px", margin: "0" } });

  function setMode(m) {
    mode = m;
    modeSeg.replaceChildren(...[["text", "algebraic text"], ["grid", "table"]].map(([k, l]) =>
      h("button", { class: mode === k ? "on" : null, type: "button", onclick: () => setMode(k) }, l)));
    body.replaceChildren(...(mode === "text" ? [ta, h("div.help", "maximize / minimize · subject to · bounds · integer · binary · end — or lp_solve style ",
      h("code", "max: 3x + 2y;  c1: x + y <= 4;  int x;"), ". Comments start with \\ or //.")] : [gridBox, h("div.field", { style: { marginTop: "10px" } }, h("span", "the model the table writes"), preview)]));
    if (mode === "grid") drawGrid();
  }

  function drawGrid() {
    const n = grid.vars.length;
    const cell = (value, on, attrs = {}) => h("input.input.mono.cell", { value: String(value), oninput: (e) => { on(e.target.value); preview.textContent = gridToText(grid); }, ...attrs });
    const head = h("tr", h("th", ""), grid.vars.map((v, j) => h("th", cell(v.name, (x) => { v.name = x.replace(/[^A-Za-z0-9_]/g, "") || `x${j + 1}`; }, { title: "variable name" }),
      h("button.x", { title: "remove variable", onclick: () => { if (n > 1) { grid.vars.splice(j, 1); grid.obj.splice(j, 1); grid.rows.forEach((r) => r.coefs.splice(j, 1)); drawGrid(); } } }, "×"))),
      h("th", ""), h("th", "rhs"), h("th", ""));
    const objRow = h("tr.obj", h("td", h("select.input.cell", { onchange: (e) => { grid.sense = e.target.value; preview.textContent = gridToText(grid); } },
      h("option", { value: "max", selected: grid.sense === "max" || null }, "maximize"), h("option", { value: "min", selected: grid.sense === "min" || null }, "minimize"))),
      grid.obj.map((a, j) => h("td", cell(a, (x) => { grid.obj[j] = x; }))), h("td"), h("td"), h("td"));
    const rows = grid.rows.map((r, i) => h("tr",
      h("td", cell(r.name, (x) => { r.name = x.replace(/[^A-Za-z0-9_]/g, "") || `c${i + 1}`; })),
      r.coefs.map((a, j) => h("td", cell(a, (x) => { r.coefs[j] = x; }))),
      h("td", h("select.input.cell", { onchange: (e) => { r.op = e.target.value; preview.textContent = gridToText(grid); } }, OPS.map((o) => h("option", { value: o, selected: r.op === o || null }, o)))),
      h("td", cell(r.rhs, (x) => { r.rhs = x; })),
      h("td", h("button.x", { title: "remove constraint", onclick: () => { if (grid.rows.length > 1) { grid.rows.splice(i, 1); drawGrid(); } } }, "×"))));
    const bRow = (label, key, ph) => h("tr.bnd", h("td.dim", label), grid.vars.map((v) => h("td", cell(v[key], (x) => { v[key] = x.trim(); }, { placeholder: ph }))), h("td"), h("td"), h("td"));
    const intRow = h("tr.bnd", h("td.dim", "integer"), grid.vars.map((v) => h("td", { style: { textAlign: "center" } },
      h("input", { type: "checkbox", checked: v.int || null, onchange: (e) => { v.int = e.target.checked; preview.textContent = gridToText(grid); } }))), h("td"), h("td"), h("td"));
    gridBox.replaceChildren(
      h("table.t.grid-ed", h("thead", head), h("tbody", objRow, rows, bRow("lower", "lo", "0"), bRow("upper", "up", "inf"), intRow)),
      h("div", { style: { display: "flex", gap: "8px", marginTop: "8px" } },
        h("button.btn", { onclick: () => { if (grid.vars.length >= 40) return toast("the table is for small models (≤ 40 variables); use the text editor"); grid.vars.push({ name: `x${grid.vars.length + 1}`, lo: "", up: "", int: false }); grid.obj.push(0); grid.rows.forEach((r) => r.coefs.push(0)); drawGrid(); } }, "+ variable"),
        h("button.btn", { onclick: () => { if (grid.rows.length >= 60) return toast("the table is for small models (≤ 60 constraints); use the text editor"); grid.rows.push({ name: `c${grid.rows.length + 1}`, coefs: grid.vars.map(() => 0), op: "<=", rhs: 0 }); drawGrid(); } }, "+ constraint"),
        h("button.btn", { onclick: () => { ta.value = gridToText(grid); setMode("text"); } }, "edit as text →")));
    preview.textContent = gridToText(grid);
  }

  async function save(run) {
    const text = mode === "grid" ? gridToText(grid) : ta.value;
    status.replaceChildren(h("span.dim", "checking…"));
    try {
      const r = await api.modelText(nameIn.value.trim(), text);
      status.replaceChildren(h("span.ok", `✓ ${r.sense} · ${r.rows} rows · ${r.cols} columns · ${r.nnz} nonzeros${r.integer ? ` · ${r.integer} integer` : ""} → ${r.path}`));
      onSaved(r.path, run);
    } catch (e) {
      status.replaceChildren(h("span.bad", e.message));
      const m = /^line (\d+)/.exec(e.message);
      if (m && mode === "text") {  // put the cursor on the line with the error
        const lines = ta.value.split("\n"), ln = Math.min(+m[1], lines.length);
        const start = lines.slice(0, ln - 1).join("\n").length + (ln > 1 ? 1 : 0);
        ta.focus(); ta.setSelectionRange(start, start + (lines[ln - 1] || "").length);
      }
    }
  }

  setMode("text");
  return h("div.panel.editor",
    h("div.panel-h", "Write your own model", h("span.right", h("button.btn", { style: { height: "24px" }, onclick: onClose }, "close"))),
    h("div.panel-b", { style: { display: "grid", gap: "10px" } },
      h("div", { style: { display: "flex", gap: "10px", alignItems: "center", flexWrap: "wrap" } }, modeSeg, exampleSel,
        h("label", { style: { display: "flex", gap: "6px", alignItems: "center" } }, h("span.dim", "name"), nameIn)),
      body,
      h("div", { style: { display: "flex", gap: "8px", alignItems: "center", flexWrap: "wrap" } },
        h("button.btn", { onclick: () => save(false) }, "Check & save"),
        h("button.btn.primary", { onclick: () => save(true) }, "Save & solve ▸"), status),
      h("div.note", "Parsed by our own reader (apps/ui/backend/lptext.py) into the model contract and saved as a .lpm file; the solver, the independent verifier and every reference solver then read the same numbers.")));
}
