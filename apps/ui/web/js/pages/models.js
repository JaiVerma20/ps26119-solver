// pages/models.js — browse the models in the repository; statistics from the solver's own reader.
import { api } from "../api.js";
import { store } from "../store.js";
import { h, fint, fnum, fbytes, toast } from "../util.js";

export function mount(root) {
  let groups = [], filter = "";
  const list = h("div.mlist");
  const detail = h("div.grid", { style: { gap: "12px" } });
  const search = h("input.input.search", { placeholder: "filter models…", oninput: (e) => { filter = e.target.value.toLowerCase(); drawList(); } });

  root.append(
    h("div.page-head", h("div", h("div.eyebrow", "03 · models"), h("h1", "Model Explorer"),
      h("p", "Every model in the repository's data folders. Statistics come from `ps26119 info` (the solver's own MPS reader); the sparsity image is a block-count picture of the constraint matrix."))),
    h("div.grid.g-side",
      h("div.grid", { style: { gap: "12px", alignContent: "start" } },
        h("div.panel", h("div.panel-h", "Library", h("span.right.dim#mcount", "")), h("div.panel-b", { style: { paddingBottom: "6px" } }, search), list),
        generator()),
      detail));

  // bench/generate_*.py through the backend: LPs of any size whose optimum is known by construction
  function generator() {
    const PRESETS = { refinery: [[12, "12 · monthly"], [365, "365 · daily"], [2190, "2190 · 4-hourly"], [8760, "8760 · hourly"]],
                      random: [[10000, "10k rows"], [100000, "100k rows"], [1000000, "1M rows"]] };
    let kind = "refinery";
    const size = h("select.input"), seed = h("input.input", { type: "number", min: 1, max: 9999, value: 1, style: { width: "70px" } });
    const fill = () => size.replaceChildren(...PRESETS[kind].map(([v, l]) => h("option", { value: v }, l)));
    const kindSeg = h("div.seg", ["refinery", "random"].map((k) => h(`button${k === kind ? ".on" : ""}`, { onclick: (e) => {
      kind = k; fill(); kindSeg.querySelectorAll("button").forEach((b) => b.classList.toggle("on", b === e.target)); } }, k === "refinery" ? "refinery planning" : "random sparse")));
    fill();
    const go = h("button.btn.primary", { onclick: async () => {
      go.disabled = true; go.textContent = "generating…";
      try {
        const g = await api.generate(kind, +size.value, +seed.value);
        toast(g.created ? g.message : `${g.path} (already generated)`);
        groups = await api.models();
        store.set({ selectedModel: g.path }); drawList(); drawDetail();
      } catch (e) { toast(e.message); } finally { go.disabled = false; go.textContent = "Generate ▸"; }
    } }, "Generate ▸");
    return h("div.panel", h("div.panel-h", "Generate a test LP"), h("div.panel-b",
      h("div.field", h("span", "family"), kindSeg),
      h("div", { style: { display: "flex", gap: "8px", alignItems: "end" } },
        h("div.field", { style: { flex: 1 } }, h("span", "size"), size), h("div.field", h("span", "seed"), seed)),
      go,
      h("div.note", { style: { padding: "10px 0 0" } }, "Built around a KKT point, so the optimum is known exactly and every solve is checked against it. Refinery: real multi-period structure (crudes, CDU, FCC, blending specs, tanks) with synthetic prices.")));
  }

  function drawList() {
    const out = [];
    let n = 0;
    for (const g of groups) {
      const ms = g.models.filter((m) => !filter || m.name.toLowerCase().includes(filter) || g.group.toLowerCase().includes(filter));
      if (!ms.length) continue;
      out.push(h("div.mgroup", { title: g.description }, `${g.group} · ${ms.length}`));
      for (const m of ms) {
        n++;
        out.push(h(`div.mitem${m.path === store.selectedModel ? ".sel" : ""}`, { onclick: () => { store.set({ selectedModel: m.path }); drawList(); drawDetail(); } },
          h("span.mono", m.name), h("span.sz", `${m.format} · ${fbytes(m.bytes)}`)));
      }
    }
    list.replaceChildren(...(out.length ? out : [h("div.empty", "no match")]));
    root.querySelector("#mcount").textContent = `${n} models`;
  }

  async function drawDetail() {
    const path = store.selectedModel;
    detail.replaceChildren(h("div.panel", h("div.panel-b.muted", `reading ${path} …`)));
    let info;
    try { info = await api.modelInfo(path); } catch (e) { detail.replaceChildren(h("div.panel", h("div.panel-b.bad", e.message))); return; }
    if (store.selectedModel !== path) return;
    if (!info.ok) { detail.replaceChildren(h("div.panel", h("div.panel-h", path), h("div.panel-b", h("span.bad", "read error: "), info.error))); return; }
    const rs = info.rows_split || [0, 0, 0, 0, 0], cs = info.columns_split || [0, 0, 0];
    const tot = (a) => a.reduce((x, y) => x + y, 0) || 1;
    const seg = (vals, colors, labels) => h("div",
      h("div.split", vals.map((v, i) => h("i", { style: { width: `${(100 * v) / tot(vals)}%`, background: colors[i] }, title: `${labels[i]} ${v}` }))),
      h("div.legend", { style: { padding: "6px 0 0" } }, vals.map((v, i) => v ? h("span", h("i", { style: { background: colors[i], height: "8px", width: "8px" } }), `${labels[i]} ${fint(v)}`) : null)));
    const spyBox = h("div.panel-b", h("div.muted", "drawing sparsity…"));
    detail.replaceChildren(
      h("div.kpis",
        h("div.kpi.hl", h("div.k", "model"), h("div.v", info.name || path.split("/").pop()), h("div.s", path)),
        h("div.kpi", h("div.k", "rows"), h("div.v", fint(info.n_rows)), h("div.s", "constraints")),
        h("div.kpi", h("div.k", "columns"), h("div.v", fint(info.n_cols)), h("div.s", cs[1] + cs[2] ? `${fint(cs[1] + cs[2])} integer` : "continuous")),
        h("div.kpi", h("div.k", "nonzeros"), h("div.v", fint(info.n_nnz)), h("div.s", info.nonzeros?.match(/\(.*\)/)?.[0] || "")),
        h("div.kpi", h("div.k", "sense"), h("div.v", info.sense || "—"), h("div.s", "objective")),
        h("div.kpi", h("div.k", "dynamism"), h("div.v", (info.dynamism || "—").split(" ")[0]), h("div.s", "max/min |a_ij|"))),
      h("div.grid.g2",
        h("div.panel", h("div.panel-h", "Structure"), h("div.panel-b",
          h("div.field", h("span", "Rows by type"), seg(rs.slice(0, 5), ["#5ce1e6", "#a78bfa", "#3ddc97", "#ffb547", "#5f6d7c"], ["E", "L", "G", "ranged", "free"])),
          h("div.field", h("span", "Columns"), seg(cs, ["#5ce1e6", "#ffb547", "#ff5d6c"], ["continuous", "integer", "binary"])),
          h("dl.kv", { style: { marginTop: "8px" } },
            h("dt", "column bounds"), h("dd", info.column_bounds || "—"),
            h("dt", "|matrix coeff|"), h("dd", info.matrix_range || "—"),
            h("dt", "|objective coeff|"), h("dd", info.objective_range || "—"),
            h("dt", "|row bounds|"), h("dd", info.row_bounds_range || "—"),
            h("dt", "|column bounds|"), h("dd", info.column_bounds_range || "—"),
            h("dt", "fingerprint"), h("dd", info.fingerprint || "—"),
            info.known ? h("dt", "known optimum") : null, info.known ? h("dd.ok", `${fnum(info.known.optimum, 14)} (by construction)`) : null,
            info.note ? h("dt", "note") : null, info.note ? h("dd.warn", info.note) : null),
          info.warnings?.length ? h("div.note", info.warnings.join(" · ")) : null,
          h("div", { style: { display: "flex", gap: "8px", marginTop: "14px" } },
            h("a.btn.primary", { href: "#/solve" }, "Solve this model ▸")))),
        h("div.panel", h("div.panel-h", "Sparsity pattern", h("span.right.dim#spyinfo", "")), spyBox)));
    try {
      const sp = await api.sparsity(path);
      if (store.selectedModel !== path) return;
      if (!sp.ok) { spyBox.replaceChildren(h("div.muted", sp.error)); return; }
      const cv = h("canvas.spy", { width: sp.cols, height: sp.rows });
      const ctx = cv.getContext("2d"), img = ctx.createImageData(sp.cols, sp.rows);
      for (let i = 0; i < sp.grid.length; i++) {
        const v = sp.grid[i];
        const t = v ? 0.25 + 0.75 * Math.log1p(v) / Math.log1p(sp.max) : 0;
        img.data[4 * i] = 6 + t * 86; img.data[4 * i + 1] = 8 + t * 217; img.data[4 * i + 2] = 11 + t * 219; img.data[4 * i + 3] = 255;
      }
      ctx.putImageData(img, 0, 0);
      spyBox.replaceChildren(cv);
      root.querySelector("#spyinfo").textContent = sp.rows === sp.model_rows && sp.cols === sp.model_cols
        ? `exact · ${sp.rows} × ${sp.cols}` : `${sp.rows} × ${sp.cols} blocks of ${fint(sp.model_rows)} × ${fint(sp.model_cols)}`;
    } catch (e) { spyBox.replaceChildren(h("div.muted", `sparsity: ${e.message}`)); }
  }

  api.models().then((g) => { groups = g; drawList(); drawDetail(); }).catch((e) => toast(e.message));
}
