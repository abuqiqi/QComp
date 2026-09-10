/* 离线选层的计算、状态和浏览器交互；纯计算接口同时用于浏览器测试。 */
"use strict";
const LayerSelection = (() => {
  const rounded = value => Number(value.toFixed(10));
  const byName = (a, b) => a.name < b.name ? -1 : a.name > b.name ? 1 : 0;
  // 为 payload 生成初始设置，五任务等权且每项独立保留指标。
  function defaults(payload) {
    return {mode: "fixed", k: 32, capEnabled: false, cap: 5, datasets: payload.datasets.map(d =>
      ({id: d.id, enabled: true, metric: d.defaultMetric, weight: 20, lower: 10, upper: 30}))};
  }
  // 校验设置，拒绝无效选择、权重与导入配置；不悄悄修正用户输入。
  function validate(payload, state) {
    if (!state || !["fixed", "stable"].includes(state.mode) || !Number.isInteger(state.k) || state.k < 1 || state.k > payload.modules.length) throw Error("K 必须是 1～252 的整数。");
    if (typeof state.capEnabled !== "boolean" || !Number.isFinite(state.cap) || state.cap < 0 || state.cap > 100) throw Error("掉点上限必须在 0～100 pp 之间。");
    if (!Array.isArray(state.datasets) || state.datasets.length !== payload.datasets.length) throw Error("数据集设置不完整。");
    state.datasets.forEach((s, i) => {
      const d = payload.datasets[i];
      if (s.id !== d.id || typeof s.enabled !== "boolean" || !Object.hasOwn(d.values, s.metric)) throw Error("数据集或指标不匹配。");
      if (!Number.isFinite(s.weight) || s.weight < 0 || s.weight > 100) throw Error("权重必须在 0～100 之间。");
      if (state.mode === "stable" && s.enabled && (![s.lower, s.upper].every(v => Number.isInteger(v) && v >= 0 && v <= 100 && v % 5 === 0) || s.lower > s.upper)) throw Error("权重范围必须为 0～100% 内的 5% 倍数，且下限不大于上限。");
    });
    const active = state.datasets.filter(d => d.enabled);
    if (!active.length) throw Error("请至少勾选一个数据集。");
    if (!active.some(d => d.weight > 0)) throw Error("已勾选数据集的权重总和为零，请增加权重。");
  }
  // 勾选集合改变时，围绕等权重置各任务范围。
  function resetRanges(state) {
    const n = state.datasets.filter(d => d.enabled).length;
    if (!n) return;
    state.datasets.forEach(d => {d.lower = Math.floor(10 / n) * 5; d.upper = Math.min(100, Math.ceil(30 / n) * 5);});
  }
  // 构造全部模块的有符号掉点、当前分数和筛选资格。
  function evaluate(payload, state) {
    validate(payload, state);
    const active = state.datasets.map((d, i) => d.enabled ? i : -1).filter(i => i >= 0);
    const total = active.reduce((sum, i) => sum + state.datasets[i].weight, 0);
    const weights = active.map(i => state.datasets[i].weight / total);
    return payload.modules.map((module, index) => {
      const drops = payload.datasets.map((d, i) => 100 * (d.baseline[state.datasets[i].metric] - d.values[state.datasets[i].metric][index]));
      const losses = active.map(i => Math.max(0, drops[i]));
      const maxDrop = Math.max(...losses);
      const score = losses.reduce((sum, value, i) => sum + value * weights[i], 0);
      return {...module, index, drops, losses, maxDrop, score, eligible: !state.capEnabled || maxDrop <= state.cap + 1e-10};
    });
  }
  // 固定权重按得分及明确的并列规则排序，未满足筛选条件的模块置后。
  function fixedOrder(rows) {
    return [...rows].sort((a, b) => Number(b.eligible) - Number(a.eligible) || rounded(a.score) - rounded(b.score) || a.maxDrop - b.maxDrop || b.saving - a.saving || byName(a, b));
  }
  // 穷举以 5% 为步长且总和等于 100% 的权重组合。
  function weightGrid(settings) {
    const bounds = settings.filter(d => d.enabled);
    const result = [];
    // 按剩余整数份额递归，最后一项直接使用余量。
    function visit(index, remaining, values) {
      const bound = bounds[index];
      if (!bound) return;
      if (index === bounds.length - 1) {
        if (remaining >= bound.lower / 5 && remaining <= bound.upper / 5) result.push([...values, remaining].map(v => v / 20));
        return;
      }
      for (let v = bound.lower / 5; v <= Math.min(bound.upper / 5, remaining); v++) visit(index + 1, remaining - v, [...values, v]);
    }
    visit(0, 20, []);
    return result;
  }
  // 为一组加权分数分配平均名次和前 K 名频率份额，消除并列名称偏差。
  function tiedRanks(scored, k) {
    const sorted = [...scored].sort((a, b) => rounded(a.score) - rounded(b.score));
    const result = [];
    for (let start = 0; start < sorted.length;) {
      let end = start + 1;
      while (end < sorted.length && rounded(sorted[end].score) === rounded(sorted[start].score)) end++;
      const share = Math.min(end - start, Math.max(0, k - start)) / (end - start);
      const rank = (start + 1 + end) / 2;
      for (let i = start; i < end; i++) result.push({index: sorted[i].index, rank, share});
      start = end;
    }
    return result;
  }
  // 分批计算入选频率和 nearest-rank 分位数；通过回调支持取消和进度显示。
  async function stability(rows, grid, k, cancelled = () => false, progress = () => {}) {
    if (!grid.length) throw Error("当前上下限没有总和为 100% 的权重组合。");
    const eligible = rows.filter(r => r.eligible);
    const stats = Object.fromEntries(eligible.map(r => [r.index, {hits: 0, worst: 0, ranks: []}]));
    for (let g = 0; g < grid.length; g++) {
      if (cancelled()) return null;
      const scored = eligible.map(r => ({index: r.index, score: r.losses.reduce((sum, d, i) => sum + d * grid[g][i], 0)}));
      for (const r of scored) stats[r.index].worst = Math.max(stats[r.index].worst, r.score);
      for (const r of tiedRanks(scored, k)) {stats[r.index].hits += r.share; stats[r.index].ranks.push(r.rank);}
      if ((g + 1) % 24 === 0 || g + 1 === grid.length) {
        progress(g + 1, grid.length);
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }
    if (cancelled()) return null;
    for (const stat of Object.values(stats)) {
      stat.ranks.sort((a, b) => a - b);
      stat.frequency = stat.hits / grid.length;
      stat.medianRank = stat.ranks[Math.ceil(grid.length * .5) - 1];
      stat.p95Rank = stat.ranks[Math.ceil(grid.length * .95) - 1];
      delete stat.ranks;
    }
    return {count: grid.length, stats};
  }
  // 稳定性按入选频率、任务损失与参数收益决定最终候选顺序。
  function stableOrder(rows, result) {
    return [...rows].sort((a, b) => {
      if (a.eligible !== b.eligible) return Number(b.eligible) - Number(a.eligible);
      if (!a.eligible) return byName(a, b);
      const x = result.stats[a.index], y = result.stats[b.index];
      return rounded(y.frequency) - rounded(x.frequency) || a.maxDrop - b.maxDrop || x.worst - y.worst || b.saving - a.saving || byName(a, b);
    });
  }
  // 导入只接受当前页面的源数据与可验证设置，候选列表随后重新计算核对。
  function importSettings(payload, snapshot) {
    if (snapshot.schemaVersion !== 1 || snapshot.kind !== "qcomp_layer_selection" || snapshot.rank !== payload.rank) throw Error("方案格式或 rank 不匹配。");
    if (!Array.isArray(snapshot.sources) || snapshot.sources.length !== payload.datasets.length) throw Error("来源不完整。");
    payload.datasets.forEach((d, i) => {
      const source = snapshot.sources[i];
      if (source.id !== d.id || source.sha256 !== d.source.sha256 || source.report_sha256 !== d.source.report_sha256) throw Error("来源哈希不匹配，不能导入其他实验的数据。");
    });
    validate(payload, snapshot.settings);
    if (!Array.isArray(snapshot.selectedModules) || new Set(snapshot.selectedModules).size !== snapshot.selectedModules.length || snapshot.selectedModules.some(n => !payload.modules.some(m => m.name === n))) throw Error("候选模块列表无效。");
    return structuredClone(snapshot.settings);
  }
  return {defaults, validate, resetRanges, evaluate, fixedOrder, weightGrid, tiedRanks, stability, stableOrder, importSettings};
})();
window.LayerSelection = LayerSelection;

(() => {
  const data = JSON.parse(document.getElementById("dashboard-data").textContent);
  let state = LayerSelection.defaults(data), stable = null, revision = 0, running = false, current = null, focused = null;
  const snapshots = {A: null, B: null};
  const $ = id => document.getElementById(id);
  // 将动态文本编码后再用于页面模板，避免来源路径或模块名成为 HTML。
  function escape(value) {return String(value).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}
  // 将比例显示为百分比，不改变计算精度。
  function percent(value) {return `${(value * 100).toFixed(2)}%`;}
  // 提取设置签名，保证旧稳定性结果不会用于新条件。
  function signature() {return JSON.stringify(state);}
  // 更新提示及错误语义。
  function status(message, error = false) {$("status").textContent = message; $("status").classList.toggle("error", error);}
  // 构建五个数据集卡片及指标、来源和权重范围控件。
  function renderCards() {
    $("datasets").innerHTML = data.datasets.map((d, i) => {
      const s = state.datasets[i];
      return `<article class="dataset-card ${s.enabled ? "" : "off"}" data-index="${i}"><label class="dataset-name"><input type="checkbox" data-field="enabled" ${s.enabled ? "checked" : ""}>${escape(d.label)}</label><small>${d.count.toLocaleString()} / ${d.total.toLocaleString()} 题</small><select aria-label="${escape(d.label)} 指标" data-field="metric">${Object.keys(d.values).map(m => `<option ${s.metric === m ? "selected" : ""}>${escape(m)}</option>`).join("")}</select><div>基线 <b class="baseline"></b></div><input aria-label="${escape(d.label)} 权重" data-field="weight" type="range" min="0" max="100" step="1" value="${s.weight}"><div class="weight-label"><span class="raw-weight"></span><strong class="normalized-weight"></strong></div><div class="range-control" ${state.mode === "stable" ? "" : "hidden"}><input aria-label="${escape(d.label)} 权重下限" data-field="lower" type="number" min="0" max="100" step="5" value="${s.lower}">～<input aria-label="${escape(d.label)} 权重上限" data-field="upper" type="number" min="0" max="100" step="5" value="${s.upper}">%</div><details><summary>查看实验来源</summary><div>${escape(d.source.path)}</div><div>SHA-256: ${escape(d.source.sha256)}</div><div>few-shot: ${d.source.configuration.num_fewshot ?? "任务默认"}</div>${d.source.aggregation ? "<div>已验证合并来源，分段数据不重复参与评分。</div>" : ""}</details></article>`;
    }).join("");
    updateLabels();
  }
  // 滑块变化时只更新文本，保留焦点与拖动状态。
  function updateLabels() {
    const total = state.datasets.reduce((sum, d) => sum + (d.enabled ? d.weight : 0), 0);
    document.querySelectorAll(".dataset-card").forEach((card, i) => {
      const s = state.datasets[i], d = data.datasets[i];
      card.querySelector(".baseline").textContent = percent(d.baseline[s.metric]);
      card.querySelector(".raw-weight").textContent = `权重值 ${s.weight}`;
      card.querySelector(".normalized-weight").textContent = s.enabled && total ? `实际 ${percent(s.weight / total)}` : "不参与评分";
    });
  }
  // 设置变更取消当前计算，清空候选并要求重新计算稳定性。
  function changed(rebuild = false) {
    revision++; running = false;
    $("progress").value = 0; $("progress-label").textContent = "";
    if (rebuild) renderCards(); else updateLabels();
    render();
  }
  // 悬停和详情使用全部任务的原始得分，并标明未参与任务。
  function description(row) {
    return [row.name, ...data.datasets.map((d, i) => {
      const s = state.datasets[i];
      return `${d.label}${s.enabled ? "" : "（未参与）"} / ${s.metric}: 基线 ${percent(d.baseline[s.metric])} → ${percent(d.values[s.metric][row.index])}; 掉点 ${row.drops[i].toFixed(4)} pp`;
    }), `参数节省 ${percent(row.saving)}`].join("\n");
  }
  // 渲染原生 SVG 热力图，选层描边与焦点描边独立显示。
  function renderHeatmap(rows, selected, result) {
    const names = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"];
    const stableMode = state.mode === "stable";
    const maximum = stableMode ? 1 : Math.max(.01, ...rows.filter(r => r.eligible).map(r => r.score));
    $("legend").textContent = stableMode ? (result ? "入选频率 · 0% → 100% · 深色更高" : "稳定性结果待计算") : `加权掉点 · 0 → ${maximum.toFixed(2)} pp · 深色更高`;
    let svg = `<svg viewBox="0 0 1250 302" role="img" aria-label="模块敏感性热力图">`;
    names.forEach((n, i) => {svg += `<text x="84" y="${53 + i * 32}" text-anchor="end" font-size="12" fill="#48655b">${n}</text>`;});
    for (let b = 0; b < 36; b++) svg += `<text x="${114 + b * 31}" y="24" text-anchor="middle" font-size="11" fill="#627572">${b}</text>`;
    for (const row of rows) {
      const value = stableMode ? (result?.stats[row.index]?.frequency ?? null) : row.score;
      const t = value === null ? 0 : Math.min(1, Math.max(0, value / maximum));
      const fill = !row.eligible || value === null ? "#e4e9e4" : `hsl(157, ${25 + t * 20}%, ${96 - t * 67}%)`;
      const sel = selected.has(row.index);
      svg += `<rect class="cell ${focused === row.index ? "focused" : ""}" tabindex="0" role="button" aria-label="${escape(row.name)}" data-module="${row.index}" x="${100 + row.block * 31}" y="${35 + names.indexOf(row.type) * 32}" width="28" height="28" rx="3" fill="${fill}" stroke="${sel ? "#122e27" : "#fff"}" stroke-width="${sel ? 2 : 1}"><title>${escape(description(row))}${value === null ? "" : `\n${stableMode ? "入选频率 " + percent(value) : "加权掉点 " + value.toFixed(4) + " pp"}`}</title></rect>`;
    }
    svg += '<text x="657" y="286" text-anchor="middle" font-size="12" fill="#627572">Transformer block</text></svg>';
    $("heatmap").innerHTML = svg;
  }
  // 根据当前排名生成表格，未通过阈值的模块仍可检查原始分数。
  function renderTable(rows, selected, result) {
    const extra = result ? "<th>入选频率</th><th>排名中位数</th><th>P95 排名</th><th>最差加权分</th>" : "";
    $("ranking-head").innerHTML = `<tr><th>排名</th><th>模块</th><th>候选</th><th>加权掉点 pp</th><th>最大掉点 pp</th><th>参数节省</th>${extra}${data.datasets.map((d, i) => `<th>${escape(d.label)}${state.datasets[i].enabled ? "" : "（未参与）"}<br>掉点 pp</th>`).join("")}</tr>`;
    $("ranking-body").innerHTML = rows.map((r, i) => {
      const stat = result?.stats[r.index];
      return `<tr id="row-${r.index}" data-module="${r.index}" tabindex="0" class="${selected.has(r.index) ? "selected" : ""} ${focused === r.index ? "focused" : ""}"><td>${r.eligible ? i + 1 : "—"}</td><td>${escape(r.name)}</td><td>${selected.has(r.index) ? "✓ 入选" : r.eligible ? "" : "超出上限"}</td><td>${r.score.toFixed(4)}</td><td>${r.maxDrop.toFixed(4)}</td><td>${percent(r.saving)}</td>${result ? `<td>${stat ? percent(stat.frequency) : "—"}</td><td>${stat?.medianRank ?? "—"}</td><td>${stat?.p95Rank ?? "—"}</td><td>${stat ? stat.worst.toFixed(4) : "—"}</td>` : ""}${r.drops.map(d => `<td class="${d < 0 ? "negative" : ""}">${d.toFixed(4)}</td>`).join("")}</tr>`;
    }).join("");
  }
  // 刷新所有视图；无有效结果时禁用导出，避免保存过期候选。
  function render() {
    $("fixed-mode").setAttribute("aria-pressed", state.mode === "fixed");
    $("stable-mode").setAttribute("aria-pressed", state.mode === "stable");
    $("stability-controls").hidden = state.mode !== "stable";
    $("cancel").disabled = !running; $("calculate").disabled = running;
    current = null;
    try {
      const rows = LayerSelection.evaluate(data, state);
      const result = stable?.signature === signature() ? stable.result : null;
      const ready = state.mode === "fixed" || result !== null;
      const ordered = state.mode === "stable" && result ? LayerSelection.stableOrder(rows, result) : LayerSelection.fixedOrder(rows);
      const candidates = ready ? ordered.filter(r => r.eligible).slice(0, state.k) : [];
      const selected = new Set(candidates.map(r => r.index));
      current = {rows: ordered, candidates, result: state.mode === "stable" ? result : null, ready};
      $("selected-count").textContent = ready ? candidates.length : "—";
      $("eligible-count").textContent = `目标 ${state.k} / 合格 ${rows.filter(r => r.eligible).length} / 总计 252`;
      $("saving").textContent = ready ? percent(candidates.reduce((s, r) => s + r.saving, 0)) : "—";
      $("max-drop").textContent = candidates.length ? `${Math.max(...candidates.map(r => r.maxDrop)).toFixed(2)} pp` : "—";
      status(ready ? (state.mode === "fixed" ? "按归一化权重即时评分。负掉点按零计入综合分数。" : `已完成 ${result.count} 组权重 · 入选频率用于衡量权重稳定性。`) : (running ? "正在计算权重稳定性…" : stable ? "设置已改变，稳定性结果待更新。" : "请点击「计算稳定性」生成候选。"));
      renderHeatmap(rows, selected, current.result); renderTable(ordered, selected, current.result);
      if (focused !== null) $("detail").textContent = description(rows.find(r => r.index === focused));
      $("detail").style.whiteSpace = "pre-line";
    } catch (error) {
      status(error.message, true);
      ["selected-count", "saving", "max-drop"].forEach(id => {$(id).textContent = "—";});
      ["eligible-count", "legend", "heatmap", "ranking-head", "ranking-body"].forEach(id => {$(id).textContent = "";});
      $("detail").textContent = "修正设置后可查看模块。";
    }
    document.querySelectorAll("[data-export]").forEach(b => {b.disabled = !current?.ready || !current.candidates.length;});
  }
  // 点击热力图定位排名，点击排名同步高亮热力图。
  function focusModule(index, scroll) {
    focused = index;
    document.querySelectorAll("[data-module]").forEach(e => e.classList.toggle("focused", Number(e.dataset.module) === index));
    const row = current?.rows.find(r => r.index === index);
    if (row) $("detail").textContent = description(row);
    if (scroll) $("row-" + index)?.scrollIntoView({block: "nearest", behavior: "smooth"});
  }
  // 执行一次当前设置的稳定性任务，变更或取消通过 revision 丢弃结果。
  async function calculate() {
    try {
      const rows = LayerSelection.evaluate(data, state), grid = LayerSelection.weightGrid(state.datasets);
      if (!grid.length) throw Error("当前上下限没有总和为 100% 的权重组合。");
      const token = ++revision, sig = signature(); running = true; stable = null; render();
      const result = await LayerSelection.stability(rows, grid, state.k, () => token !== revision, (done, total) => {
        $("progress").max = total; $("progress").value = done; $("progress-label").textContent = `${done} / ${total} 组`;
      });
      if (token !== revision) return;
      running = false; stable = {signature: sig, result}; render();
    } catch (error) {running = false; render(); status(error.message, true);}
  }
  // 创建可追溯方案，包含设置、来源、模块顺序和稳定性统计。
  function snapshot() {
    if (!current?.ready || !current.candidates.length) throw Error("当前没有可导出的有效候选。");
    return {schemaVersion: 1, kind: "qcomp_layer_selection", rank: data.rank,
      createdAt: new Date().toISOString(), sources: data.datasets.map(d => ({id: d.id, ...d.source})),
      settings: structuredClone(state), selectedModules: current.candidates.map(r => r.name),
      parameterSaving: current.candidates.reduce((sum, r) => sum + r.saving, 0),
      stability: current.result, scoring: "weighted_positive_drop_pp",
      note: "单层实验推导的候选，联合压缩尚未评测。"};
  }
  // 用浏览器下载保存文件，不依赖服务或网络。
  function download(name, content, mime) {
    const url = URL.createObjectURL(new Blob([content], {type: mime}));
    const anchor = document.createElement("a"); anchor.href = url; anchor.download = name;
    document.body.append(anchor); anchor.click(); anchor.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  // 导出各任务基线、压缩得分、有符号掉点和当前排名。
  function exportCSV() {
    const fields = ["rank", "module", "selected", "eligible", "weighted_drop_pp", "max_drop_pp", "parameter_saving", "frequency", "median_rank", "p95_rank", "worst_weighted_drop_pp"];
    data.datasets.forEach((d, i) => fields.push(`${d.id}:${state.datasets[i].metric}:baseline`, `${d.id}:score`, `${d.id}:drop_pp`));
    const selected = new Set(current.candidates.map(r => r.name));
    const rows = current.rows.map((r, index) => {
      const stat = current.result?.stats[r.index];
      return [r.eligible ? index + 1 : "", r.name, selected.has(r.name), r.eligible, r.score, r.maxDrop, r.saving, stat?.frequency ?? "", stat?.medianRank ?? "", stat?.p95Rank ?? "", stat?.worst ?? "", ...data.datasets.flatMap((d, i) => [d.baseline[state.datasets[i].metric], d.values[state.datasets[i].metric][r.index], r.drops[i]])];
    });
    download("layer-ranking.csv", "\ufeff" + [fields, ...rows].map(row => row.map(v => '"' + String(v).replaceAll('"', '""') + '"').join(",")).join("\r\n"), "text/csv;charset=utf-8");
  }
  // 比较 A/B 快照，保留各自完整设置并展示有序增删模块。
  function renderComparison() {
    const {A, B} = snapshots;
    let text = ["A", "B"].map(key => {
      const s = snapshots[key];
      return s ? `<details><summary>方案 ${key} · ${s.selectedModules.length} 个模块 · 参数节省 ${percent(s.parameterSaving)}</summary><pre>${escape(JSON.stringify(s.settings, null, 2))}</pre></details>` : `<span>方案 ${key} 尚未保存。 </span>`;
    }).join("");
    if (A && B) {
      const a = new Set(A.selectedModules), b = new Set(B.selectedModules);
      const added = B.selectedModules.filter(n => !a.has(n)), removed = A.selectedModules.filter(n => !b.has(n));
      text += `<p>共同模块 ${A.selectedModules.filter(n => b.has(n)).length} · B 新增 ${added.length} · B 移除 ${removed.length} · 参数节省差 ${(100 * (B.parameterSaving - A.parameterSaving)).toFixed(2)} 个百分点</p><details><summary>查看新增 / 移除模块</summary><pre>新增：\n${escape(added.join("\n") || "无")}\n\n移除：\n${escape(removed.join("\n") || "无")}</pre></details>`;
    }
    $("comparison").innerHTML = text;
  }
  // 导入时重新评分并核对模块顺序，拒绝伪造或过期候选；成功前不改变页面设置。
  async function importSnapshot(value) {
    const settings = LayerSelection.importSettings(data, value);
    const token = ++revision; running = false;
    const rows = LayerSelection.evaluate(data, settings);
    let result = null;
    if (settings.mode === "stable") {
      status("正在验证导入方案的稳定性排名…");
      result = await LayerSelection.stability(rows, LayerSelection.weightGrid(settings.datasets), settings.k, () => revision !== token);
      if (!result || token !== revision) return;
    }
    const ordered = result ? LayerSelection.stableOrder(rows, result) : LayerSelection.fixedOrder(rows);
    const names = ordered.filter(r => r.eligible).slice(0, settings.k).map(r => r.name);
    if (JSON.stringify(names) !== JSON.stringify(value.selectedModules)) throw Error("候选名单与设置重新计算的结果不一致。");
    state = settings; stable = result ? {signature: signature(), result} : null;
    $("k").value = state.k; $("cap").value = state.cap; $("cap-enabled").checked = state.capEnabled; $("cap").disabled = !state.capEnabled;
    renderCards(); render(); status("方案已导入，并通过来源与候选名单校验。");
  }

  $("datasets").addEventListener("input", event => {
    const field = event.target.dataset.field, card = event.target.closest("[data-index]");
    if (!field || !card) return;
    const entry = state.datasets[Number(card.dataset.index)];
    entry[field] = field === "enabled" ? event.target.checked : field === "metric" ? event.target.value : event.target.valueAsNumber;
    if (field === "enabled") LayerSelection.resetRanges(state);
    changed(field === "enabled");
  });
  $("equal").onclick = () => {state.datasets.forEach(d => {d.weight = 20;}); changed(true);};
  $("k").oninput = event => {state.k = event.target.valueAsNumber; changed();};
  document.querySelectorAll("[data-k]").forEach(b => {b.onclick = () => {state.k = Number(b.dataset.k); $("k").value = state.k; changed();};});
  $("cap-enabled").onchange = event => {state.capEnabled = event.target.checked; $("cap").disabled = !state.capEnabled; changed();};
  $("cap").oninput = event => {state.cap = event.target.valueAsNumber; changed();};
  $("fixed-mode").onclick = () => {state.mode = "fixed"; changed(true);};
  $("stable-mode").onclick = () => {state.mode = "stable"; changed(true);};
  $("calculate").onclick = calculate;
  $("cancel").onclick = () => {revision++; running = false; render(); status("计算已取消，未生成新的稳定性方案。");};
  ["heatmap", "ranking-body"].forEach(id => {
    $(id).addEventListener("click", event => {const cell = event.target.closest("[data-module]"); if (cell) focusModule(Number(cell.dataset.module), id === "heatmap");});
    $(id).addEventListener("keydown", event => {if (["Enter", " "].includes(event.key) && event.target.matches("[data-module]")) {event.preventDefault(); focusModule(Number(event.target.dataset.module), id === "heatmap");}});
  });
  ["A", "B"].forEach(key => {$("save-" + key.toLowerCase()).onclick = () => {snapshots[key] = snapshot(); renderComparison();};});
  $("export-json").onclick = () => download("layer-selection.json", JSON.stringify(snapshot(), null, 2), "application/json");
  $("export-csv").onclick = exportCSV;
  $("import-json").onclick = () => $("import-file").click();
  $("import-file").onchange = async event => {
    const file = event.target.files[0]; if (!file) return;
    try {await importSnapshot(JSON.parse(await file.text()));} catch (error) {render(); status(error.message, true);}
    event.target.value = "";
  };
  renderCards(); render();
})();
