const state = {
  summary: null,
  facts: [],
  chunks: [],
  voice: [],
  synth: [],
  themes: [],
  relations: [],
  pipelineSummary: null,
  pipelineEvents: [],
  llmJobs: [],
  llmRuns: [],
  llmTraces: [],
  timelineOverview: null,
  timelineNodes: [],
  timelineEdges: [],
  timelineGroups: [],
  selectedFactId: null,
  selectedChunkId: null,
  selectedChunk: null,
  selectedVoiceId: null,
  selectedSynthId: null,
  selectedThemeId: null,
  selectedLlmTraceRef: null,
  selectedPipelineDetail: null,
  selectedTimelineDetail: null,
  selectedFactIds: new Set(),
};

const byId = (id) => document.getElementById(id);
const qsa = (selector, root = document) => Array.from(root.querySelectorAll(selector));

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const text = await response.text();
  let data = {};
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    data = { raw: text };
  }
  if (!response.ok) {
    throw new Error(data.error || data.raw || `HTTP ${response.status}`);
  }
  return data;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function parseIds(value) {
  return String(value || "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function joinIds(value) {
  return Array.isArray(value) ? value.join(", ") : "";
}

function prettyJson(value) {
  return JSON.stringify(value ?? {}, null, 2);
}

function reviewPill(status) {
  const value = status || "pending";
  return `<span class="pill ${value}">${value}</span>`;
}

function toast(message, isError = false) {
  const node = byId("toast");
  node.textContent = message;
  node.classList.remove("hidden");
  node.style.background = isError ? "#7f2d2d" : "#2d2a26";
  clearTimeout(node._timer);
  node._timer = setTimeout(() => node.classList.add("hidden"), 3200);
}

function setTab(tab) {
  qsa(".tab").forEach((item) => item.classList.toggle("is-active", item.dataset.tab === tab));
  qsa(".tab-panel").forEach((item) => item.classList.toggle("is-active", item.dataset.panel === tab));
}

function activateTab(tab) {
  setTab(tab);
  const loaders = {
    chunks: loadChunks,
    pipeline: loadPipelineExplorer,
    timeline: loadTimelineExplorer,
    llm: loadLlmExplorer,
  };
  const loader = loaders[tab];
  if (loader) {
    loader().catch((err) => toast(err.message, true));
  }
}

function currentTabId() {
  return qsa(".tab.is-active")[0]?.dataset.tab || "facts";
}

function fillSelect(select, options, includeAll = true, current = "") {
  const values = includeAll ? [""] : [];
  values.push(...options);
  select.innerHTML = values
    .map((value) => `<option value="${escapeHtml(value)}">${value || "Все"}</option>`)
    .join("");
  select.value = current;
}

function renderSummary() {
  if (!state.summary) return;
  const cards = [
    ["Факты", state.summary.counts.facts],
    ["Chunks", state.summary.counts.chunks || 0],
    ["Voice", state.summary.counts.voice],
    ["Synth", state.summary.counts.synth],
    ["Темы", state.summary.counts.themes],
    ["Связи", state.summary.counts.relations],
    ["Events", state.summary.counts.metadata_events || 0],
    ["LLM jobs", state.summary.counts.llm_jobs || 0],
    ["LLM traces", state.summary.counts.llm_traces || 0],
    ["Timeline nodes", state.summary.counts.timeline_nodes || 0],
  ];
  byId("summaryGrid").innerHTML = cards
    .map(([label, value]) => `<div class="summary-card"><div class="muted">${label}</div><strong>${value}</strong></div>`)
    .join("");
  fillSelect(byId("factCategory"), state.summary.categories || []);
  fillSelect(byId("factBook"), state.summary.books || []);
  fillSelect(byId("factReview"), state.summary.review_statuses || []);
  fillSelect(byId("chunkBook"), state.summary.books || []);
  fillSelect(byId("voiceReview"), state.summary.review_statuses || []);
  fillSelect(byId("synthReview"), state.summary.review_statuses || []);
  fillSelect(byId("timelineBook"), state.summary.books || []);
}

async function loadSummary() {
  state.summary = await api("/api/summary");
  renderSummary();
}

async function loadFacts() {
  const params = new URLSearchParams({
    search: byId("factSearch").value,
    category: byId("factCategory").value,
    book: byId("factBook").value,
    review_status: byId("factReview").value,
    limit: "500",
  });
  const payload = await api(`/api/facts?${params.toString()}`);
  state.facts = payload.items || [];
  renderFacts();
  if (state.selectedFactId) {
    await loadFactDetail(state.selectedFactId).catch(() => {});
  }
}

function renderFacts() {
  byId("factsCount").textContent = `${state.facts.length} записей`;
  byId("factsList").innerHTML = state.facts
    .map((item) => `
      <article class="list-item ${item.id === state.selectedFactId ? "is-selected" : ""}" data-fact-item="${escapeHtml(item.id)}">
        <div class="list-item-row">
          <input type="checkbox" class="fact-select" data-fact-id="${escapeHtml(item.id)}" ${state.selectedFactIds.has(item.id) ? "checked" : ""}>
          <div class="list-item-main">
            <div class="list-item-title">${escapeHtml(item.subject || "(без subject)")}</div>
            <div>${escapeHtml(item.fact_preview || item.fact || "")}</div>
            <div class="list-item-meta">
              <span>${escapeHtml(item.category || "unknown")}</span>
              <span>${escapeHtml(item.time_scope || "unclear")}</span>
              <span>${escapeHtml(item.source_book || "")}</span>
              ${reviewPill(item.review_status)}
            </div>
          </div>
        </div>
      </article>
    `)
    .join("");

  qsa("[data-fact-item]", byId("factsList")).forEach((node) =>
    node.addEventListener("click", () => openFact(node.dataset.factItem))
  );
  qsa(".fact-select", byId("factsList")).forEach((box) =>
    box.addEventListener("click", (event) => {
      event.stopPropagation();
      const id = box.dataset.factId;
      if (box.checked) state.selectedFactIds.add(id);
      else state.selectedFactIds.delete(id);
    })
  );
}

async function openFact(id) {
  state.selectedFactId = id;
  renderFacts();
  await loadFactDetail(id);
}

async function loadFactDetail(id) {
  const payload = await api(`/api/facts/${encodeURIComponent(id)}`);
  const item = payload.item;
  const form = byId("factForm");
  byId("factDetailEmpty").classList.add("hidden");
  form.classList.remove("hidden");

  [
    "id",
    "category",
    "subject",
    "time_scope",
    "source_book",
    "chapter",
    "review_status",
    "fact",
    "review_note",
    "source_excerpt",
  ].forEach((name) => {
    const field = form.elements.namedItem(name);
    if (field) field.value = item[name] ?? "";
  });
  form.elements.namedItem("chunk_idx").value = item.chunk_idx ?? "";
  form.elements.namedItem("review_score").value = item.review_score ?? "";
  form.elements.namedItem("theme_ids").value = joinIds(item.theme_ids);

  const relations = payload.relations || [];
  byId("factRelations").innerHTML = relations.length
    ? relations
        .map((rel) => `
          <div class="list-item">
            <div><strong>${escapeHtml(rel.relation_type)}</strong> → ${escapeHtml(rel.target_fact_id === item.id ? rel.source_fact_id : rel.target_fact_id)}</div>
            <div class="muted">${escapeHtml(rel.note || "")}</div>
            <div class="form-actions">
              <button type="button" class="relation-edit-btn" data-relation-id="${escapeHtml(rel.id)}">Редактировать</button>
              <button type="button" class="relation-delete-btn danger" data-relation-id="${escapeHtml(rel.id)}">Удалить</button>
            </div>
          </div>
        `)
        .join("")
    : `<div class="empty-state">Связей пока нет.</div>`;
  qsa(".relation-delete-btn", byId("factRelations")).forEach((btn) =>
    btn.addEventListener("click", () => deleteRelation(btn.dataset.relationId))
  );
  qsa(".relation-edit-btn", byId("factRelations")).forEach((btn) =>
    btn.addEventListener("click", () => editRelation(btn.dataset.relationId, relations))
  );

  const linkedSamples = payload.linked_samples || [];
  byId("factLinkedSamples").innerHTML = linkedSamples.length
    ? linkedSamples
        .map((sample) => `
          <div class="list-item">
            <div><strong>${escapeHtml(sample.kind)}</strong></div>
            <div>${escapeHtml(sample.assistant_preview || "")}</div>
          </div>
        `)
        .join("")
    : `<div class="empty-state">Связанных примеров пока нет.</div>`;
}

async function saveFact() {
  const form = byId("factForm");
  const payload = {
    category: form.elements.namedItem("category").value,
    subject: form.elements.namedItem("subject").value,
    fact: form.elements.namedItem("fact").value,
    time_scope: form.elements.namedItem("time_scope").value,
    source_book: form.elements.namedItem("source_book").value,
    chapter: form.elements.namedItem("chapter").value,
    source_excerpt: form.elements.namedItem("source_excerpt").value,
    review_status: form.elements.namedItem("review_status").value,
    review_note: form.elements.namedItem("review_note").value,
    theme_ids: parseIds(form.elements.namedItem("theme_ids").value),
    chunk_idx: form.elements.namedItem("chunk_idx").value ? Number(form.elements.namedItem("chunk_idx").value) : null,
    review_score: form.elements.namedItem("review_score").value ? Number(form.elements.namedItem("review_score").value) : null,
  };
  const id = form.elements.namedItem("id").value;
  if (id) {
    await api(`/api/facts/${encodeURIComponent(id)}`, { method: "PATCH", body: JSON.stringify(payload) });
  } else {
    const created = await api("/api/facts", { method: "POST", body: JSON.stringify(payload) });
    state.selectedFactId = created.id;
  }
  toast("Факт сохранён");
  await loadSummary();
  await loadFacts();
}

function newFact() {
  state.selectedFactId = null;
  const form = byId("factForm");
  byId("factDetailEmpty").classList.add("hidden");
  form.classList.remove("hidden");
  form.reset();
  form.elements.namedItem("time_scope").value = "unclear";
  form.elements.namedItem("review_status").value = "pending";
}

async function deleteFact() {
  const form = byId("factForm");
  const id = form.elements.namedItem("id").value;
  if (!id) return;
  if (!window.confirm("Удалить факт?")) return;
  await api(`/api/facts/${encodeURIComponent(id)}`, { method: "DELETE" });
  state.selectedFactId = null;
  form.classList.add("hidden");
  byId("factDetailEmpty").classList.remove("hidden");
  toast("Факт удалён");
  await loadSummary();
  await loadFacts();
}

async function reanalyzeFacts(bundle) {
  const factIds = Array.from(state.selectedFactIds);
  if (!factIds.length) {
    toast("Сначала выбери один или несколько фактов", true);
    return;
  }
  const result = await api("/api/facts/reanalyze", {
    method: "POST",
    body: JSON.stringify({ fact_ids: factIds, bundle }),
  });
  toast(`Получено предложений: ${(result.suggestions || []).length}`);
  console.log("reanalyze", result);
}

async function createRelation(event) {
  event.preventDefault();
  if (!state.selectedFactId) {
    toast("Сначала выбери факт", true);
    return;
  }
  const form = byId("relationCreateForm");
  await api("/api/relations", {
    method: "POST",
    body: JSON.stringify({
      source_fact_id: state.selectedFactId,
      target_fact_id: form.elements.namedItem("target_fact_id").value,
      relation_type: form.elements.namedItem("relation_type").value,
      note: form.elements.namedItem("note").value,
    }),
  });
  form.reset();
  form.elements.namedItem("relation_type").value = "related_to";
  toast("Связь создана");
  await loadSummary();
  await loadThemesAndRelations();
  await loadFactDetail(state.selectedFactId);
}

async function editRelation(relationId, relations) {
  const current = relations.find((item) => item.id === relationId);
  if (!current) return;
  const relationType = window.prompt("Тип связи", current.relation_type || "related_to");
  if (relationType === null) return;
  const note = window.prompt("Комментарий", current.note || "");
  if (note === null) return;
  await api(`/api/relations/${encodeURIComponent(relationId)}`, {
    method: "PATCH",
    body: JSON.stringify({ relation_type: relationType, note }),
  });
  toast("Связь обновлена");
  await loadThemesAndRelations();
  await loadFactDetail(state.selectedFactId);
}

async function deleteRelation(relationId) {
  if (!window.confirm("Удалить связь?")) return;
  await api(`/api/relations/${encodeURIComponent(relationId)}`, { method: "DELETE" });
  toast("Связь удалена");
  await loadSummary();
  await loadThemesAndRelations();
  if (state.selectedFactId) await loadFactDetail(state.selectedFactId);
}

async function generateFromFacts(kind) {
  const factIds = Array.from(state.selectedFactIds);
  if (!factIds.length) {
    toast("Нужно выбрать факты", true);
    return;
  }
  const created = await api("/api/samples/generate", {
    method: "POST",
    body: JSON.stringify({
      fact_ids: factIds,
      kind,
      instruction: byId("generateInstruction").value,
    }),
  });
  toast(`${kind} пример создан`);
  await loadSummary();
  await loadSamples(kind);
  if (kind === "voice") {
    state.selectedVoiceId = created.id;
    setTab("voice");
  } else {
    state.selectedSynthId = created.id;
    setTab("synth");
  }
}

async function loadSamples(kind) {
  const searchId = kind === "voice" ? "voiceSearch" : "synthSearch";
  const reviewId = kind === "voice" ? "voiceReview" : "synthReview";
  const params = new URLSearchParams({
    kind,
    search: byId(searchId).value,
    review_status: byId(reviewId).value,
    limit: "500",
  });
  const payload = await api(`/api/samples?${params.toString()}`);
  state[kind] = payload.items || [];
  renderSamples(kind);
  const selectedId = kind === "voice" ? state.selectedVoiceId : state.selectedSynthId;
  if (selectedId) await loadSampleDetail(kind, selectedId).catch(() => {});
}

function renderSamples(kind) {
  const selectedId = kind === "voice" ? state.selectedVoiceId : state.selectedSynthId;
  const listId = kind === "voice" ? "voiceList" : "synthList";
  const countId = kind === "voice" ? "voiceCount" : "synthCount";
  byId(countId).textContent = `${state[kind].length} записей`;
  byId(listId).innerHTML = state[kind]
    .map((item) => `
      <article class="list-item ${item.id === selectedId ? "is-selected" : ""}" data-sample-kind="${kind}" data-sample-id="${escapeHtml(item.id)}">
        <div class="list-item-title">${escapeHtml(item.user_preview || "(без user)")}</div>
        <div>${escapeHtml(item.assistant_preview || "")}</div>
        <div class="list-item-meta">
          <span>${escapeHtml(item.source_book || "")}</span>
          ${reviewPill(item.review_status)}
        </div>
      </article>
    `)
    .join("");
  qsa(`[data-sample-kind="${kind}"]`, byId(listId)).forEach((node) =>
    node.addEventListener("click", () => openSample(kind, node.dataset.sampleId))
  );
}

async function openSample(kind, id) {
  if (kind === "voice") state.selectedVoiceId = id;
  else state.selectedSynthId = id;
  renderSamples(kind);
  await loadSampleDetail(kind, id);
}

async function loadSampleDetail(kind, id) {
  const payload = await api(`/api/samples/${encodeURIComponent(id)}`);
  const item = payload.item;
  const form = byId(kind === "voice" ? "voiceForm" : "synthForm");
  byId(kind === "voice" ? "voiceDetailEmpty" : "synthDetailEmpty").classList.add("hidden");
  form.classList.remove("hidden");
  ["id", "system", "user", "assistant", "source_book", "chapter", "review_status", "review_note", "source_excerpt"].forEach((name) => {
    const field = form.elements.namedItem(name);
    if (field) field.value = item[name] ?? "";
  });
  form.elements.namedItem("chunk_idx").value = item.chunk_idx ?? "";
  form.elements.namedItem("review_score").value = item.review_score ?? "";
  form.elements.namedItem("linked_fact_ids").value = joinIds(item.linked_fact_ids);
  form.elements.namedItem("theme_ids").value = joinIds(item.theme_ids);
}

function newSample(kind) {
  const form = byId(kind === "voice" ? "voiceForm" : "synthForm");
  byId(kind === "voice" ? "voiceDetailEmpty" : "synthDetailEmpty").classList.add("hidden");
  form.classList.remove("hidden");
  form.reset();
  form.elements.namedItem("review_status").value = "pending";
}

async function saveSample(kind) {
  const form = byId(kind === "voice" ? "voiceForm" : "synthForm");
  const payload = {
    kind,
    system: form.elements.namedItem("system").value,
    user: form.elements.namedItem("user").value,
    assistant: form.elements.namedItem("assistant").value,
    source_book: form.elements.namedItem("source_book").value,
    chapter: form.elements.namedItem("chapter").value,
    source_excerpt: form.elements.namedItem("source_excerpt").value,
    review_status: form.elements.namedItem("review_status").value,
    review_note: form.elements.namedItem("review_note").value,
    linked_fact_ids: parseIds(form.elements.namedItem("linked_fact_ids").value),
    theme_ids: parseIds(form.elements.namedItem("theme_ids").value),
    chunk_idx: form.elements.namedItem("chunk_idx").value ? Number(form.elements.namedItem("chunk_idx").value) : null,
    review_score: form.elements.namedItem("review_score").value ? Number(form.elements.namedItem("review_score").value) : null,
  };
  const id = form.elements.namedItem("id").value;
  if (id) {
    await api(`/api/samples/${encodeURIComponent(id)}`, { method: "PATCH", body: JSON.stringify(payload) });
  } else {
    const created = await api("/api/samples", { method: "POST", body: JSON.stringify(payload) });
    if (kind === "voice") state.selectedVoiceId = created.id;
    else state.selectedSynthId = created.id;
  }
  toast(`${kind} сохранён`);
  await loadSummary();
  await loadSamples(kind);
}

async function deleteSample(kind) {
  const form = byId(kind === "voice" ? "voiceForm" : "synthForm");
  const id = form.elements.namedItem("id").value;
  if (!id) return;
  if (!window.confirm("Удалить пример?")) return;
  await api(`/api/samples/${encodeURIComponent(id)}`, { method: "DELETE" });
  if (kind === "voice") state.selectedVoiceId = null;
  else state.selectedSynthId = null;
  form.classList.add("hidden");
  byId(kind === "voice" ? "voiceDetailEmpty" : "synthDetailEmpty").classList.remove("hidden");
  toast(`${kind} удалён`);
  await loadSummary();
  await loadSamples(kind);
}

async function loadThemesAndRelations() {
  const [themesPayload, relationsPayload] = await Promise.all([
    api("/api/themes"),
    api("/api/relations"),
  ]);
  state.themes = themesPayload.items || [];
  state.relations = relationsPayload.items || [];
  renderThemes();
  renderRelations();
}

function renderThemes() {
  byId("themesList").innerHTML = state.themes
    .map((item) => `
      <article class="list-item ${item.id === state.selectedThemeId ? "is-selected" : ""}" data-theme-id="${escapeHtml(item.id)}">
        <div class="list-item-title">${escapeHtml(item.name)}</div>
        <div class="muted">${escapeHtml(item.description || "")}</div>
      </article>
    `)
    .join("");
  qsa("[data-theme-id]", byId("themesList")).forEach((node) =>
    node.addEventListener("click", () => openTheme(node.dataset.themeId))
  );
  const options = state.themes
    .map((item) => `<option value="${escapeHtml(item.id)}">${escapeHtml(item.name)}</option>`)
    .join("");
  byId("mergeSourceTheme").innerHTML = options;
  byId("mergeTargetTheme").innerHTML = options;
}

function renderRelations() {
  byId("relationsList").innerHTML = state.relations.length
    ? state.relations
        .slice(0, 200)
        .map((item) => `
          <div class="list-item">
            <div><strong>${escapeHtml(item.relation_type)}</strong></div>
            <div class="muted">${escapeHtml(item.source_fact_id)} → ${escapeHtml(item.target_fact_id)}</div>
            <div>${escapeHtml(item.note || "")}</div>
          </div>
        `)
        .join("")
    : `<div class="empty-state">Связей пока нет.</div>`;
}

function openTheme(id) {
  state.selectedThemeId = id;
  renderThemes();
  const theme = state.themes.find((item) => item.id === id);
  if (!theme) return;
  const form = byId("themeForm");
  byId("themeDetailEmpty").classList.add("hidden");
  form.classList.remove("hidden");
  ["id", "name", "description", "color"].forEach((name) => {
    form.elements.namedItem(name).value = theme[name] ?? "";
  });
}

async function saveTheme() {
  const form = byId("themeForm");
  const payload = {
    name: form.elements.namedItem("name").value,
    description: form.elements.namedItem("description").value,
    color: form.elements.namedItem("color").value,
  };
  const id = form.elements.namedItem("id").value;
  if (id) {
    await api(`/api/themes/${encodeURIComponent(id)}`, { method: "PATCH", body: JSON.stringify(payload) });
  } else {
    const created = await api("/api/themes", { method: "POST", body: JSON.stringify(payload) });
    state.selectedThemeId = created.id;
  }
  toast("Тема сохранена");
  await loadSummary();
  await loadThemesAndRelations();
}

async function deleteTheme() {
  const form = byId("themeForm");
  const id = form.elements.namedItem("id").value;
  if (!id) return;
  if (!window.confirm("Удалить тему?")) return;
  await api(`/api/themes/${encodeURIComponent(id)}`, { method: "DELETE" });
  state.selectedThemeId = null;
  form.classList.add("hidden");
  byId("themeDetailEmpty").classList.remove("hidden");
  toast("Тема удалена");
  await loadSummary();
  await loadThemesAndRelations();
}

async function mergeThemes(event) {
  event.preventDefault();
  const sourceThemeId = byId("mergeSourceTheme").value;
  const targetThemeId = byId("mergeTargetTheme").value;
  if (!sourceThemeId || !targetThemeId) {
    toast("Нужно выбрать две темы", true);
    return;
  }
  await api("/api/themes/merge", {
    method: "POST",
    body: JSON.stringify({ source_theme_id: sourceThemeId, target_theme_id: targetThemeId }),
  });
  toast("Темы объединены");
  await loadSummary();
  await loadFacts();
  await loadSamples("voice");
  await loadSamples("synth");
  await loadThemesAndRelations();
}

async function loadChunks() {
  const params = new URLSearchParams({
    search: byId("chunkSearch").value,
    book: byId("chunkBook").value,
    has_dialogues: byId("chunkHasDialogues").value,
    has_knowledge: byId("chunkHasKnowledge").value,
    limit: "500",
  });
  const payload = await api(`/api/chunks?${params.toString()}`);
  state.chunks = payload.items || [];
  renderChunks();
  if (state.selectedChunkId) {
    await loadChunkDetail(state.selectedChunkId).catch(() => {});
  }
}

function renderChunks() {
  byId("chunkCount").textContent = `${state.chunks.length} записей`;
  byId("chunksList").innerHTML = state.chunks
    .map((item) => `
      <article class="list-item ${item.id === state.selectedChunkId ? "is-selected" : ""}" data-chunk-item="${escapeHtml(item.id)}">
        <div class="list-item-title">${escapeHtml(item.source_book || "Без книги")}</div>
        <div>${escapeHtml(item.preview || "")}</div>
        <div class="list-item-meta">
          <span>${escapeHtml(item.chapter || "")}</span>
          <span>#${escapeHtml(item.chunk_idx ?? "")}</span>
          <span>dialogues=${escapeHtml(item.dialogue_count ?? 0)}</span>
          <span>facts=${escapeHtml(item.knowledge_count ?? 0)}</span>
        </div>
      </article>
    `)
    .join("");
  qsa("[data-chunk-item]", byId("chunksList")).forEach((node) =>
    node.addEventListener("click", () => openChunk(node.dataset.chunkItem))
  );
}

async function openChunk(id) {
  state.selectedChunkId = id;
  renderChunks();
  await loadChunkDetail(id);
}

async function loadChunkDetail(id) {
  const payload = await api(`/api/chunks/${encodeURIComponent(id)}`);
  const item = payload.item || {};
  state.selectedChunk = item;
  byId("chunkDetailEmpty").classList.add("hidden");
  byId("chunkDetail").classList.remove("hidden");
  byId("chunkDetailId").value = item.id || "";
  byId("chunkDetailBook").value = item.source_book || "";
  byId("chunkDetailChapter").value = item.chapter || "";
  byId("chunkDetailIdx").value = item.chunk_idx ?? "";
  byId("chunkDetailDialogues").value = item.dialogue_count ?? 0;
  byId("chunkDetailKnowledge").value = item.knowledge_count ?? 0;
  byId("chunkDetailText").value = item.source_excerpt || "";

  const linkedFacts = payload.linked_facts || [];
  byId("chunkLinkedFacts").innerHTML = linkedFacts.length
    ? linkedFacts
        .map((fact) => `
          <div class="list-item">
            <div><strong>${escapeHtml(fact.subject || "")}</strong></div>
            <div>${escapeHtml(fact.fact_preview || fact.fact || "")}</div>
            <div class="form-actions">
              <button type="button" class="chunk-open-fact" data-fact-id="${escapeHtml(fact.id)}">Открыть факт</button>
            </div>
          </div>
        `)
        .join("")
    : `<div class="empty-state">Связанных фактов пока нет.</div>`;
  qsa(".chunk-open-fact", byId("chunkLinkedFacts")).forEach((btn) =>
    btn.addEventListener("click", () => {
      setTab("facts");
      openFact(btn.dataset.factId).catch((err) => toast(err.message, true));
    })
  );

  const linkedSamples = payload.linked_samples || [];
  byId("chunkLinkedSamples").innerHTML = linkedSamples.length
    ? linkedSamples
        .map((sample) => `
          <div class="list-item">
            <div><strong>${escapeHtml(sample.kind || "")}</strong></div>
            <div>${escapeHtml(sample.assistant_preview || "")}</div>
          </div>
        `)
        .join("")
    : `<div class="empty-state">Связанных примеров пока нет.</div>`;
}

function createFactFromSelectedChunk() {
  if (!state.selectedChunk) {
    toast("Сначала выбери чанк", true);
    return;
  }
  newFact();
  const form = byId("factForm");
  form.elements.namedItem("source_book").value = state.selectedChunk.source_book || "";
  form.elements.namedItem("chapter").value = state.selectedChunk.chapter || "";
  form.elements.namedItem("chunk_idx").value = state.selectedChunk.chunk_idx ?? "";
  form.elements.namedItem("source_excerpt").value = state.selectedChunk.source_excerpt || "";
  setTab("facts");
}

async function loadPipelineExplorer() {
  const search = byId("pipelineEventSearch").value;
  const [summary, events, jobs] = await Promise.all([
    api("/api/pipeline/summary"),
    api(`/api/pipeline/events?${new URLSearchParams({ search, limit: "300" }).toString()}`),
    api("/api/llm/jobs?limit=200"),
  ]);
  state.pipelineSummary = summary;
  state.pipelineEvents = events.items || [];
  state.llmJobs = jobs.items || [];
  renderPipelineExplorer();
}

function renderPipelineExplorer() {
  const metadata = state.pipelineSummary?.metadata || {};
  byId("pipelineMetadata").value = prettyJson(metadata);
  byId("pipelineEventCount").textContent = `${state.pipelineEvents.length} событий`;
  byId("pipelineJobsCount").textContent = `${state.llmJobs.length} jobs`;
  byId("pipelineEventsList").innerHTML = state.pipelineEvents.length
    ? state.pipelineEvents
        .map((item) => `
          <article class="list-item pipeline-event" data-pipeline-event='${escapeHtml(JSON.stringify(item))}'>
            <div class="list-item-title">${escapeHtml(item.type || "event")}</div>
            <div>${escapeHtml(item.message || item.current_stage || "")}</div>
            <div class="list-item-meta">
              <span>${escapeHtml(item.status || "")}</span>
              <span>${escapeHtml(item.current_book || "")}</span>
            </div>
          </article>
        `)
        .join("")
    : `<div class="empty-state">Событий пока нет.</div>`;
  qsa(".pipeline-event", byId("pipelineEventsList")).forEach((node) =>
    node.addEventListener("click", () => {
      const payload = JSON.parse(node.dataset.pipelineEvent || "{}");
      state.selectedPipelineDetail = payload;
      byId("pipelineDetail").value = prettyJson(payload);
    })
  );
  byId("pipelineJobsList").innerHTML = state.llmJobs.length
    ? state.llmJobs
        .map((item) => `
          <article class="list-item pipeline-job" data-job-id="${escapeHtml(item.id)}">
            <div class="list-item-title">${escapeHtml(item.job_type || item.id || "")}</div>
            <div>${escapeHtml(item.request_preview || "")}</div>
            <div class="list-item-meta">
              <span>${escapeHtml(item.ts || "")}</span>
            </div>
          </article>
        `)
        .join("")
    : `<div class="empty-state">LLM jobs пока нет.</div>`;
  qsa(".pipeline-job", byId("pipelineJobsList")).forEach((node) =>
    node.addEventListener("click", async () => {
      const payload = await api(`/api/llm/jobs/${encodeURIComponent(node.dataset.jobId)}`);
      state.selectedPipelineDetail = payload;
      byId("pipelineDetail").value = prettyJson(payload);
    })
  );
}

async function loadTimelineExplorer() {
  const search = byId("timelineSearch").value;
  const book = byId("timelineBook").value;
  const nodeType = byId("timelineNodeType").value;
  const [overview, nodesPayload, edgesPayload, groupsPayload] = await Promise.all([
    api("/api/timeline/overview"),
    api(`/api/timeline/nodes?${new URLSearchParams({ search, node_type: nodeType, limit: "300" }).toString()}`),
    api(`/api/timeline/edges?${new URLSearchParams({ search, limit: "300" }).toString()}`),
    api(`/api/timeline/groups?${new URLSearchParams({ search, book, limit: "300" }).toString()}`),
  ]);
  state.timelineOverview = overview;
  state.timelineNodes = nodesPayload.items || [];
  state.timelineEdges = edgesPayload.items || [];
  state.timelineGroups = groupsPayload.items || [];
  renderTimelineExplorer();
}

function renderTimelineExplorer() {
  const overview = state.timelineOverview || { counts: {}, node_types: {} };
  fillSelect(byId("timelineNodeType"), Object.keys(overview.node_types || {}), true, byId("timelineNodeType").value);
  const cards = [
    ["Nodes", overview.counts?.nodes || 0],
    ["Edges", overview.counts?.edges || 0],
    ["Groups", overview.counts?.groups || 0],
  ];
  byId("timelineSummaryGrid").innerHTML = cards
    .map(([label, value]) => `<div class="summary-card"><div class="muted">${label}</div><strong>${value}</strong></div>`)
    .join("");
  byId("timelineGroupsCount").textContent = `${state.timelineGroups.length} groups`;
  byId("timelineNodesCount").textContent = `${state.timelineNodes.length} nodes`;
  byId("timelineEdgesCount").textContent = `${state.timelineEdges.length} edges`;
  byId("timelineGroupsList").innerHTML = state.timelineGroups.length
    ? state.timelineGroups
        .map((item) => `
          <article class="list-item timeline-group" data-timeline-group='${escapeHtml(JSON.stringify(item.raw || item))}'>
            <div class="list-item-title">${escapeHtml(item.chapter || item.book_name || "")}</div>
            <div>${escapeHtml(item.preview || "")}</div>
            <div class="list-item-meta">
              <span>${escapeHtml(item.book_name || "")}</span>
              <span>facts=${escapeHtml(item.fact_count ?? 0)}</span>
              <span>events=${escapeHtml(item.event_count ?? 0)}</span>
            </div>
          </article>
        `)
        .join("")
    : `<div class="empty-state">Timeline groups пока нет.</div>`;
  qsa(".timeline-group", byId("timelineGroupsList")).forEach((node) =>
    node.addEventListener("click", () => {
      const payload = JSON.parse(node.dataset.timelineGroup || "{}");
      state.selectedTimelineDetail = payload;
      byId("timelineDetail").value = prettyJson(payload);
    })
  );
  byId("timelineNodesList").innerHTML = state.timelineNodes.length
    ? state.timelineNodes
        .map((item) => `
          <article class="list-item timeline-node" data-timeline-node='${escapeHtml(JSON.stringify(item))}'>
            <div class="list-item-title">${escapeHtml(item.label || item.id || "")}</div>
            <div class="list-item-meta">
              <span>${escapeHtml(item.type || "")}</span>
              <span>${escapeHtml(item.chapter || item.book || "")}</span>
            </div>
          </article>
        `)
        .join("")
    : `<div class="empty-state">Timeline nodes пока нет.</div>`;
  qsa(".timeline-node", byId("timelineNodesList")).forEach((node) =>
    node.addEventListener("click", () => {
      const payload = JSON.parse(node.dataset.timelineNode || "{}");
      state.selectedTimelineDetail = payload;
      byId("timelineDetail").value = prettyJson(payload);
    })
  );
  byId("timelineEdgesList").innerHTML = state.timelineEdges.length
    ? state.timelineEdges
        .map((item) => `
          <article class="list-item timeline-edge" data-timeline-edge='${escapeHtml(JSON.stringify(item))}'>
            <div class="list-item-title">${escapeHtml(item.type || "")}</div>
            <div>${escapeHtml(item.source_label || item.source || "")} → ${escapeHtml(item.target_label || item.target || "")}</div>
            <div class="list-item-meta">
              <span>${escapeHtml(item.book || "")}</span>
              <span>${escapeHtml(item.chapter || "")}</span>
            </div>
          </article>
        `)
        .join("")
    : `<div class="empty-state">Timeline edges пока нет.</div>`;
  qsa(".timeline-edge", byId("timelineEdgesList")).forEach((node) =>
    node.addEventListener("click", () => {
      const payload = JSON.parse(node.dataset.timelineEdge || "{}");
      state.selectedTimelineDetail = payload;
      byId("timelineDetail").value = prettyJson(payload);
    })
  );
}

async function loadLlmExplorer() {
  const runId = byId("llmRunSelect").value;
  const search = byId("llmSearch").value;
  const [runsPayload, tracesPayload] = await Promise.all([
    api("/api/llm/runs"),
    api(`/api/llm/traces?${new URLSearchParams({ run_id: runId, search, limit: "300" }).toString()}`),
  ]);
  state.llmRuns = runsPayload.items || [];
  state.llmTraces = tracesPayload.items || [];
  renderLlmExplorer(runId);
  if (state.selectedLlmTraceRef) {
    await loadLlmTraceDetail(state.selectedLlmTraceRef).catch(() => {});
  }
}

function renderLlmExplorer(currentRunId = "") {
  fillSelect(byId("llmRunSelect"), state.llmRuns.map((item) => item.id), true, currentRunId);
  byId("llmTraceCount").textContent = `${state.llmTraces.length} trace`;
  byId("llmRunsInfo").textContent = state.llmRuns.length
    ? state.llmRuns.map((item) => `${item.id}: ${item.trace_count}`).join(" | ")
    : "Run-ов пока нет.";
  byId("llmTraceList").innerHTML = state.llmTraces
    .map((item) => `
      <article class="list-item ${item.id === state.selectedLlmTraceRef ? "is-selected" : ""}" data-llm-trace="${escapeHtml(item.id)}">
        <div class="list-item-title">${escapeHtml(item.model || item.trace_id)}</div>
        <div>${escapeHtml(item.user_preview || "")}</div>
        <div class="list-item-meta">
          <span>${escapeHtml(item.run_id || "")}</span>
          <span>${escapeHtml(item.provider || "")}</span>
          <span>${escapeHtml(item.log_prefix || "")}</span>
          <span>${escapeHtml(item.last_status || "")}</span>
        </div>
      </article>
    `)
    .join("");
  qsa("[data-llm-trace]", byId("llmTraceList")).forEach((node) =>
    node.addEventListener("click", () => openLlmTrace(node.dataset.llmTrace))
  );
}

async function openLlmTrace(traceRef) {
  state.selectedLlmTraceRef = traceRef;
  renderLlmExplorer(byId("llmRunSelect").value);
  await loadLlmTraceDetail(traceRef);
}

async function loadLlmTraceDetail(traceRef) {
  const payload = await api(`/api/llm/traces/${encodeURIComponent(traceRef)}`);
  const editable = payload.editable_request || {};
  const form = byId("llmForm");
  byId("llmDetailEmpty").classList.add("hidden");
  form.classList.remove("hidden");
  form.elements.namedItem("trace_ref").value = payload.summary?.id || traceRef;
  form.elements.namedItem("model_override").value = editable.model_override || "";
  form.elements.namedItem("max_tokens").value = editable.max_tokens ?? "";
  form.elements.namedItem("temperature").value = editable.temperature ?? "";
  form.elements.namedItem("log_prefix").value = editable.log_prefix || "";
  form.elements.namedItem("response_format").value = editable.response_format ? JSON.stringify(editable.response_format, null, 2) : "";
  form.elements.namedItem("system").value = editable.system || "";
  form.elements.namedItem("user").value = editable.user || "";

  const attempts = Array.isArray(payload.trace?.attempts) ? payload.trace.attempts : [];
  byId("llmAttempts").innerHTML = attempts.length
    ? attempts
        .map((attempt) => `
          <div class="list-item">
            <div><strong>${escapeHtml(attempt.status || "")}</strong> #${escapeHtml(attempt.attempt || "")}</div>
            <div class="muted">${escapeHtml(attempt.ts || "")} · ${escapeHtml(String(attempt.elapsed_seconds ?? ""))}s</div>
            <div>${escapeHtml((attempt.content || attempt.error || "").slice(0, 220))}</div>
          </div>
        `)
        .join("")
    : `<div class="empty-state">Попыток нет.</div>`;
  const lastAttempt = attempts.length ? attempts[attempts.length - 1] : {};
  byId("llmLastResponse").value = lastAttempt.content || JSON.stringify(lastAttempt.raw_response || {}, null, 2);
}

function newLlmRun() {
  state.selectedLlmTraceRef = null;
  const form = byId("llmForm");
  byId("llmDetailEmpty").classList.add("hidden");
  form.classList.remove("hidden");
  form.reset();
  form.elements.namedItem("trace_ref").value = "";
  form.elements.namedItem("max_tokens").value = "800";
  form.elements.namedItem("temperature").value = "0.2";
  form.elements.namedItem("log_prefix").value = "[studio][llm]";
  byId("llmAttempts").innerHTML = `<div class="empty-state">Это новый ручной запуск.</div>`;
  byId("llmLastResponse").value = "";
}

async function runLlmPrompt() {
  const form = byId("llmForm");
  const payload = {
    system: form.elements.namedItem("system").value,
    user: form.elements.namedItem("user").value,
    model_override: form.elements.namedItem("model_override").value,
    max_tokens: form.elements.namedItem("max_tokens").value ? Number(form.elements.namedItem("max_tokens").value) : null,
    temperature: form.elements.namedItem("temperature").value ? Number(form.elements.namedItem("temperature").value) : null,
    log_prefix: form.elements.namedItem("log_prefix").value,
    response_format: form.elements.namedItem("response_format").value,
  };
  const created = await api("/api/llm/run", { method: "POST", body: JSON.stringify(payload) });
  state.selectedLlmTraceRef = created.trace_ref;
  toast(`LLM trace создан: ${created.trace_ref}`);
  await loadSummary();
  await loadLlmExplorer();
  await loadLlmTraceDetail(created.trace_ref);
}

async function rerunLlmTrace() {
  const traceRef = byId("llmForm").elements.namedItem("trace_ref").value;
  if (!traceRef) {
    toast("Сначала выбери trace или создай новый запуск", true);
    return;
  }
  const form = byId("llmForm");
  const payload = {
    system: form.elements.namedItem("system").value,
    user: form.elements.namedItem("user").value,
    model_override: form.elements.namedItem("model_override").value,
    max_tokens: form.elements.namedItem("max_tokens").value ? Number(form.elements.namedItem("max_tokens").value) : null,
    temperature: form.elements.namedItem("temperature").value ? Number(form.elements.namedItem("temperature").value) : null,
    log_prefix: form.elements.namedItem("log_prefix").value,
    response_format: form.elements.namedItem("response_format").value,
  };
  const created = await api(`/api/llm/traces/${encodeURIComponent(traceRef)}/rerun`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
  state.selectedLlmTraceRef = created.trace_ref;
  toast(`Trace перезапущен: ${created.trace_ref}`);
  await loadSummary();
  await loadLlmExplorer();
  await loadLlmTraceDetail(created.trace_ref);
}

async function undoLast() {
  await api("/api/undo", { method: "POST", body: JSON.stringify({}) });
  toast("Последнее изменение отменено");
  await refreshStudio();
}

async function exportFinal() {
  const result = await api("/api/export/final", { method: "POST", body: JSON.stringify({}) });
  toast(`Экспорт сохранён: ${result.export_dir}`);
}

async function bootstrap() {
  await Promise.all([
    loadSummary(),
    loadChunks(),
    loadFacts(),
    loadSamples("voice"),
    loadSamples("synth"),
    loadThemesAndRelations(),
  ]);
}

async function refreshStudio() {
  const tab = currentTabId();
  await bootstrap();
  if (["chunks", "pipeline", "timeline", "llm"].includes(tab)) {
    await ({
      chunks: loadChunks,
      pipeline: loadPipelineExplorer,
      timeline: loadTimelineExplorer,
      llm: loadLlmExplorer,
    }[tab]());
  }
  setTab(tab);
}

function bindEvents() {
  qsa(".tab").forEach((tab) => tab.addEventListener("click", () => activateTab(tab.dataset.tab)));
  byId("refreshBtn").addEventListener("click", () => refreshStudio().catch((err) => toast(err.message, true)));
  byId("undoBtn").addEventListener("click", () => undoLast().catch((err) => toast(err.message, true)));
  byId("exportBtn").addEventListener("click", () => exportFinal().catch((err) => toast(err.message, true)));

  byId("factFilterBtn").addEventListener("click", () => loadFacts().catch((err) => toast(err.message, true)));
  byId("newFactBtn").addEventListener("click", newFact);
  byId("saveFactBtn").addEventListener("click", () => saveFact().catch((err) => toast(err.message, true)));
  byId("deleteFactBtn").addEventListener("click", () => deleteFact().catch((err) => toast(err.message, true)));
  byId("reanalyzeSelectedBtn").addEventListener("click", () => reanalyzeFacts(false).catch((err) => toast(err.message, true)));
  byId("reanalyzeBundleBtn").addEventListener("click", () => reanalyzeFacts(true).catch((err) => toast(err.message, true)));
  byId("generateVoiceBtn").addEventListener("click", () => generateFromFacts("voice").catch((err) => toast(err.message, true)));
  byId("generateSynthBtn").addEventListener("click", () => generateFromFacts("synth").catch((err) => toast(err.message, true)));
  byId("relationCreateForm").addEventListener("submit", (event) => createRelation(event).catch((err) => toast(err.message, true)));

  byId("chunkFilterBtn").addEventListener("click", () => loadChunks().catch((err) => toast(err.message, true)));
  byId("chunkCreateFactBtn").addEventListener("click", createFactFromSelectedChunk);

  byId("voiceFilterBtn").addEventListener("click", () => loadSamples("voice").catch((err) => toast(err.message, true)));
  byId("newVoiceBtn").addEventListener("click", () => newSample("voice"));
  byId("saveVoiceBtn").addEventListener("click", () => saveSample("voice").catch((err) => toast(err.message, true)));
  byId("deleteVoiceBtn").addEventListener("click", () => deleteSample("voice").catch((err) => toast(err.message, true)));

  byId("synthFilterBtn").addEventListener("click", () => loadSamples("synth").catch((err) => toast(err.message, true)));
  byId("newSynthBtn").addEventListener("click", () => newSample("synth"));
  byId("saveSynthBtn").addEventListener("click", () => saveSample("synth").catch((err) => toast(err.message, true)));
  byId("deleteSynthBtn").addEventListener("click", () => deleteSample("synth").catch((err) => toast(err.message, true)));

  byId("newThemeBtn").addEventListener("click", () => {
    state.selectedThemeId = null;
    const form = byId("themeForm");
    byId("themeDetailEmpty").classList.add("hidden");
    form.classList.remove("hidden");
    form.reset();
    form.elements.namedItem("color").value = "#ffb347";
  });
  byId("saveThemeBtn").addEventListener("click", () => saveTheme().catch((err) => toast(err.message, true)));
  byId("deleteThemeBtn").addEventListener("click", () => deleteTheme().catch((err) => toast(err.message, true)));
  byId("themeMergeForm").addEventListener("submit", (event) => mergeThemes(event).catch((err) => toast(err.message, true)));

  byId("pipelineRefreshBtn").addEventListener("click", () => loadPipelineExplorer().catch((err) => toast(err.message, true)));
  byId("timelineFilterBtn").addEventListener("click", () => loadTimelineExplorer().catch((err) => toast(err.message, true)));

  byId("llmFilterBtn").addEventListener("click", () => loadLlmExplorer().catch((err) => toast(err.message, true)));
  byId("llmNewRunBtn").addEventListener("click", newLlmRun);
  byId("llmRunBtn").addEventListener("click", () => runLlmPrompt().catch((err) => toast(err.message, true)));
  byId("llmRerunBtn").addEventListener("click", () => rerunLlmTrace().catch((err) => toast(err.message, true)));
}

document.addEventListener("DOMContentLoaded", async () => {
  bindEvents();
  try {
    await bootstrap();
  } catch (err) {
    toast(err.message || String(err), true);
  }
});
