/**
 * DealBrief AI — app.js
 * Handles all UI interactions and communication with the FastAPI backend.
 */

"use strict";

// ── Helpers ──────────────────────────────────────────────────────────────────

const $ = (id) => document.getElementById(id);

/**
 * Thin fetch wrapper — throws with a readable message on non-2xx responses.
 */
async function api(url, options = {}) {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.detail || data.message || `HTTP ${res.status}`);
  }
  return data;
}

/**
 * Basic HTML escape to prevent XSS in user-supplied text.
 */
function esc(s) {
  return String(s ?? "").replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}

/**
 * Convert markdown-like headings/bold to HTML (minimal subset).
 * Used to render the LLM-generated DealBrief text nicely.
 */
function renderMarkdown(text) {
  return esc(text)
    // ## Heading
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    // **bold**
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    // - list items
    .replace(/^- (.+)$/gm, '<li>$1</li>')
    // Wrap consecutive <li> in <ul>
    .replace(/(<li>[\s\S]+?<\/li>)(?=\s*(?:<li>|<\/ul>|$))/g, (m) => `<ul>${m}</ul>`)
    // Blank lines become paragraph breaks
    .replace(/\n{2,}/g, '</p><p>')
    // Remaining single newlines
    .replace(/\n/g, '<br/>')
    // Wrap in <p>
    .replace(/^(.+)$/, '<p>$1</p>');
}

// ── Health check ─────────────────────────────────────────────────────────────

async function checkHealth() {
  const badge = $("statusBadge");
  const dot   = $("statusDot");
  const text  = $("statusText");
  const foot  = $("footerMode");

  try {
    const h = await api("/api/health");

    const live = h.hindsight_connected && h.groq_configured;
    const mode = live ? "live" : "demo";

    badge.className = `status-badge ${mode}`;

    if (live) {
      text.textContent = `Hindsight ● Groq (${h.groq_model || "model"})`;
      foot.textContent = "LIVE MODE — Hindsight + Groq active";
      foot.className = "footer-mode live-mode";
    } else {
      const parts = [];
      if (!h.hindsight_connected) parts.push("Hindsight not connected");
      if (!h.groq_configured)    parts.push("Groq not configured");
      text.textContent = parts.join(" · ") || "Demo mode";
      foot.textContent = "DEMO MODE — configure .env for live";
      foot.className = "footer-mode demo-mode";
    }

    return h;
  } catch (e) {
    badge.className = "status-badge error";
    text.textContent = "Backend offline";
    foot.textContent = "BACKEND OFFLINE";
    foot.className = "footer-mode";
  }
}

// ── Retain interaction ────────────────────────────────────────────────────────

$("interactionForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  await retainInteraction();
});

async function retainInteraction() {
  const btn = $("retainBtn");
  const msg = $("retainMsg");

  const payload = {
    prospect:      $("prospect").value.trim(),
    company:       $("company").value.trim(),
    meeting_type:  $("meetingType").value,
    date:          $("meetingDate").value,
    summary:       $("summary").value.trim(),
    requirements:  $("requirements").value.trim(),
    objections:    $("objections").value.trim(),
    budget:        $("budget").value.trim(),
    competitors:   $("competitors").value.trim(),
    stakeholders:  $("stakeholders").value.trim(),
    commitments:   $("commitments").value.trim(),
  };

  if (!payload.prospect || !payload.company) {
    showMsg(msg, "error", "⚠ Prospect name and company are required.");
    return;
  }
  if (!payload.summary && !payload.requirements && !payload.objections) {
    showMsg(msg, "error", "⚠ Please fill at least one of: Summary, Requirements, or Objections.");
    return;
  }

  btn.disabled = true;
  btn.innerHTML = '<span class="btn-icon">…</span> Retaining…';
  showMsg(msg, "", "");

  try {
    const data = await api("/api/interactions", {
      method: "POST",
      body: JSON.stringify(payload),
    });

    const sourceLabel = data.source === "hindsight"
      ? "✓ Retained in Hindsight memory"
      : "✓ Stored locally (demo mode)";

    showMsg(msg, "success", `${sourceLabel} — ${payload.company}`);

    // Sync the brief panel's prospect + company to match
    $("briefProspect").value = payload.prospect;
    $("briefCompany").value  = payload.company;

    // Refresh memory panel
    await refreshMemories(payload.prospect, payload.company);
    checkHealth();
  } catch (err) {
    showMsg(msg, "error", `⚠ ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<span class="btn-icon">↗</span> Save to Memory';
  }
}

function showMsg(el, type, text) {
  el.className = `inline-msg ${type}`;
  el.textContent = text;
}

// ── Seed demo data ────────────────────────────────────────────────────────────

$("seedBtn").addEventListener("click", async () => {
  const btn = $("seedBtn");
  btn.disabled = true;
  btn.textContent = "⚡ Seeding…";

  try {
    const data = await api("/api/demo/seed", { method: "POST" });
    const msg = $("retainMsg");
    showMsg(msg, "success", `✓ ${data.message}`);

    // Update prospect fields to match seeded data
    $("prospect").value     = "Rahul Sharma";
    $("company").value      = "Acme Technologies";
    $("briefProspect").value = "Rahul Sharma";
    $("briefCompany").value  = "Acme Technologies";

    // Refresh memory panel
    await refreshMemories("Rahul Sharma", "Acme Technologies");
    checkHealth();
  } catch (err) {
    showMsg($("retainMsg"), "error", `⚠ Seed failed: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = "⚡ Seed Demo Data";
  }
});

// ── Generate DealBrief ────────────────────────────────────────────────────────

$("briefBtn").addEventListener("click", generateBrief);

async function generateBrief() {
  const btn     = $("briefBtn");
  const output  = $("briefOutput");
  const badge   = $("memoryActiveBadge");

  const prospect = $("briefProspect").value.trim();
  const company  = $("briefCompany").value.trim();
  const question = $("question").value.trim() || "What should I know before my next call?";

  if (!prospect) {
    output.className = "brief-output";
    output.innerHTML = '<p style="color:var(--danger);padding:16px;">⚠ Please enter a prospect name.</p>';
    return;
  }

  btn.disabled = true;
  btn.innerHTML = '<span class="btn-icon">…</span> Recalling memory…';

  output.className = "brief-output brief-output--loading";
  output.innerHTML = `
    <div class="loading-spinner"></div>
    <div class="loading-text">Querying Hindsight memory…</div>
  `;
  badge.classList.add("hidden");

  try {
    const data = await api("/api/brief", {
      method: "POST",
      body: JSON.stringify({ prospect, company, question }),
    });

    // Show memory panel
    renderMemories(data.memories, data.memory_source);

    // Show MEMORY ACTIVE badge if real memories were found
    if (data.memories && data.memories.length > 0) {
      badge.classList.remove("hidden");
    }

    // Update source tag
    updateSourceTag(data.memory_source);

    // Render brief
    output.className = "brief-output brief-output--loaded";
    output.innerHTML = buildBriefHTML(data);

  } catch (err) {
    output.className = "brief-output";
    output.innerHTML = `<p style="color:var(--danger);padding:16px;">⚠ ${esc(err.message)}</p>`;
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<span class="btn-icon">✦</span> Generate DealBrief';
  }
}

function buildBriefHTML(data) {
  const memSrc = data.memory_source || "demo";
  const llmSrc = data.llm_source    || "demo";
  const count  = data.memory_count  || 0;

  const memTag = memSrc === "hindsight"
    ? '<span class="brief-tag hindsight">✦ HINDSIGHT MEMORY</span>'
    : '<span class="brief-tag demo-tag">◯ DEMO MEMORY</span>';

  const llmTag = llmSrc === "groq"
    ? `<span class="brief-tag groq-tag">◈ GROQ ${esc(data.llm_model || "")}</span>`
    : '<span class="brief-tag demo-tag">◈ DEMO BRIEF</span>';

  const countTag = `<span class="brief-tag memory-count">${count} memor${count === 1 ? "y" : "ies"} recalled</span>`;

  return `
    <div class="brief-meta">
      ${memTag}
      ${llmTag}
      ${countTag}
    </div>
    <div class="brief-content">
      ${renderMarkdown(data.brief || data.answer || "")}
    </div>
  `;
}

// ── Recall memory ─────────────────────────────────────────────────────────────

$("refreshMemBtn").addEventListener("click", () => {
  const prospect = $("briefProspect").value.trim() || $("prospect").value.trim();
  const company  = $("briefCompany").value.trim()  || $("company").value.trim();
  refreshMemories(prospect, company);
});

async function refreshMemories(prospect, company) {
  const grid = $("memoriesGrid");

  grid.innerHTML = `
    <div class="empty-memories">
      <div class="loading-spinner" style="margin:0 auto 10px;"></div>
      <p>Querying Hindsight…</p>
    </div>
  `;

  try {
    const companyParam = company ? `?company=${encodeURIComponent(company)}` : "";
    const data = await api(`/api/memories/${encodeURIComponent(prospect)}${companyParam}`);

    renderMemories(data.memories, data.source);
    updateSourceTag(data.source);
  } catch (err) {
    grid.innerHTML = `
      <div class="empty-memories">
        <p style="color:var(--danger);">⚠ ${esc(err.message)}</p>
      </div>
    `;
  }
}

function renderMemories(items, source) {
  const grid  = $("memoriesGrid");
  const badge = $("memoryActiveBadge");

  if (!items || items.length === 0) {
    grid.innerHTML = `
      <div class="empty-memories">
        <div class="empty-icon small">◎</div>
        <p>No memories found for this prospect.</p>
        <p class="empty-sub">Log an interaction above, then click Recall Memory.</p>
      </div>
    `;
    badge.classList.add("hidden");
    return;
  }

  if (items.length > 0) badge.classList.remove("hidden");
  updateSourceTag(source);

  grid.innerHTML = items.map((m, i) => buildMemoryCard(m, i)).join("");
}

function buildMemoryCard(m, index) {
  const typeLabel = esc(m.type || "memory").toUpperCase();
  const scoreHTML = m.score != null
    ? `<span class="memory-score">score: ${m.score}</span>`
    : "";

  const contextHTML = m.context
    ? `<div class="memory-context">${esc(m.context)}</div>`
    : "";

  const tagsHTML = m.tags && m.tags.length
    ? `<div class="memory-tags">${m.tags.map(t => `<span class="memory-tag-chip">${esc(t)}</span>`).join("")}</div>`
    : "";

  return `
    <article class="memory-card-item">
      <div class="memory-card-header">
        <div class="memory-index">${String(index + 1).padStart(2, "0")}</div>
        <span class="memory-type-badge">${typeLabel}</span>
        ${scoreHTML}
      </div>
      <p class="memory-text">${esc(m.text)}</p>
      ${tagsHTML}
      ${contextHTML}
    </article>
  `;
}

function updateSourceTag(source) {
  const tag  = $("memorySourceTag");
  const span = $("memorySourceText");
  tag.classList.remove("hidden");
  span.textContent = source === "hindsight" ? "HINDSIGHT" : "DEMO";
}

// ── Sync prospect across panels ───────────────────────────────────────────────

// When the sidebar prospect name changes, mirror it to the brief panel
$("prospect").addEventListener("change", () => {
  $("briefProspect").value = $("prospect").value;
});
$("company").addEventListener("change", () => {
  $("briefCompany").value = $("company").value;
});

// ── Init ──────────────────────────────────────────────────────────────────────

// Set today's date as default in the date picker
(function setDefaultDate() {
  const today = new Date().toISOString().split("T")[0];
  $("meetingDate").value = today;
})();

// Initial health check
checkHealth();
