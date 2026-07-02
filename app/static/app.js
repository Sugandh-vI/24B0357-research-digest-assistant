/**
 * app.js — Research Digest AI frontend logic
 * Handles: digest form, result rendering, collapsible sections, chat Q&A
 */
(function () {
  "use strict";

  /* ── DOM refs ── */
  const searchForm    = document.getElementById("searchForm");
  const topicInput    = document.getElementById("topicInput");
  const searchBtn     = document.getElementById("searchBtn");
  const searchError   = document.getElementById("searchError");
  const progressArea  = document.getElementById("progressArea");
  const progressBar   = document.getElementById("progressBar");
  const progressLabel = document.getElementById("progressLabel");
  const digestSection = document.getElementById("digestSection");
  const digestHeading = document.getElementById("digestHeading");
  const digestList    = document.getElementById("digestList");
  const paperCount    = document.getElementById("paperCount");
  const chatSection   = document.getElementById("chatSection");
  const chatForm      = document.getElementById("chatForm");
  const chatInput     = document.getElementById("chatInput");
  const chatBtn       = document.getElementById("chatBtn");
  const chatError     = document.getElementById("chatError");
  const chatMessages  = document.getElementById("chatMessages");

  const stepSearch    = document.getElementById("step-search");
  const stepSummarize = document.getElementById("step-summarize");
  const stepCritique  = document.getElementById("step-critique");
  const stepRank      = document.getElementById("step-rank");

  /* ── Helpers ── */
  function showError(el, msg) {
    el.textContent = msg;
    el.hidden = false;
  }
  function hideError(el) { el.hidden = true; }

  function setStep(active) {
    const steps = { search: stepSearch, summarize: stepSummarize, critique: stepCritique, rank: stepRank };
    const order = ["search", "summarize", "critique", "rank"];
    const activeIdx = order.indexOf(active);
    order.forEach((name, i) => {
      const el = steps[name];
      el.classList.remove("active", "done");
      if (i < activeIdx) el.classList.add("done");
      else if (i === activeIdx) el.classList.add("active");
    });
  }

  function animateProgress(targetPct, label, step) {
    progressBar.style.width = targetPct + "%";
    progressLabel.textContent = label;
    if (step) setStep(step);
  }

  function escHtml(str) {
    return String(str || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function scoreClass(score) {
    const s = parseInt(score, 10);
    if (s >= 5) return "rank-5";
    if (s >= 4) return "rank-4";
    if (s >= 3) return "rank-3";
    if (s >= 2) return "rank-2";
    return "rank-1";
  }

  /* ── Render paper card ── */
  function renderPaper(paper, idx) {
    const critique   = paper.critique || {};
    const score      = critique.relevance_score || "?";
    const authors    = (paper.authors || []).slice(0, 3).join(", ") + (paper.authors?.length > 3 ? " et al." : "");
    const cardId     = `paper-${idx}`;

    const card = document.createElement("article");
    card.className = "paper-card";
    card.setAttribute("role", "listitem");
    card.style.animationDelay = `${idx * 60}ms`;

    card.innerHTML = `
      <div class="rank-badge ${scoreClass(score)}" title="Relevance score ${score}/5">${score}/5</div>
      <h3 class="paper-title">
        <a href="${escHtml(paper.link)}" target="_blank" rel="noopener noreferrer"
           id="${cardId}-title">${escHtml(paper.title)}</a>
      </h3>
      <div class="paper-meta">
        <span class="authors">${escHtml(authors)}</span>
        <span class="separator">·</span>
        <span class="date">${escHtml(paper.published)}</span>
      </div>
      <div class="paper-sections">
        <!-- Summary -->
        <div class="paper-section">
          <button class="section-toggle" aria-expanded="true"
                  aria-controls="${cardId}-summary" id="${cardId}-summary-btn">
            📝 Plain-English Summary <span class="chevron">▼</span>
          </button>
          <div class="section-body open" id="${cardId}-summary" role="region">
            ${escHtml(paper.summary || "No summary available.")}
          </div>
        </div>
        <!-- Critique -->
        <div class="paper-section">
          <button class="section-toggle" aria-expanded="false"
                  aria-controls="${cardId}-critique" id="${cardId}-critique-btn">
            🔬 Peer Review Critique <span class="chevron">▼</span>
          </button>
          <div class="section-body" id="${cardId}-critique" role="region">
            <div class="critique-grid">
              <span class="critique-chip chip-score">Relevance: ${escHtml(String(score))}/5</span>
              ${critique.target_reader ? `<span class="critique-chip chip-reader">👤 ${escHtml(critique.target_reader)}</span>` : ""}
            </div>
            <strong style="font-size:0.82rem;color:var(--color-text-muted);">Limitations:</strong><br/>
            ${escHtml(critique.limitations || "Not available.")}
          </div>
        </div>
      </div>
    `;

    // Collapsible toggle logic
    card.querySelectorAll(".section-toggle").forEach((btn) => {
      btn.addEventListener("click", () => {
        const expanded = btn.getAttribute("aria-expanded") === "true";
        btn.setAttribute("aria-expanded", String(!expanded));
        const body = document.getElementById(btn.getAttribute("aria-controls"));
        body.classList.toggle("open", !expanded);
      });
    });

    return card;
  }

  /* ── Digest form submit ── */
  searchForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    hideError(searchError);

    const topic = topicInput.value.trim();
    if (!topic) { showError(searchError, "Please enter a research topic."); return; }

    // UI: loading state
    searchBtn.disabled = true;
    searchBtn.querySelector(".btn-text").hidden = true;
    searchBtn.querySelector(".btn-spinner").hidden = false;
    progressArea.hidden = false;
    digestSection.hidden = true;
    chatSection.hidden = true;
    digestList.innerHTML = "";

    // Animated progress stages
    animateProgress(10, "🔍 Searching arXiv for papers…", "search");

    const progressTimer = [
      setTimeout(() => animateProgress(35, "📝 Summarizing papers with LLM…", "summarize"), 3000),
      setTimeout(() => animateProgress(65, "🔬 Critiquing papers…", "critique"), 8000),
      setTimeout(() => animateProgress(88, "📊 Ranking by relevance…", "rank"), 16000),
    ];

    try {
      const res = await fetch("/api/digest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ topic }),
      });

      progressTimer.forEach(clearTimeout);
      animateProgress(100, "✅ Digest ready!", "rank");

      const data = await res.json();

      if (!res.ok) {
        showError(searchError, data.detail || "Server error. Please try again.");
        return;
      }

      if (data.status === "no_results" || !data.papers?.length) {
        showError(searchError, data.message || "No papers found. Try a different topic.");
        return;
      }

      // Render digest
      digestSection.hidden = false;
      digestHeading.textContent = `📄 Digest: "${topic}"`;
      paperCount.textContent = `${data.papers.length} paper${data.papers.length !== 1 ? "s" : ""}`;

      data.papers.forEach((paper, idx) => {
        digestList.appendChild(renderPaper(paper, idx));
      });

      // Scroll to digest
      digestSection.scrollIntoView({ behavior: "smooth", block: "start" });

      // Enable chat
      chatSection.hidden = false;
      chatMessages.innerHTML = "";
      addChatBubble("assistant", "I've read all the papers above. What would you like to know?", []);

    } catch (err) {
      progressTimer.forEach(clearTimeout);
      showError(searchError, "Network error — is the server running? " + err.message);
    } finally {
      searchBtn.disabled = false;
      searchBtn.querySelector(".btn-text").hidden = false;
      searchBtn.querySelector(".btn-spinner").hidden = true;
      setTimeout(() => { progressArea.hidden = true; }, 1500);
    }
  });

  /* ── Chat / Q&A ── */
  function addChatBubble(role, text, citations) {
    const wrap = document.createElement("div");
    wrap.className = `chat-bubble ${role}`;

    const label = document.createElement("span");
    label.className = "bubble-label";
    label.textContent = role === "user" ? "You" : "Research AI";

    const bubble = document.createElement("div");
    bubble.className = "bubble-text";
    bubble.textContent = text;

    if (citations && citations.length > 0) {
      const citLabel = document.createElement("p");
      citLabel.style.cssText = "font-size:0.75rem;color:var(--color-text-subtle);margin-top:0.5rem;";
      citLabel.textContent = "Sources:";
      bubble.appendChild(citLabel);

      const ul = document.createElement("ul");
      ul.className = "citations-list";
      citations.forEach((c) => {
        const li = document.createElement("li");
        li.innerHTML = `📎 <a href="${escHtml(c.link)}" target="_blank" rel="noopener">${escHtml(c.title)}</a>`;
        ul.appendChild(li);
      });
      bubble.appendChild(ul);
    }

    wrap.append(label, bubble);
    chatMessages.appendChild(wrap);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return wrap;
  }

  function addThinkingBubble() {
    const wrap = document.createElement("div");
    wrap.className = "chat-bubble assistant";
    const thinking = document.createElement("div");
    thinking.className = "chat-thinking";
    thinking.innerHTML = "<span></span><span></span><span></span>";
    wrap.appendChild(thinking);
    chatMessages.appendChild(wrap);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return wrap;
  }

  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    hideError(chatError);

    const question = chatInput.value.trim();
    if (!question) return;

    addChatBubble("user", question, []);
    chatInput.value = "";
    chatBtn.disabled = true;
    chatBtn.querySelector(".btn-text").hidden = true;
    chatBtn.querySelector(".btn-spinner").hidden = false;

    const thinkingEl = addThinkingBubble();

    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const data = await res.json();
      thinkingEl.remove();

      if (!res.ok) {
        addChatBubble("assistant", "⚠️ " + (data.detail || "Something went wrong."), []);
        return;
      }

      addChatBubble("assistant", data.answer || "No answer returned.", data.citations || []);

    } catch (err) {
      thinkingEl.remove();
      showError(chatError, "Network error: " + err.message);
    } finally {
      chatBtn.disabled = false;
      chatBtn.querySelector(".btn-text").hidden = false;
      chatBtn.querySelector(".btn-spinner").hidden = true;
    }
  });

})();
