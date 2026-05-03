# agents_drug.py
# Drugs@FDA dashboard chart AI: aggregated CONTEXT → LLM (Ollama or OpenAI), plus an optional
# TOOL2-style **multi-agent orchestrator** (Ollama-only) with explicit **tool calling** + **RAG**.
#
# Architecture (default path — Template 3):
#   Step A — aggregate_full_dashboard_context(df)  … same filtered AP rows as the three charts
#   Step B — summarize_dashboard_charts(context)   … one “analyst” LLM call (no tools in this path)
#
# Architecture (advanced — compare to pokeapp/agents_pokemon.py terminal_demo pattern):
#   Agent 1 — Planner: JSON plan (what to query / what RAG query to use)
#   Agent 2 — Data: Ollama tool calls → dashboard_df_metrics, openFDA drugsfda search, query URL
#   Agent 3 — RAG: Ollama tool call → retrieve_dashboard_notes (bundled markdown corpus)
#   Agent 4 — Insight: final stakeholder explanation (no tools)
#   Agent 5 — Validator (mandatory when Insight is non-empty): QC edit pass — verification, contradiction checks, no new numbers
#
# When DASHBOARD_AI_ORCHESTRATOR=1, each agent’s text is **printed** to stdout (server logs);
# the **return value** for Shiny is the final narrative plus a **Quality control** Markdown footer
# (pipeline description, Validator rules when enabled, and lightweight automated checks).
# To see the same agent trace in your laptop terminal without Shiny:  python agents_drug.py run
#
# Env: AI_BACKEND, OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_API_KEY, OPENFDA_API_KEY,
#      DASHBOARD_AI_ORCHESTRATOR
#      DASHBOARD_AI_ORCH_PARALLEL (default 1) — run Agent 2 Data + Agent 3 RAG Ollama rounds in parallel
# RAG corpus: rag/drugsfda_dashboard_notes.md
#
# pip install pandas requests python-dotenv  (env_load uses dotenv)

from __future__ import annotations

# 0. SETUP ######################################################################

## 0.1 Load packages #############################################################

import json
import os
import re
import sys
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import env_load  # noqa: F401 — loads `.env` / `.env.txt` from app directory (like dotenv in labs)
import pandas as pd
import requests

## 0.2 Optional: load functions #################################################
# Lab scripts use:  from functions import agent_run, df_as_text
# This app inlines chat HTTP + aggregation so Shiny has no extra functions.py dependency.


# 1. CONFIGURATION ##############################################################

# Defaults match openFDA / Ollama common tags; override via environment.
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OLLAMA_MODEL = "llama3.2:latest"
OLLAMA_CHAT_PATH = "/api/chat"
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"

# AI_BACKEND=ollama | openai  (see README)


# 2. PROMPT TEMPLATE — Template 3 (Chart explanation) ###########################

# Filled with aggregate_full_dashboard_context(...) for CONTEXT.
CHART_EXPLANATION_PROMPT = """ROLE:
You are a data analyst explaining trends to a business audience.

TASK:
Interpret ONLY the CONTEXT block below (the filtered sample shown on the dashboard). Do not use external FDA knowledge, industry rumors, or facts not present in CONTEXT. If CONTEXT is silent on a point, say so briefly instead of inferring.

CONTEXT:
{aggregated_chart_data}

REQUIRED ANALYSIS (cover each; use only numbers and labels appearing in CONTEXT):

(1) Approval type mix — Compare NDA vs ANDA vs BLA proportions. Identify which category dominates and by how much: state each major category’s share (%) of filtered AP events (or equivalent counts from CONTEXT).

(2) Time series — Identify the peak year and the approval count in that year. Compute or compare against the average annual approval count across the years in CONTEXT; quantify how far the peak is above or below that average. State whether the series is overall increasing, decreasing, or roughly stable across the year range (justify using year-by-year counts in CONTEXT).

(3) Sponsors — Name the top three sponsors as ranked in CONTEXT and give each sponsor’s share (%) of total approvals under the metric described in CONTEXT (events or distinct apps). Compare first place vs second place using counts and/or percentage-point gaps from CONTEXT only.

(4) Numeric discipline — Always include concrete numbers (counts, years, percentages, differences) when describing magnitude. Do not rely on vague qualifiers (“high,” “low,” “large,” “small”) without the supporting figure from CONTEXT.

OUTPUT FORMAT:
• 3–6 concise sentences (or short bullet lines) that satisfy (1)–(4).
• Emphasize what changed in this dataset, why it matters for this filtered window, and magnitudes backed solely by CONTEXT.
• Use Markdown in your answer (e.g. **bold** for key numbers, `-` bullet lists) so the dashboard can render it clearly.
"""


# 3. SMALL HELPERS — labels & backend ###########################################

# Match app_drug.py pie labels (NDA / ANDA / BLA / Other).
_KIND_LABELS = {
    "NDA": "NDA (brand)",
    "ANDA": "ANDA (generic)",
    "BLA": "BLA (biologic)",
    "Other": "Other / unknown",
}


def _sponsor_key(name: str, normalize: bool) -> str:
    """Same grouping key as dashboard Top sponsors chart."""
    raw = " ".join(str(name or "").split()).strip() or "Unknown"
    return raw.upper() if normalize else raw


def _llm_backend() -> str:
    """Return 'openai' or 'ollama' (default ollama)."""
    return (os.getenv("AI_BACKEND") or "ollama").strip().lower()


# 4. DATA AGGREGATION — CONTEXT string for the LLM ###############################
# Analogous to df_as_text(...) in the lab: turn dashboard-aligned data into prompt-sized text.


def aggregate_full_dashboard_context(
    df: pd.DataFrame,
    *,
    sponsor_normalize: bool,
    sponsor_metric: str,
    top_n_sponsors: int,
    show_rolling_avg: bool,
    roll_window: int,
) -> str:
    """
    Build CONTEXT aligned with the three dashboard visuals:
    Approval type pie, Approvals per year, Top sponsors (same filters & aggregations as app_drug.py).

    Parameters
    ----------
    df : DataFrame
        filtered_approved_for_charts() — AP rows with year + application_kind + sponsor_name.
    sponsor_normalize, sponsor_metric, top_n_sponsors
        Mirror sidebar inputs for Top sponsors chart.
    show_rolling_avg, roll_window
        Mirror trend chart rolling average options.

    Returns
    -------
    str
        Multi-section text passed into CHART_EXPLANATION_PROMPT as {aggregated_chart_data}.
    """
    if df is None or df.empty:
        return (
            "No rows: data may not be loaded, or the current year range / application-type filter "
            "produced no AP approval events."
        )
    lines: list[str] = []
    lines.append("=== Foundation (same filtered AP rows as all charts) ===")
    lines.append(f"Total AP approval events: {len(df)}")
    if "application_number" in df.columns:
        lines.append(f"Distinct applications: {df['application_number'].nunique()}")
    if "year" in df.columns:
        lines.append(f"Year range: {int(df['year'].min())}–{int(df['year'].max())}")

    lines.append("")
    lines.append("=== 🧾 Approval type — NDA vs ANDA vs BLA (pie chart) ===")
    if "application_kind" in df.columns:
        vc = df["application_kind"].value_counts()
        total_ev = len(df)
        for k in vc.index:
            lab = _KIND_LABELS.get(str(k), str(k))
            c = int(vc[k])
            pct = 100.0 * c / total_ev if total_ev else 0.0
            lines.append(f"  {lab}: {c} events ({pct:.1f}% of filtered AP events)")
    else:
        lines.append("  (application_kind not available)")

    lines.append("")
    lines.append("=== 📈 Approvals per year (trend chart) ===")
    if "year" in df.columns:
        yearly = df.groupby("year", as_index=False).size().rename(columns={"size": "n"}).sort_values("year")
        for _, row in yearly.iterrows():
            lines.append(f"  Year {int(row['year'])}: {int(row['n'])} AP approvals")
        if not yearly.empty:
            imax = yearly["n"].idxmax()
            peak_y = int(yearly.loc[imax, "year"])
            peak_n = int(yearly.loc[imax, "n"])
            lines.append(f"  Peak year: {peak_y} with {peak_n} approvals (annotated on chart)")
            mean_n = float(yearly["n"].mean())
            lines.append(f"  Mean annual approvals (average of per-year counts in this window): {mean_n:.2f}")
            lines.append(
                f"  Peak vs mean: {peak_n} vs {mean_n:.2f} "
                f"(difference {float(peak_n) - mean_n:+.1f} approvals vs the annual average)"
            )
            if len(yearly) >= 2:
                mid = max(1, len(yearly) // 2)
                first_m = float(yearly.iloc[:mid]["n"].mean())
                second_m = float(yearly.iloc[mid:]["n"].mean())
                lines.append(
                    f"  Trend hint (split window): earlier years mean={first_m:.2f} approvals/year; "
                    f"later years mean={second_m:.2f} approvals/year"
                )
        if show_rolling_avg and len(yearly) >= 1:
            w = max(1, int(roll_window))
            roll = yearly["n"].rolling(window=w, min_periods=1).mean()
            lines.append(f"  {w}-year rolling average (matches dashed line if enabled):")
            for i, (_, row) in enumerate(yearly.iterrows()):
                lines.append(f"    {int(row['year'])}: {float(roll.iloc[i]):.2f}")
    else:
        lines.append("  (year not available)")

    lines.append("")
    lines.append("=== 🏢 Top sponsors (horizontal bar chart) ===")
    lines.append(
        f"  Metric: {'each AP event' if sponsor_metric == 'events' else 'distinct applications per sponsor'}; "
        f"grouping: {'normalized (trim + uppercase)' if sponsor_normalize else 'raw sponsor name'}; "
        f"top {top_n_sponsors} shown on chart."
    )
    if "sponsor_name" not in df.columns:
        lines.append("  (sponsor_name not available)")
    else:
        dfx = df.copy()
        dfx["_skey"] = dfx["sponsor_name"].map(lambda x: _sponsor_key(x, sponsor_normalize))
        first_label = dfx.groupby("_skey", as_index=False)["sponsor_name"].first()
        if sponsor_metric == "distinct_apps":
            sub = dfx.drop_duplicates(["_skey", "application_number"])
            counts = sub.groupby("_skey").size()
        else:
            counts = dfx.groupby("_skey").size()
        total = float(counts.sum()) if len(counts) else 0.0
        top = counts.sort_values(ascending=False).head(int(top_n_sponsors))
        rank = 0
        for skey in top.index:
            rank += 1
            c = int(top[skey])
            lab = first_label.loc[first_label["_skey"] == skey, "sponsor_name"].iloc[0]
            pct = 100.0 * float(c) / total if total else 0.0
            lines.append(f"  #{rank} {str(lab)[:70]}: count={c} ({pct:.1f}% of metric total)")
        if len(top) >= 2:
            k0, k1 = top.index[0], top.index[1]
            c0, c1 = int(top[k0]), int(top[k1])
            p0 = 100.0 * float(c0) / total if total else 0.0
            p1 = 100.0 * float(c1) / total if total else 0.0
            lab0 = first_label.loc[first_label["_skey"] == k0, "sponsor_name"].iloc[0]
            lab1 = first_label.loc[first_label["_skey"] == k1, "sponsor_name"].iloc[0]
            lines.append("")
            lines.append(
                f"  Top sponsor vs 2nd: '{str(lab0)[:50]}' count={c0} ({p0:.1f}%) vs "
                f"'{str(lab1)[:50]}' count={c1} ({p1:.1f}%) — "
                f"count gap {c0 - c1:+d}, share gap {p0 - p1:+.1f} percentage points"
            )
        lines.append("")
        lines.append("  Top 3 sponsors only (use these names and shares for narrative):")
        for i, skey in enumerate(list(top.index)[:3], start=1):
            c = int(top[skey])
            lab = first_label.loc[first_label["_skey"] == skey, "sponsor_name"].iloc[0]
            pct = 100.0 * float(c) / total if total else 0.0
            lines.append(f"    {i}. {str(lab)[:70]} — count {c}, {pct:.1f}% of metric total")

    return "\n".join(lines)


def aggregate_dashboard_charts_context(df: pd.DataFrame) -> str:
    """Backward-compatible wrapper with fixed defaults (legacy callers / tests)."""
    return aggregate_full_dashboard_context(
        df,
        sponsor_normalize=False,
        sponsor_metric="events",
        top_n_sponsors=10,
        show_rolling_avg=False,
        roll_window=3,
    )


# 5. BUILD USER PROMPT ##########################################################


def build_chart_explanation_prompt(aggregated_chart_data: str) -> str:
    """Inject aggregated CONTEXT into CHART_EXPLANATION_PROMPT."""
    return CHART_EXPLANATION_PROMPT.format(
        aggregated_chart_data=(aggregated_chart_data.strip() or "(no aggregated data)")
    )


# 6. LLM CLIENTS — OpenAI Chat Completions vs Ollama /api/chat ##################
# Lab `agent()` posts to Ollama; we mirror both backends for deployment flexibility.


def _openai_chat(
    messages: list[dict[str, Any]],
    *,
    temperature: float = 0.35,
    timeout: float = 90.0,
) -> dict[str, Any]:
    """POST https://api.openai.com/v1/chat/completions — OpenAI-shaped JSON response."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY is not set.")
    model = os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    r = requests.post(OPENAI_CHAT_URL, headers=headers, json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _ollama_chat(
    messages: list[dict[str, Any]],
    *,
    temperature: float = 0.35,
    timeout: float = 90.0,
) -> dict[str, Any]:
    """POST {OLLAMA_HOST}/api/chat — normalize to {'choices': [{'message': ...}]} for one code path."""
    host = (os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL
    url = f"{host}{OLLAMA_CHAT_PATH}"
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 896},
    }
    headers = {}
    # Ollama Cloud (https://ollama.com) requires an API key for programmatic access.
    api_key = (os.getenv("OLLAMA_API_KEY") or "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    r = requests.post(url, json=body, headers=headers or None, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    # Some Ollama Cloud / proxy setups may return an OpenAI-shaped payload already.
    if isinstance(data, dict) and isinstance(data.get("choices"), list):
        return data

    # Ollama native shape: {"message": {"role": "...", "content": "..."}, ...}
    if isinstance(data, dict) and isinstance(data.get("message"), dict):
        msg = data.get("message") or {}
        # If server returned an auth-style payload but with 200, surface it.
        if not (msg.get("content") or "").strip() and isinstance(data.get("error"), str):
            signin = data.get("signin_url")
            extra = f"\nSign in: {signin}" if isinstance(signin, str) and signin else ""
            raise ValueError(f"Ollama error: {data.get('error')}{extra}")
        return {"choices": [{"message": msg}]}

    # Unknown shape — return a truncated diagnostic so the UI isn't blank.
    snippet = str(data)[:1200]
    raise ValueError(
        "Unexpected Ollama /api/chat response shape (no 'message' or 'choices'). "
        f"Host={host} Model={model}\nResponse: {snippet}"
    )


def _ollama_generate(
    prompt: str,
    *,
    temperature: float = 0.35,
    timeout: float = 90.0,
) -> str:
    """
    POST {OLLAMA_HOST}/api/generate and return the 'response' string.
    This is a robust fallback when /api/chat returns empty 'content' (e.g. thinking-only payloads).
    """
    host = (os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL
    url = f"{host}/api/generate"
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    headers = {}
    api_key = (os.getenv("OLLAMA_API_KEY") or "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    r = requests.post(url, json=body, headers=headers or None, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return str(data.get("response") or "").strip()


def _llm_chat(
    messages: list[dict[str, Any]],
    *,
    temperature: float = 0.35,
    timeout: float = 90.0,
) -> dict[str, Any]:
    """Route to OpenAI or Ollama based on AI_BACKEND and credentials."""
    if _llm_backend() == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("AI_BACKEND=openai but OPENAI_API_KEY is not set.")
        return _openai_chat(messages, temperature=temperature, timeout=timeout)
    return _ollama_chat(messages, temperature=temperature, timeout=timeout)


def _message_text(msg: dict[str, Any]) -> str:
    """Extract assistant string from either API shape."""
    c = msg.get("content")
    if isinstance(c, str):
        return (c or "").strip()
    # Some providers may return a list of content parts; join any string parts.
    if isinstance(c, list):
        parts: list[str] = []
        for p in c:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and isinstance(p.get("text"), str):
                parts.append(p["text"])
        return "\n".join([s.strip() for s in parts if s and s.strip()]).strip()
    return ""


# -----------------------------------------------------------------------------
# 7. DASHBOARD ORCHESTRATOR — Planner → Data (tools) → RAG (tools) → Insight → Validator
# -----------------------------------------------------------------------------
# Enable: DASHBOARD_AI_ORCHESTRATOR=1   (Ollama-only; requires tool-capable model)
# Quality control: Validator always runs after a non-empty Insight (not env-gated).
# Each agent’s output is printed to stdout; summarize_dashboard_charts_tool_rag returns final text + QC footer.


def _orch_truthy(val: str | None) -> bool:
    if val is None:
        return False
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def dashboard_orchestrator_enabled() -> bool:
    return _orch_truthy(os.getenv("DASHBOARD_AI_ORCHESTRATOR"))


def _orch_parallel_data_rag_enabled() -> bool:
    """Run Data + RAG tool rounds concurrently (saves wall time when Ollama allows parallel /api/chat)."""
    v = (os.getenv("DASHBOARD_AI_ORCH_PARALLEL") or "1").strip().lower()
    return v not in {"0", "false", "no", "off"}


def _orch_hr(char: str = "=", n: int = 79) -> str:
    return char * n


def _orch_tool_out_ok(out: Any) -> bool:
    """True if dispatch returned a normal payload (dict `error` key means tool-level failure)."""
    if isinstance(out, dict) and out.get("error"):
        return False
    return True


def _orch_tool_transcript_lines(transcript: list[dict[str, Any]], *, max_items: int = 12) -> str:
    if not transcript:
        return "(no tool calls recorded)"
    n = len(transcript)
    n_ok = sum(1 for row in transcript if row.get("ok", True))
    n_fail = n - n_ok
    if n_fail == 0:
        header = f"Tool round: {n} tool call(s); all completed successfully."
    else:
        header = (
            f"Tool round: {n} tool call(s); {n_ok} successful, {n_fail} reported an error "
            "(see [failed] lines and previews below)."
        )
    lines: list[str] = [header, ""]
    for i, row in enumerate(transcript[:max_items], start=1):
        tool = row.get("tool")
        args = row.get("args")
        prev = row.get("output_preview")
        ok = row.get("ok", True)
        tag = "[success]" if ok else "[failed]"
        inv = str(row.get("invoked_by") or "model")
        via = f"  (via {inv})" if inv != "model" else ""
        lines.append(f"{i}) {tool}({args})  {tag}{via}")
        if prev:
            lines.append(f"   preview: {prev}")
    if len(transcript) > max_items:
        lines.append(f"... ({len(transcript) - max_items} more tool events omitted)")
    return "\n".join(lines)


def _orch_ollama_base() -> str:
    return (os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")


def _orch_ollama_model() -> str:
    return (os.getenv("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL).strip()


def _orch_ollama_headers() -> dict[str, str]:
    key = (os.getenv("OLLAMA_API_KEY") or "").strip()
    if not key:
        return {}
    return {"Authorization": f"Bearer {key}"}


def _orch_require_ollama_backend() -> None:
    backend = (os.getenv("AI_BACKEND") or "ollama").strip().lower()
    if backend not in {"", "ollama"}:
        raise ValueError(
            f"Dashboard orchestrator is Ollama-only, but AI_BACKEND={backend!r}. "
            "Set AI_BACKEND=ollama (or unset) to use this path."
        )


def _orch_native_chat(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    format_json: bool = False,
    temperature: float = 0.2,
    num_predict: int = 900,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """POST Ollama /api/chat; returns native JSON (supports tool_calls)."""
    body: dict[str, Any] = {
        "model": _orch_ollama_model(),
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    if tools:
        body["tools"] = tools
    if format_json:
        body["format"] = "json"

    url = f"{_orch_ollama_base()}/api/chat"
    r = requests.post(url, json=body, headers=_orch_ollama_headers(), timeout=timeout)
    r.raise_for_status()
    return r.json()


def _orch_msg_text(data: dict[str, Any]) -> str:
    msg = data.get("message") or {}
    return str(msg.get("content") or "").strip()


def _orch_chat_plain_text(
    messages: list[dict[str, Any]],
    *,
    temperature: float = 0.2,
    num_predict: int = 900,
    timeout: float = 120.0,
) -> str:
    """
    No-tool assistant completion: POST /api/chat, then read text via _message_text (handles
    multipart content). If still empty — common with Ollama Cloud “thinking” payloads where
    `content` is blank — fall back to /api/generate with the same system+user text inlined
    (same pattern as summarize_dashboard_charts for the single-LLM path).
    """
    data = _orch_native_chat(
        messages,
        tools=None,
        format_json=False,
        temperature=temperature,
        num_predict=num_predict,
        timeout=timeout,
    )
    msg = data.get("message") or {}
    txt = _message_text(msg).strip()
    if txt:
        return txt
    lines: list[str] = []
    for m in messages:
        role = str(m.get("role") or "").strip()
        c = m.get("content")
        if isinstance(c, str) and c.strip():
            lines.append(f"{role.upper()}:\n{c.strip()}")
    combined = "\n\n".join(lines)
    if not combined.strip():
        return ""
    print(
        "  (note: /api/chat returned empty assistant text for this agent; retrying via /api/generate)",
        flush=True,
    )
    return _ollama_generate(combined, temperature=temperature, timeout=timeout).strip()


def _orch_parse_tool_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            return json.loads(raw)
        except Exception:
            return {}
    return {}


@lru_cache(maxsize=1)
def _orch_load_rag_corpus() -> str:
    path = Path(__file__).resolve().parent / "rag" / "drugsfda_dashboard_notes.md"
    return path.read_text(encoding="utf-8")


def rag_retrieve_dashboard_notes(*, query: str, max_chars: int = 2200) -> str:
    """
    RAG retriever over bundled markdown (lexical chunk scoring — class demo, not a vector DB).
    """
    corpus = _orch_load_rag_corpus()
    chunks = [c.strip() for c in re.split(r"\n{2,}", corpus) if c.strip()]
    q_terms = [t for t in re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", (query or "").lower())]
    q_terms = list(dict.fromkeys(q_terms))[:40]

    scored: list[tuple[float, str]] = []
    for ch in chunks:
        low = ch.lower()
        score = 0.0
        for t in q_terms:
            if t in low:
                score += low.count(t)
        scored.append((score, ch))

    scored.sort(key=lambda x: x[0], reverse=True)
    out: list[str] = []
    total = 0
    if scored and scored[0][0] <= 0:
        scored = [(0.0, ch) for _, ch in scored[:6]]

    for score, ch in scored:
        piece = f"(score={score:.0f})\n{ch}"
        if total + len(piece) > max_chars:
            break
        out.append(piece)
        total += len(piece)

    if not out:
        out.append(chunks[0] if chunks else "(empty corpus)")

    return "\n\n---\n\n".join(out)


def tool_openfda_drugsfda_search(*, search: str, limit: int = 5) -> dict[str, Any]:
    """Tool: GET openFDA drug/drugsfda.json with lucene-style search."""
    lim = max(1, min(int(limit), 25))
    q = (search or "").strip()
    if not q:
        return {"error": "search is empty"}

    params: dict[str, Any] = {"limit": lim, "search": q}
    key = (os.getenv("OPENFDA_API_KEY") or "").strip()
    if key:
        params["api_key"] = key

    url = "https://api.fda.gov/drug/drugsfda.json"
    r = requests.get(url, params=params, timeout=60.0)
    r.raise_for_status()
    data = r.json()
    results = data.get("results") or []

    slim: list[dict[str, Any]] = []
    for rec in results[:lim]:
        slim.append(
            {
                "application_number": rec.get("application_number"),
                "sponsor_name": rec.get("sponsor_name"),
                "submission_count": len(rec.get("submissions") or []),
                "product_count": len(rec.get("products") or []),
            }
        )

    meta = data.get("meta") or {}
    return {
        "endpoint": url,
        "search": q,
        "limit": lim,
        "results_total": (meta.get("results") or {}).get("total"),
        "results_returned": len(results),
        "sample": slim,
    }


def tool_dashboard_df_metrics(*, df_json: str) -> dict[str, Any]:
    """Tool: deterministic metrics from filtered dashboard rows (JSON array, same as charts)."""
    try:
        rows = json.loads(df_json)
    except Exception as e:
        return {"error": f"invalid df_json: {e}"}

    if not isinstance(rows, list) or not rows:
        return {"error": "df_json must be a non-empty JSON array"}

    dfx = pd.DataFrame(rows)
    out: dict[str, Any] = {"rows": int(len(dfx)), "columns": [str(c) for c in dfx.columns.tolist()]}

    if "year" in dfx.columns:
        out["year_min"] = int(dfx["year"].min()) if len(dfx) else None
        out["year_max"] = int(dfx["year"].max()) if len(dfx) else None

    if "application_kind" in dfx.columns:
        vc = dfx["application_kind"].astype(str).value_counts()
        out["application_kind_counts"] = {str(k): int(v) for k, v in vc.items()}

    if "sponsor_name" in dfx.columns:
        vc = dfx["sponsor_name"].astype(str).value_counts().head(5)
        out["top_sponsors_events"] = [{"sponsor": str(k), "events": int(v)} for k, v in vc.items()]

    return out


def tool_build_openfda_query_url(*, search: str) -> dict[str, str]:
    """Tool: build a reproducible openFDA drugsfda query URL."""
    q = (search or "").strip()
    if not q:
        return {"error": "search is empty"}
    base = "https://api.fda.gov/drug/drugsfda.json"
    qs = urllib.parse.urlencode({"search": q, "limit": 25})
    return {"url": f"{base}?{qs}"}


def _orch_tool_dispatch(name: str, arguments: dict[str, Any]) -> Any:
    if name == "openfda_drugsfda_search":
        return tool_openfda_drugsfda_search(**arguments)
    if name == "dashboard_df_metrics":
        return tool_dashboard_df_metrics(**arguments)
    if name == "build_openfda_query_url":
        return tool_build_openfda_query_url(**arguments)
    if name == "retrieve_dashboard_notes":
        return rag_retrieve_dashboard_notes(**arguments)
    return {"error": f"unknown tool: {name}"}


def ollama_tool_round(
    *,
    system: str,
    user: str,
    tools: list[dict[str, Any]],
    max_rounds: int = 3,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Run up to max_rounds tool-calling rounds against Ollama /api/chat.
    Returns (final assistant text, transcript rows for logging).
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    transcript: list[dict[str, Any]] = []

    for round_idx in range(1, max_rounds + 1):
        data = _orch_native_chat(messages, tools=tools, temperature=0.1, num_predict=900)
        msg = data.get("message") or {}
        tool_calls = msg.get("tool_calls") or []

        if not tool_calls:
            return _orch_msg_text(data), transcript

        messages.append({"role": "assistant", "content": msg.get("content"), "tool_calls": tool_calls})

        for tc in tool_calls:
            fn = ((tc.get("function") or {}).get("name") or "").strip()
            raw_args = (tc.get("function") or {}).get("arguments")
            args = _orch_parse_tool_args(raw_args)

            try:
                out = _orch_tool_dispatch(fn, args)
            except Exception as e:
                out = {"error": str(e)}

            payload = json.dumps(out, default=str)
            transcript.append(
                {
                    "round": round_idx,
                    "tool": fn,
                    "args": args,
                    "output_preview": payload[:800],
                    "ok": _orch_tool_out_ok(out),
                    "invoked_by": "model",
                }
            )

            messages.append({"role": "tool", "tool_name": fn, "content": payload[:12000]})

    return "Tool calling stopped early (too many tool rounds).", transcript


# Explicit tool surface (documentation / homework table — compare pokeapp TOOLS_REGISTRY)
DASHBOARD_ORCH_TOOLS_REGISTRY: List[Dict[str, str]] = [
    {
        "name": "dashboard_df_metrics",
        "purpose": "Compute summary stats from the filtered dashboard dataframe JSON.",
        "parameters": "df_json: str — records JSON array (first N rows passed by orchestrator).",
        "returns": "dict: row count, columns, year range, kind counts, top sponsors.",
    },
    {
        "name": "openfda_drugsfda_search",
        "purpose": "Query openFDA drugsfda.json (supplemental evidence beyond the chart slice).",
        "parameters": "search: str, limit: int.",
        "returns": "dict: sample application rows + meta totals.",
    },
    {
        "name": "build_openfda_query_url",
        "purpose": "Emit the exact HTTPS query URL for transparency / reproducibility.",
        "parameters": "search: str.",
        "returns": "dict with url.",
    },
    {
        "name": "retrieve_dashboard_notes",
        "purpose": "RAG over rag/drugsfda_dashboard_notes.md (interpretation guardrails).",
        "parameters": "query: str, optional max_chars.",
        "returns": "str — scored note chunks.",
    },
]


@dataclass
class PlannerPlan:
    run_df_metrics: bool = True
    openfda_search: str | None = None
    openfda_limit: int = 8
    rag_query: str = ""


def _orch_default_plan(ctx: str) -> PlannerPlan:
    q = " ".join(ctx.splitlines())[:400]
    return PlannerPlan(
        run_df_metrics=True,
        openfda_search=None,
        openfda_limit=8,
        rag_query=q,
    )


@dataclass
class PlannerAgentOutput:
    plan: PlannerPlan
    raw_json_text: str
    parse_ok: bool
    parse_error: str | None = None


def planner_agent(*, dashboard_context: str) -> PlannerAgentOutput:
    """Agent 1 (Planner): JSON plan only — no tools."""
    system = "\n".join(
        [
            "You are the Planner Agent.",
            "Decide what the downstream agents should do to explain a Drugs@FDA dashboard window.",
            "Return ONLY valid JSON (no markdown) with this schema:",
            "{",
            '  "run_df_metrics": true/false,',
            '  "openfda_search": string|null,',
            '  "openfda_limit": integer,',
            '  "rag_query": string',
            "}",
            "Rules:",
            "- Prefer run_df_metrics=true.",
            "- openfda_search should be a concise openFDA `search` string OR null if not needed.",
            "- rag_query should be short keywords useful for retrieving interpretation notes.",
        ]
    )
    user = "DASHBOARD_CONTEXT:\n" + (dashboard_context or "").strip()
    try:
        data = _orch_native_chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            format_json=True,
        )
        blob = _orch_msg_text(data)
        obj = json.loads(blob) if blob else {}
        plan = PlannerPlan(
            run_df_metrics=bool(obj.get("run_df_metrics", True)),
            openfda_search=(obj.get("openfda_search") if obj.get("openfda_search") not in (None, "", "null") else None),
            openfda_limit=int(obj.get("openfda_limit") or 8),
            rag_query=str(obj.get("rag_query") or "")[:500],
        )
        pretty = json.dumps(
            {
                "run_df_metrics": plan.run_df_metrics,
                "openfda_search": plan.openfda_search,
                "openfda_limit": plan.openfda_limit,
                "rag_query": plan.rag_query,
            },
            indent=2,
        )
        return PlannerAgentOutput(plan=plan, raw_json_text=blob.strip() or pretty, parse_ok=True)
    except Exception as e:
        plan = _orch_default_plan(user)
        pretty = json.dumps(
            {
                "run_df_metrics": plan.run_df_metrics,
                "openfda_search": plan.openfda_search,
                "openfda_limit": plan.openfda_limit,
                "rag_query": plan.rag_query,
            },
            indent=2,
        )
        return PlannerAgentOutput(plan=plan, raw_json_text=pretty, parse_ok=False, parse_error=str(e))


def data_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "dashboard_df_metrics",
                "description": "Compute deterministic metrics from the filtered dashboard dataframe JSON (same rows as charts).",
                "parameters": {
                    "type": "object",
                    "properties": {"df_json": {"type": "string"}},
                    "required": ["df_json"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "openfda_drugsfda_search",
                "description": "Query openFDA drugsfda.json with a lucene-style search string (tool calling).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search": {"type": "string"},
                        "limit": {"type": "integer"},
                    },
                    "required": ["search"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "build_openfda_query_url",
                "description": "Build a clickable openFDA drugsfda query URL for transparency.",
                "parameters": {
                    "type": "object",
                    "properties": {"search": {"type": "string"}},
                    "required": ["search"],
                },
            },
        },
    ]


def _orch_transcript_has_tool_ok(transcript: list[dict[str, Any]], name: str) -> bool:
    for row in transcript:
        if (row.get("tool") or "").strip() != name:
            continue
        if row.get("ok", True):
            return True
    return False


def _orch_transcript_row(
    *,
    round_idx: int,
    tool: str,
    args: dict[str, Any],
    out: Any,
    invoked_by: str,
) -> dict[str, Any]:
    payload = json.dumps(out, default=str)
    return {
        "round": round_idx,
        "tool": tool,
        "args": args,
        "output_preview": payload[:800],
        "ok": _orch_tool_out_ok(out),
        "invoked_by": invoked_by,
    }


def _orch_summarize_metrics_for_notes(out: dict[str, Any]) -> list[str]:
    if out.get("error"):
        return [f"- **dashboard_df_metrics** error: {out['error']}"]
    bits: list[str] = [f"- Filtered chart slice: **{out.get('rows')}** rows."]
    ymin, ymax = out.get("year_min"), out.get("year_max")
    if ymin is not None and ymax is not None:
        bits.append(f"- Year range in slice: **{ymin}–{ymax}**.")
    cts = out.get("application_kind_counts") or {}
    if isinstance(cts, dict) and cts:
        parts = [f"{k}: {v}" for k, v in sorted(cts.items(), key=lambda kv: -int(kv[1]))[:8]]
        bits.append("- Application kind counts (events): " + ", ".join(parts) + ".")
    tops = out.get("top_sponsors_events") or []
    if tops:
        sp = ", ".join(f"{x.get('sponsor')} ({x.get('events')} events)" for x in tops[:5])
        bits.append(f"- Top sponsors (sample): {sp}.")
    return bits


def _orch_summarize_openfda_for_notes(out: dict[str, Any]) -> list[str]:
    if out.get("error"):
        return [f"- **openFDA** sample error: {out['error']}"]
    rt = out.get("results_total")
    rr = out.get("results_returned")
    lines: list[str] = [f"- openFDA supplement search: **{rr}** row(s) returned (total matching: **{rt}**)."]
    for i, row in enumerate((out.get("sample") or [])[:3], start=1):
        lines.append(
            f"  - Sample {i}: app **{row.get('application_number')}**, sponsor **{row.get('sponsor_name')}**."
        )
    return lines


def _orch_server_fallback_data_plan_tools(
    *,
    plan: PlannerPlan,
    df_json: str,
    existing: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    """
    If the Data agent (Ollama) omits tool calls, still execute PLAN_JSON requirements here so
    DATA_NOTES and logs show successful tool runs.
    """
    rows: list[dict[str, Any]] = []
    bullets: list[str] = []
    base_round = max((int(r.get("round") or 0) for r in existing), default=0)
    rnext = base_round + 1

    need_metrics = plan.run_df_metrics and not _orch_transcript_has_tool_ok(existing, "dashboard_df_metrics")
    q = (plan.openfda_search or "").strip()
    need_openfda = bool(q) and not _orch_transcript_has_tool_ok(existing, "openfda_drugsfda_search")

    if need_metrics:
        margs = {"df_json": df_json}
        mout = tool_dashboard_df_metrics(**margs)
        rows.append(
            _orch_transcript_row(
                round_idx=rnext,
                tool="dashboard_df_metrics",
                args=margs,
                out=mout,
                invoked_by="server_fallback",
            )
        )
        bullets.extend(_orch_summarize_metrics_for_notes(mout if isinstance(mout, dict) else {"error": str(mout)}))
        rnext += 1

    if need_openfda:
        lim = max(1, min(int(plan.openfda_limit or 8), 25))
        oargs = {"search": q, "limit": lim}
        try:
            oout = tool_openfda_drugsfda_search(**oargs)
        except Exception as e:
            oout = {"error": str(e)}
        rows.append(
            _orch_transcript_row(
                round_idx=rnext,
                tool="openfda_drugsfda_search",
                args=oargs,
                out=oout,
                invoked_by="server_fallback",
            )
        )
        bullets.extend(_orch_summarize_openfda_for_notes(oout if isinstance(oout, dict) else {"error": str(oout)}))
        rnext += 1
        uargs = {"search": q}
        try:
            uout = tool_build_openfda_query_url(**uargs)
        except Exception as e:
            uout = {"error": str(e)}
        rows.append(
            _orch_transcript_row(
                round_idx=rnext,
                tool="build_openfda_query_url",
                args=uargs,
                out=uout,
                invoked_by="server_fallback",
            )
        )
        if isinstance(uout, dict) and uout.get("url"):
            bullets.append(f"- Reproducible query URL: `{uout['url']}`")

    return rows, "\n".join(bullets).strip()


def rag_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "retrieve_dashboard_notes",
                "description": "Retrieve relevant stakeholder interpretation notes from the bundled markdown corpus (RAG).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_chars": {"type": "integer"},
                    },
                    "required": ["query"],
                },
            },
        }
    ]


def _qc_pct_allowlist(pool: str) -> set[str]:
    """Normalized `%` literals that appear anywhere in the grounded context blob."""
    return {m.replace(" ", "").lower() for m in re.findall(r"\d+\.?\d*\s*%", pool or "")}


def _qc_automated_chart_checks(final: str, draft: str, pool: str, *, validator_ran: bool) -> list[str]:
    """Lightweight post-hoc checks for orchestrator output (evidence for QC / coursework)."""
    lines: list[str] = []
    final = (final or "").strip()
    draft = (draft or "").strip()
    allowed_pct = _qc_pct_allowlist(pool)

    if not final:
        lines.append("- **Non-empty output:** `FAIL` (final text is empty).")
        return lines

    lines.append("- **Non-empty output:** `PASS`.")

    if validator_ran and draft:
        if len(final) < max(40, int(0.12 * len(draft))) and len(draft) > 220:
            lines.append(
                "- **Length vs Insight draft:** `WARN` — final text is much shorter than the draft; "
                "check server logs in case the Validator over-trimmed."
            )
        else:
            lines.append("- **Length vs Insight draft:** `PASS` (no abnormal collapse).")
        if final.strip() == draft.strip():
            lines.append("- **Validator delta:** `INFO` — Validator output matches the draft verbatim (no edits applied).")
    elif draft:
        lines.append("- **Validator:** not used (Insight draft was empty — Validator skipped).")

    pct_tokens = re.findall(r"\d+\.?\d*\s*%", final)
    bad: list[str] = []
    for raw in pct_tokens:
        norm = raw.replace(" ", "").lower()
        if norm not in allowed_pct:
            bad.append(raw.strip())
    if bad:
        uniq = list(dict.fromkeys(bad))[:6]
        lines.append(
            "- **Percentage spot-check vs CONTEXT/DATA_NOTES:** `WARN` — these % literals were not found "
            f"in the grounded text blob: {', '.join(uniq)}. (Heuristic; small mismatches can be formatting.)"
        )
    else:
        if pct_tokens:
            lines.append("- **Percentage spot-check vs CONTEXT/DATA_NOTES:** `PASS` (all % tokens appear in grounded pool).")
        else:
            lines.append("- **Percentage spot-check:** `SKIP` (no `%` tokens in the answer).")

    return lines


def _chart_ai_qc_markdown_footer(
    *,
    validator_ran: bool,
    insight: str,
    validated: str,
    ctx: str,
    data_notes: str,
) -> str:
    """Visible QC evidence appended to the chart AI answer (Markdown)."""
    pool = f"{ctx}\n{data_notes}"
    auto_lines = _qc_automated_chart_checks(
        (validated if validator_ran else insight),
        insight,
        pool,
        validator_ran=validator_ran,
    )
    body: list[str] = [
        "---",
        "### Quality Control (Dashboard Chart AI)",
        "",
        "**Pipeline**",
    ]
    if validator_ran:
        body.extend(
            [
                "1. **Insight agent** — Draft stakeholder narrative from CONTEXT + DATA_NOTES + RAG_NOTES.",
                "2. **Validator agent** — Second pass with explicit QC rules (see below).",
                "3. **Final** — Text shown above this footer (Markdown-rendered in the app).",
                "",
                "**Validator agent — enforced QC rules**",
                "- **Output verification:** Every quantitative claim must remain traceable to **DASHBOARD_CONTEXT** or **DATA_NOTES** (no new numbers, no invented sponsors or years).",
                "- **Error checking:** Remove internal contradictions, unsupported superlatives, and speculation not grounded in those sources.",
                "- **Stakeholder safety:** Clear, cautious wording; label uncertainty where the data are thin.",
                "- **Format:** Keep 3–6 sentences or short bullets; preserve Markdown where sensible.",
            ]
        )
    else:
        body.extend(
            [
                "**Pipeline note:** The **Insight** draft was empty, so the **Validator** step did not run. "
                "Check server logs for earlier agents if you expected narrative output.",
            ]
        )
    body.extend(
        [
            "",
            "**Automated post-checks (Python, after the model returns)**",
            *auto_lines,
        ]
    )
    return "\n".join(body)


def summarize_dashboard_charts_tool_rag(aggregated_chart_data: str, df: pd.DataFrame) -> str:
    """
    Ollama-only multi-agent pipeline for the dashboard chart AI box.

    Prints each agent to stdout (like agents_pokemon.terminal_demo).
    Returns final Markdown narrative plus a **Quality control** footer for the Shiny UI.
    """
    _orch_require_ollama_backend()
    if not dashboard_orchestrator_enabled():
        raise RuntimeError("dashboard orchestrator disabled")
    if df is None or df.empty:
        return "No filtered rows available for orchestrated dashboard AI (load data / adjust filters)."

    ctx = (aggregated_chart_data or "").strip() or "(empty CONTEXT)"
    df_small = df.head(2000)
    df_json = df_small.to_json(orient="records", date_format="iso")

    print("", flush=True)
    print(_orch_hr("="), flush=True)
    print("agents_drug.py — Dashboard chart AI (Ollama multi-agent + tools + RAG)", flush=True)
    print(_orch_hr("="), flush=True)
    print("Tool surface (function-calling registry):", flush=True)
    for t in DASHBOARD_ORCH_TOOLS_REGISTRY:
        print(f"  - {t.get('name')}: {t.get('purpose')}", flush=True)
    print(f"OLLAMA_HOST={_orch_ollama_base()}  OLLAMA_MODEL={_orch_ollama_model()}", flush=True)
    print(
        f"DASHBOARD_AI_ORCH_PARALLEL={'1' if _orch_parallel_data_rag_enabled() else '0'} "
        "(Agent 2 + 3 tool rounds run concurrently when 1)",
        flush=True,
    )
    print("Chart AI QC: Validator runs automatically after a non-empty Insight.", flush=True)
    print("", flush=True)

    print("Agent 1 — Planner (JSON plan, no tools) ...", flush=True)
    planner = planner_agent(dashboard_context=ctx)
    plan = planner.plan
    planner_print = "\n".join(
        [
            f"parse_ok: {planner.parse_ok}",
            *( [f"parse_error: {planner.parse_error}"] if planner.parse_error else [] ),
            "",
            "PLAN_JSON:",
            planner.raw_json_text.strip(),
            "",
            "PLAN_NORMALIZED:",
            json.dumps(
                {
                    "run_df_metrics": plan.run_df_metrics,
                    "openfda_search": plan.openfda_search,
                    "openfda_limit": plan.openfda_limit,
                    "rag_query": plan.rag_query,
                },
                indent=2,
            ),
        ]
    ).strip()
    print("----- AGENT 1 OUTPUT (Planner) -----", flush=True)
    print(planner_print, flush=True)
    print("------------------------------------", flush=True)
    print("", flush=True)

    data_system = "\n".join(
        [
            "You are the Data Agent.",
            "Use tools to gather evidence for the dashboard interpretation.",
            "Hard requirements:",
            "- Always call `dashboard_df_metrics` if run_df_metrics is true in PLAN_JSON.",
            "- If PLAN_JSON.openfda_search is non-null, call `openfda_drugsfda_search` with that search string and the limit.",
            "- If you call openfda_drugsfda_search, also call `build_openfda_query_url` with the same search string.",
            "After tools, respond with a short bullet list of factual findings + caveats (no storytelling yet).",
        ]
    )
    plan_blob = json.dumps(
        {
            "run_df_metrics": plan.run_df_metrics,
            "openfda_search": plan.openfda_search,
            "openfda_limit": plan.openfda_limit,
        },
        indent=2,
    )
    data_user = "\n\n".join(
        [
            "PLAN_JSON:",
            plan_blob,
            "",
            "DASHBOARD_CONTEXT:",
            ctx,
            "",
            "FILTERED_DF_JSON (first up to 2000 rows):",
            df_json[:12000],
        ]
    )
    rag_query = (plan.rag_query or "").strip() or " ".join(ctx.splitlines())[:240]
    rag_system = "\n".join(
        [
            "You are the RAG Agent.",
            "Use `retrieve_dashboard_notes` to pull the most relevant interpretation notes for this dashboard window.",
            "After the tool result, summarize the retrieved notes in 3–6 bullets (what to emphasize / what to avoid claiming).",
        ]
    )
    rag_user = "\n\n".join(["DASHBOARD_CONTEXT:\n" + ctx, "RAG_QUERY:\n" + rag_query])

    def _orch_data_tool_round() -> tuple[str, list[dict[str, Any]]]:
        return ollama_tool_round(system=data_system, user=data_user, tools=data_tools(), max_rounds=3)

    def _orch_rag_tool_round() -> tuple[str, list[dict[str, Any]]]:
        return ollama_tool_round(system=rag_system, user=rag_user, tools=rag_tools(), max_rounds=2)

    if _orch_parallel_data_rag_enabled():
        print(
            "Agent 2 — Data & Agent 3 — RAG (parallel Ollama tool rounds; set DASHBOARD_AI_ORCH_PARALLEL=0 to serialize) ...",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_d = pool.submit(_orch_data_tool_round)
            fut_r = pool.submit(_orch_rag_tool_round)
            data_notes, data_dbg = fut_d.result()
            rag_notes, rag_dbg = fut_r.result()
    else:
        print("Agent 2 — Data (Ollama tool calling) ...", flush=True)
        data_notes, data_dbg = _orch_data_tool_round()
        print("Agent 3 — RAG (tool: retrieve_dashboard_notes) ...", flush=True)
        rag_notes, rag_dbg = _orch_rag_tool_round()

    model_dbg = list(data_dbg)
    fb_rows, fb_notes = _orch_server_fallback_data_plan_tools(plan=plan, df_json=df_json, existing=data_dbg)
    if fb_rows:
        data_dbg = data_dbg + fb_rows
        if fb_notes.strip():
            data_notes = (
                (data_notes.strip() + "\n\n" + fb_notes.strip()).strip() if data_notes.strip() else fb_notes.strip()
            )

    print("----- AGENT 2 OUTPUT (Data — model text) -----", flush=True)
    print(data_notes.strip() or "(no output)", flush=True)
    print("----- AGENT 2 TOOL TRANSCRIPT -----", flush=True)
    print(_orch_tool_transcript_lines(data_dbg), flush=True)
    print("----- AGENT 2 STATUS -----", flush=True)
    if fb_rows:
        n_ok = sum(1 for r in fb_rows if r.get("ok", True))
        if n_ok == len(fb_rows):
            print("tools_completed_successfully (server-side fallback ran missing plan steps)", flush=True)
        else:
            print(
                f"tools_partial_success_in_fallback ({n_ok}/{len(fb_rows)} ok); inspect transcript [failed] lines",
                flush=True,
            )
    elif model_dbg:
        print("tools_completed_successfully (model invoked tool calls)", flush=True)
    else:
        print(
            "no_data_tools_required_by_plan (PLAN_JSON: run_df_metrics=false and no openfda_search)",
            flush=True,
        )
    print("-----------------------------------", flush=True)
    print("", flush=True)

    print("----- AGENT 3 OUTPUT (RAG — model text) -----", flush=True)
    print(rag_notes.strip() or "(no output)", flush=True)
    print("----- AGENT 3 TOOL TRANSCRIPT -----", flush=True)
    print(_orch_tool_transcript_lines(rag_dbg), flush=True)
    print("-----------------------------------", flush=True)
    print("", flush=True)

    insight_system = "\n".join(
        [
            "You are the Insight Agent.",
            "Write the final dashboard explanation for business stakeholders.",
            "Ground quantitative claims in DASHBOARD_CONTEXT and DATA_NOTES.",
            "Use RAG_NOTES only for interpretation guidance (sampling limits, AP-only foundation, etc.).",
            "If supplemental openFDA samples were retrieved, label them clearly and do not contradict DASHBOARD_CONTEXT.",
            "Output: 3–6 concise sentences OR short bullets; use Markdown (**bold**, lists) for the UI.",
        ]
    )
    insight_user = "\n\n".join(
        [
            "DASHBOARD_CONTEXT:\n" + ctx,
            "DATA_NOTES:\n" + data_notes,
            "RAG_NOTES:\n" + rag_notes,
        ]
    )
    print("Agent 4 — Insight (final narrative, no tools) ...", flush=True)
    insight = _orch_chat_plain_text(
        [{"role": "system", "content": insight_system}, {"role": "user", "content": insight_user}],
        temperature=0.2,
        num_predict=900,
        timeout=120.0,
    ).strip()
    print("----- AGENT 4 OUTPUT (Insight — this text is shown in the app) -----", flush=True)
    print(insight or "(no output)", flush=True)
    print("----------------------------------------------------------------------", flush=True)
    print("", flush=True)

    validated = insight
    validator_ran = False
    if insight:
        val_system = "\n".join(
            [
                "You are the Validator Agent — a quality-control editor for dashboard chart AI.",
                "",
                "You receive DASHBOARD_CONTEXT, DATA_NOTES (tool-grounded facts), and a DRAFT narrative.",
                "Return ONLY the revised narrative (no preamble, no meta-commentary about your edits).",
                "",
                "Quality control checklist (apply every time):",
                "1) **Output verification** — Every number, percentage, year, sponsor name, and count in your output must appear",
                "   in DASHBOARD_CONTEXT or DATA_NOTES (or be clearly qualitative, e.g. “mix”, “trend”). Remove or rephrase anything you cannot verify.",
                "2) **Error checking** — Fix contradictions, vague claims contradicted by the context, and overconfident language when the data are sparse.",
                "3) **No fabrication** — Do not invent sponsors, products, dates, or statistics. Do not add new quantitative claims.",
                "4) **Stakeholder tone** — Concise, neutral, decision-useful; flag uncertainty briefly when appropriate.",
                "5) **Length & format** — At most 3–6 short sentences or bullets. Preserve Markdown from the draft where it still fits.",
            ]
        )
        val_user = "\n\n".join(
            [
                "DASHBOARD_CONTEXT:\n" + ctx,
                "DATA_NOTES:\n" + data_notes,
                "DRAFT:\n" + insight,
            ]
        )
        print("Agent 5 — Validator (QC pass: verification + contradiction checks) ...", flush=True)
        validated = _orch_chat_plain_text(
            [{"role": "system", "content": val_system}, {"role": "user", "content": val_user}],
            temperature=0.2,
            num_predict=600,
            timeout=120.0,
        ).strip() or insight
        validator_ran = True
        print("----- AGENT 5 OUTPUT (Validator) -----", flush=True)
        print(validated, flush=True)
        print("--------------------------------------", flush=True)
        print("", flush=True)

    final_out = (validated if validator_ran else insight).strip()
    qc_footer = _chart_ai_qc_markdown_footer(
        validator_ran=validator_ran,
        insight=insight,
        validated=validated,
        ctx=ctx,
        data_notes=data_notes,
    )
    combined = (final_out or "(no output)") + "\n\n" + qc_footer
    print("===== FINAL (returned to Shiny UI) =====", flush=True)
    print(final_out or "(no output)", flush=True)
    print("(QC footer appended in Markdown — see app panel.)", flush=True)
    print(_orch_hr("="), flush=True)
    return combined


# 8. HTTP ERROR HELPERS — user-readable failures #################################


def _friendly_openai_http_error(e: requests.HTTPError) -> str:
    resp = e.response
    if resp is None:
        return f"OpenAI HTTP error: {e}"
    try:
        data = resp.json()
        err = data.get("error") or {}
        code = err.get("code") or ""
        msg = (err.get("message") or "").strip()
        if code == "insufficient_quota" or "exceeded your current quota" in msg.lower():
            return (
                "OpenAI: insufficient quota / billing. See https://platform.openai.com/account/billing\n"
                f"(code: {code or 'insufficient_quota'})"
            )
        if code == "rate_limit_exceeded":
            return "OpenAI: rate limit — wait briefly and try again."
    except Exception:
        pass
    body = (resp.text or "")[:800] if resp else ""
    return f"OpenAI HTTP error: {e}\n{body}"


def _friendly_ollama_http_error(e: requests.HTTPError) -> str:
    mod = os.getenv("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL
    if e.response is None:
        return f"Ollama HTTP error: {e}"
    try:
        data = e.response.json()
        em = str(data.get("error") or "")
        el = em.lower()
        if e.response.status_code == 404 and "not found" in el and "model" in el:
            return (
                f"Ollama: model '{mod}' not found. Run: ollama pull {mod}  OR set OLLAMA_MODEL to a name from ollama list.\n"
                f"Server: {em}"
            )
    except Exception:
        pass
    snippet = (e.response.text or "")[:600] if e.response else ""
    return f"Ollama HTTP error: {e}\n{snippet}"


# 9. DASHBOARD “AGENT” — single LLM interpretation step ########################
# Workflow (compare to lab Agent 1 fetch → Agent 2 analyze → Agent 3 press release):
#   Step A — aggregate_full_dashboard_context(df)  [this file, section 4]
#   Step B — summarize_dashboard_charts(context)    [this section — role ~ “analyst / narrator”]


def summarize_dashboard_charts(aggregated_chart_data: str, df: pd.DataFrame | None = None) -> str:
    """
    Run Template 3: turn aggregated CONTEXT into a short narrative for the business audience.

    Default backend: Ollama. Set AI_BACKEND=openai and OPENAI_API_KEY for OpenAI.

    Optional (TOOL2-style): set `DASHBOARD_AI_ORCHESTRATOR=1` and provide `df` to run an
    **Ollama-only** multi-agent workflow (section 7) with **tool calling** + **RAG** over
    `rag/drugsfda_dashboard_notes.md`. Per-agent traces are **printed** to stdout; the string
    returned to Shiny is the **Validator** narrative (after Insight) plus a **Quality control** footer.

    Returns
    -------
    str
        Model text (orchestrator: Validator output + QC footer), or an error message string (never raises to Shiny).
    """
    # Optional TOOL2-style path: Ollama multi-agent + tool calling + bundled markdown RAG.
    # Enable with DASHBOARD_AI_ORCHESTRATOR=1 (Validator QC always follows a non-empty Insight).
    if dashboard_orchestrator_enabled():
        if df is None or df.empty:
            return (
                "Dashboard AI orchestrator is enabled, but there is no filtered chart dataframe to ground tools.\n"
                "Load data and ensure filters produce non-empty AP rows."
            )
        try:
            return summarize_dashboard_charts_tool_rag(aggregated_chart_data, df)
        except Exception as e:
            return f"Dashboard AI orchestrator error: {e}"

    prompt = build_chart_explanation_prompt(aggregated_chart_data)
    messages = [{"role": "user", "content": prompt}]
    try:
        data = _llm_chat(messages, temperature=0.35, timeout=90.0)
        txt = _message_text(data["choices"][0]["message"]).strip()
        if not txt:
            # Ollama Cloud can return thinking-only messages (content=""). Fall back to /api/generate.
            if _llm_backend() != "openai":
                try:
                    alt = _ollama_generate(prompt, temperature=0.35, timeout=90.0)
                    if alt:
                        return alt
                except Exception:
                    pass

            backend = _llm_backend()
            need = "OLLAMA_API_KEY (for https://ollama.com)" if backend != "openai" else "OPENAI_API_KEY"
            return (
                "AI summary returned empty text.\n"
                f"- Backend: {backend}\n"
                f"- Check: {need}\n"
                "If this persists, try a different model or backend."
            )
        return txt
    except ValueError as e:
        return str(e)
    except requests.HTTPError as e:
        if _llm_backend() == "openai":
            return _friendly_openai_http_error(e)
        return _friendly_ollama_http_error(e)
    except requests.RequestException as e:
        # If Ollama isn't reachable (common on deployed servers), try OpenAI if available.
        if _llm_backend() != "openai" and os.getenv("OPENAI_API_KEY"):
            try:
                data = _openai_chat(messages, temperature=0.35, timeout=90.0)
                return _message_text(data["choices"][0]["message"]).strip()
            except Exception as e2:
                pass
        hint = (
            "If using Ollama: ensure `ollama serve` and a valid OLLAMA_MODEL. "
            "If using OpenAI: set OPENAI_API_KEY."
        )
        return f"{type(e).__name__}: {e}\n\n{hint}"
    except Exception as e:
        return f"{type(e).__name__}: {e}"


# 10. CLI — offline tool registry (compare agents_pokemon.py __main__) ##########
def terminal_demo_dashboard_orch_tools() -> int:
    """No network: print the orchestrator tool surface for screenshots / homework tables."""
    print()
    print("=" * 63)
    print("agents_drug.py — Dashboard orchestrator (tool registry, offline)")
    print("=" * 63)
    print("\nTool definitions (function-calling surface when DASHBOARD_AI_ORCHESTRATOR=1):\n")
    for t in DASHBOARD_ORCH_TOOLS_REGISTRY:
        print(f"  - {t.get('name')}")
        print(f"      purpose: {t.get('purpose')}")
        print(f"      parameters: {t.get('parameters')}")
        print(f"      returns: {t.get('returns')}")
        print()
    print("To print every agent in this terminal (needs Ollama + tool model):", flush=True)
    print("  python agents_drug.py run", flush=True)
    print("Or set DASHBOARD_AI_ORCHESTRATOR=1 and use Shiny “Explain chart trends” (watch the `shiny run` terminal).\n")
    return 0


def _demo_orchestrator_sample_df() -> pd.DataFrame:
    """Small AP-like frame so dashboard_df_metrics and CONTEXT match chart semantics."""
    rows: list[dict[str, Any]] = []
    spec = [
        (2020, "NDA", "Demo Sponsor A", "NDA-001"),
        (2020, "NDA", "Demo Sponsor A", "NDA-002"),
        (2021, "ANDA", "Demo Sponsor B", "ANDA-010"),
        (2022, "BLA", "Demo Sponsor C", "BLA-020"),
        (2023, "NDA", "Demo Sponsor A", "NDA-003"),
    ]
    for year, kind, sponsor, appn in spec:
        rows.append(
            {
                "year": year,
                "application_kind": kind,
                "sponsor_name": sponsor,
                "application_number": appn,
            }
        )
    return pd.DataFrame(rows)


def terminal_demo_dashboard_orchestrate_full() -> int:
    """
    Run Planner → Data → RAG → Insight → Validator (when Insight is non-empty) once.

    Prints the same agent sections as the Shiny orchestrator path; requires a running
    Ollama server and a tool-capable OLLAMA_MODEL. May call openFDA if the planner asks.
    """
    os.environ["DASHBOARD_AI_ORCHESTRATOR"] = "1"
    df = _demo_orchestrator_sample_df()
    ctx = aggregate_dashboard_charts_context(df)

    print("", flush=True)
    print("=" * 63, flush=True)
    print("agents_drug.py run — full orchestrator (terminal agent trace)", flush=True)
    print("=" * 63, flush=True)
    print("Using sample AP rows + aggregated CONTEXT (not the live Shiny filter).\n", flush=True)

    try:
        final = summarize_dashboard_charts_tool_rag(ctx, df)
    except Exception as e:
        print(f"\nOrchestrator error: {e}\n", flush=True)
        return 1

    print("\n--- Returned final (this is what Shiny shows in the chart AI box) ---\n", flush=True)
    print(final, flush=True)
    print("", flush=True)
    return 0


if __name__ == "__main__":
    args = [a.strip().lower() for a in sys.argv[1:] if a.strip()]
    if args and args[0] in ("run", "orchestrate", "agents", "full"):
        raise SystemExit(terminal_demo_dashboard_orchestrate_full())
    raise SystemExit(terminal_demo_dashboard_orch_tools())
