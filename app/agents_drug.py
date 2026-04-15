# agents_drug.py
# Dashboard chart explanation with aggregated CONTEXT → LLM (Ollama or OpenAI).
# Pattern aligned with dsai: 08_function_calling / 04_multiple_agents_with_function_calling.py
#   https://github.com/timothyfraser/dsai/blob/main/08_function_calling/04_multiple_agents_with_function_calling.py
# Here: (1) build text CONTEXT from the same DataFrame as the charts, (2) one "analyst" LLM call
# with Template 3 (no tool calling in this module — compare to agent_run + tools in the lab script).

from __future__ import annotations

# 0. SETUP ######################################################################

## 0.1 Load packages #############################################################

import os
from typing import Any

import env_load  # noqa: F401 — loads `.env` / `.env.txt` from app directory (like dotenv in labs)
import pandas as pd
import requests

# pip install pandas requests python-dotenv  (env_load uses dotenv)

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
    r = requests.post(url, json=body, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    msg = data.get("message") or {}
    return {"choices": [{"message": msg}]}


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
    return (c or "").strip() if isinstance(c, str) else ""


# 7. HTTP ERROR HELPERS — user-readable failures #################################


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


# 8. DASHBOARD “AGENT” — single LLM interpretation step ########################
# Workflow (compare to lab Agent 1 fetch → Agent 2 analyze → Agent 3 press release):
#   Step A — aggregate_full_dashboard_context(df)  [this file, section 4]
#   Step B — summarize_dashboard_charts(context)    [this section — role ~ “analyst / narrator”]


def summarize_dashboard_charts(aggregated_chart_data: str) -> str:
    """
    Run Template 3: turn aggregated CONTEXT into a short narrative for the business audience.

    Default backend: Ollama. Set AI_BACKEND=openai and OPENAI_API_KEY for OpenAI.

    Returns
    -------
    str
        Model text, or an error message string (never raises to Shiny).
    """
    prompt = build_chart_explanation_prompt(aggregated_chart_data)
    messages = [{"role": "user", "content": prompt}]
    try:
        data = _llm_chat(messages, temperature=0.35, timeout=90.0)
        return _message_text(data["choices"][0]["message"]).strip()
    except ValueError as e:
        return str(e)
    except requests.HTTPError as e:
        if _llm_backend() == "openai":
            return _friendly_openai_http_error(e)
        return _friendly_ollama_http_error(e)
    except requests.RequestException as e:
        hint = (
            "If using Ollama: ensure `ollama serve` and a valid OLLAMA_MODEL. "
            "If using OpenAI: set OPENAI_API_KEY."
        )
        return f"{type(e).__name__}: {e}\n\n{hint}"
    except Exception as e:
        return f"{type(e).__name__}: {e}"


# 9. OPTIONAL: CLI / SCRIPT TEST ##################################################
# if __name__ == "__main__":
#     print(summarize_dashboard_charts("Total AP events: 100\nNDA (brand): 40 (40.0%)..."))
