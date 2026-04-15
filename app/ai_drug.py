# ai_drug.py
# AI-powered narrative summary for a Drugs@FDA application record (openFDA-shaped dict).
# Follows the LAB_ai_reporter pattern: compact data → prompt → Ollama and/or OpenAI.
#
# Env (optional .env next to the app):
#   OPENAI_API_KEY      — if set, OpenAI chat completions is used by default
#   OPENAI_MODEL        — default gpt-4o-mini
#   OLLAMA_HOST         — default http://localhost:11434
#   OLLAMA_MODEL        — default llama3.2
#   AI_BACKEND          — openai | ollama (optional; overrides auto pick)

from __future__ import annotations

import json
import os
from typing import Any

import env_load  # noqa: F401 — loads .env from app directory
import requests

MAX_SUBMISSION_ROWS = 40
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2:latest"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


def _classify_application_kind(app_no: Any) -> str:
    if app_no is None:
        return "Other"
    s = str(app_no).strip().upper()
    if s.startswith("NDA"):
        return "NDA"
    if s.startswith("ANDA"):
        return "ANDA"
    if s.startswith("BLA"):
        return "BLA"
    return "Other"


def compact_record_for_ai(record: dict[str, Any]) -> dict[str, Any]:
    """
    Reduce and structure api_drug.extract_record output for LLM prompts.
    Truncates long submission lists to limit tokens.
    """
    products = record.get("products") or []
    subs = record.get("submissions") or []
    app_no = record.get("application_number")
    kind = _classify_application_kind(app_no)
    n_ap = sum(1 for s in subs if s.get("submission_status") == "AP")

    ingredients: list[dict[str, Any]] = []
    for p in products:
        for ai in p.get("active_ingredients") or []:
            if not isinstance(ai, dict):
                continue
            ingredients.append(
                {
                    "ingredient_name": ai.get("name"),
                    "strength": ai.get("strength"),
                    "product_brand": p.get("brand_name"),
                }
            )

    sub_sample = subs[:MAX_SUBMISSION_ROWS]
    trunc_note: str | None = None
    if len(subs) > MAX_SUBMISSION_ROWS:
        trunc_note = f"Showing first {MAX_SUBMISSION_ROWS} of {len(subs)} submission rows."

    return {
        "application_number": app_no,
        "sponsor_name": record.get("sponsor_name"),
        "application_kind_from_id": kind,
        "approval_submissions_AP_count": n_ap,
        "products": [
            {
                "brand_name": p.get("brand_name"),
                "generic_name": p.get("generic_name"),
                "marketing_status": p.get("marketing_status"),
                "reference_drug": p.get("reference_drug"),
                "dosage_form": p.get("dosage_form"),
                "route": p.get("route"),
            }
            for p in products
        ],
        "active_ingredients": ingredients,
        "submissions_sample": [
            {
                "submission_status_date": s.get("submission_status_date"),
                "submission_status": s.get("submission_status"),
                "submission_type": s.get("submission_type"),
            }
            for s in sub_sample
        ],
        "submissions_note": trunc_note,
    }


def build_summary_prompt(record: dict[str, Any]) -> str:
    """User-style prompt with embedded JSON (LAB: format data for AI consumption)."""
    payload = compact_record_for_ai(record)
    blob = json.dumps(payload, indent=2, default=str)
    return f"""You are a medical regulatory data assistant. The JSON below is derived from the openFDA Drugs@FDA API for a single drug application (not medical advice).

{blob}

Instructions:
- Write a factual summary for an analyst or pharmacist: 3–5 short paragraphs OR clear bullet sections.
- Cover: identity (brand/generic when present), sponsor, application type (NDA/ANDA/BLA) from the id, approval-related submissions (AP count; dates if visible in the sample), active ingredients and strengths, and marketing / reference-drug status across products.
- Do not invent clinical claims, dosing advice, or facts not supported by the JSON. If something is missing, say so briefly.
- Synthesize; do not dump or repeat the raw JSON."""


def call_ollama(
    prompt: str,
    *,
    model: str | None = None,
    host: str | None = None,
    timeout: float = 120.0,
) -> str:
    """POST /api/generate — same pattern as dsai 02_ollama.py."""
    base = (host or os.getenv("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST).rstrip("/")
    model = model or os.getenv("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL
    url = f"{base}/api/generate"
    body = {"model": model, "prompt": prompt, "stream": False}
    r = requests.post(url, json=body, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return str(data.get("response", "")).strip()


def call_openai(
    prompt: str,
    *,
    model: str | None = None,
    timeout: float = 120.0,
) -> str:
    """Chat Completions API (widely available; complements dsai 04_openai.py Responses example)."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY is not set.")
    model = model or os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You summarize FDA Drugs@FDA application data accurately and concisely. You do not give medical advice.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
    }
    r = requests.post(url, headers=headers, json=body, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return str(data["choices"][0]["message"]["content"]).strip()


def summarize_drug_application(record: dict[str, Any]) -> str:
    """
    Run the reporting pipeline: build prompt, then call OpenAI and/or Ollama.
    Returns plain text (caller can wrap in UI). On total failure, returns a help string (no bare raise).
    """
    prompt = build_summary_prompt(record)
    backend = (os.getenv("AI_BACKEND") or "").strip().lower()

    if backend == "openai":
        try:
            return call_openai(prompt)
        except Exception as e:
            return f"OpenAI error: {e}"

    if backend == "ollama":
        try:
            return call_ollama(prompt)
        except Exception as e:
            return (
                "Could not reach Ollama. Start the server (e.g. ollama serve), pull a model, "
                f"and check OLLAMA_HOST / OLLAMA_MODEL.\nDetails: {e}"
            )

    # Auto: prefer OpenAI if key present, else local Ollama
    if os.getenv("OPENAI_API_KEY"):
        try:
            return call_openai(prompt)
        except Exception as e:
            return f"OpenAI error: {e}"

    try:
        return call_ollama(prompt)
    except Exception as e:
        return (
            "No AI summary available. Configure one of:\n"
            "• Set OPENAI_API_KEY (and optionally OPENAI_MODEL) in .env for OpenAI, or\n"
            "• Run Ollama locally; optional OLLAMA_HOST, OLLAMA_MODEL.\n"
            "• Or set AI_BACKEND=openai or AI_BACKEND=ollama to force one backend.\n"
            f"Details: {e}"
        )
