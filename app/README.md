# Drugs@FDA Explorer

A **Shiny for Python** dashboard that pulls application-level records from the [openFDA Drugs@FDA API](https://open.fda.gov/apis/drug/drugsfda/), visualizes approved-drug trends and sponsor activity, and shows structured **Drug info** for a selected application.

- **Dashboard (bottom) → “Chart trends — AI summary”** uses `agents_drug.py`: metrics aligned with **approval-type pie**, **approvals-per-year trend** (peak + optional rolling average), and **top sponsors** (same Top N, metric, and normalization as the bar chart). **Template 3** prompt: 2–3 sentences. Default LLM: **Ollama**; optional **OpenAI** via `AI_BACKEND=openai` and `OPENAI_API_KEY`.
- **Drug info → “AI summary”** uses `ai_drug.py`: a **simple narrative** of the selected application (JSON context → one LLM call).

## What it does

- **Dashboard** — Fetches up to 1000 applications (configurable), foundation dataset from **AP** submissions with parseable dates; KPIs, approval-type pie, per-year trend, top sponsors, and optional **Explain chart trends** AI blurb.
- **Drug info** — Application picker, structured cards, and **Generate AI summary** for the selected drug application.
- **About** — Short notes on data sources and configuration.

| File | Role |
|------|------|
| `app_drug.py` | Shiny Express UI, charts, reactive data |
| `api_drug.py` | HTTP client for `drug/drugsfda.json`, normalizes records |
| `ai_drug.py` | Compact record + prompt; **simple Drug info summary** (Ollama or OpenAI) |
| `agents_drug.py` | **Dashboard chart explanation** (aggregate + Template 3 prompt) |
| `env_load.py` | Loads `.env` / `.env.txt` from the app directory |

## Prerequisites

- **Python 3.9+** (3.10+ recommended)
- Network access to `https://api.fda.gov` when loading data
- For AI features: **[Ollama](https://ollama.com)** locally, and/or **OpenAI** API key

## Install

```bash
pip install shiny shinyswatch plotly pandas requests python-dotenv
```

## Run locally

```bash
cd /path/to/app
shiny run app_drug.py
```

## Configuration (optional `.env`)

Place **`.env`** (or **`.env.txt`**) in the **`app/`** folder.

| Variable | Purpose |
|----------|---------|
| `AI_BACKEND` | `ollama` (default) or `openai` |
| `OLLAMA_HOST` | Default `http://localhost:11434` |
| `OLLAMA_MODEL` | Default `llama3.2:latest` — must match `ollama list` |
| `OPENAI_API_KEY` | Required if `AI_BACKEND=openai` |
| `OPENAI_MODEL` | Defaults to `gpt-4o-mini` if unset |
| `OPENFDA_API_KEY` | Optional; higher openFDA limits |

The dashboard works without any LLM; AI buttons show errors if Ollama/OpenAI is unavailable.

## Notes

- Plotly charts are embedded as HTML (no `shinywidgets`).
- Foundation analytics use **AP** submissions with parseable `submission_status_date` only.
