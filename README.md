# Drugs@FDA Explorer

A **Shiny for Python** dashboard that pulls application-level records from the [openFDA Drugs@FDA API](https://open.fda.gov/apis/drug/drugsfda/), visualizes approved-drug trends and sponsor activity, and shows structured **Drug info** (identity, approvals, ingredients, products) for a selected application. An optional **AI summary** turns the same record into a short narrative using OpenAI or a local Ollama model.

## What it does

- **Dashboard** — Fetches up to 1000 applications (configurable), builds a foundation dataset from submissions with status **AP** (approved) and valid dates, then drives:
  - KPI cards (approval events, distinct applications, year span)
  - Approval-type mix (NDA / ANDA / BLA, inferred from `application_number`)
  - Approvals-per-year trend (optional rolling average)
  - Top sponsors (with filters aligned to the charts)
- **Drug info** — Pick an application to see drug identity, approval-style KPIs, active ingredients, product details, and an **AI summary** (on demand) built in `ai_drug.py`.
- **About** — Short notes on data sources and configuration.

Supporting modules:

| File | Role |
|------|------|
| `app_drug.py` | Shiny Express UI, charts, reactive data |
| `api_drug.py` | HTTP client for `drug/drugsfda.json`, normalizes records |
| `ai_drug.py` | Compacts a record, builds a prompt, calls OpenAI or Ollama |

## Prerequisites

- **Python 3.9+** (3.10+ recommended)
- Network access to `https://api.fda.gov` when loading data

## Install

From this directory (`app/`):

```bash
pip install shiny shinyswatch plotly pandas requests python-dotenv
```

## Run locally

```bash
cd /path/to/app
shiny run app_drug.py
```

Open the URL shown in the terminal (often `http://127.0.0.1:8000`). Use **Refresh from openFDA** in the sidebar to load data, then explore the tabs.

## Configuration (optional `.env`)

Place a `.env` file next to the app files if you want keys or AI tuning.

| Variable | Purpose |
|----------|---------|
| `OPENFDA_API_KEY` | Higher rate limits for openFDA ([API key docs](https://open.fda.gov/apis/authentication/)) |
| `OPENAI_API_KEY` | Enables AI summaries via OpenAI Chat Completions |
| `OPENAI_MODEL` | Defaults to `gpt-4o-mini` if unset |
| `OLLAMA_HOST` | Defaults to `http://localhost:11434` |
| `OLLAMA_MODEL` | Defaults to `llama3.2` if unset |
| `AI_BACKEND` | `openai` or `ollama` to force a backend; otherwise OpenAI is used when `OPENAI_API_KEY` is set, else Ollama |

AI summaries are **not** required for the rest of the app; if no AI backend is available, the Drug info tab explains how to configure one.

## Notes

- Charts embed Plotly as HTML (no `shinywidgets`), which avoids common Plotly 6 / ipywidgets comm issues.
- Foundation analytics use **AP** submissions with parseable `submission_status_date` only; other statuses may still appear in raw Drug info fields.
