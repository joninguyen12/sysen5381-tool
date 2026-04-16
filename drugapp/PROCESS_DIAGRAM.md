# Drugs@FDA dashboard — data flow, orchestration, grounding & tools

This diagram is **digital** (Mermaid). It maps **assignment language** (agentic orchestration, RAG-style grounding, tools) to what the app implements in `app_drug.py`, `api_drug.py`, `agents_drug.py`, and `ai_drug.py`.

**Implementation note (accuracy):** the LLM does **not** call tools via OpenAI function-calling or similar at runtime. **Tools** here are **application-layer functions** chained in code. **Grounding** is **structured tabular/text context** built from the API and filtered DataFrames—not embedding search over a vector store.

```mermaid
flowchart TB
    subgraph External["External data"]
        FDA["openFDA Drugs@FDA API\n(drug/drugsfda.json)"]
    end

    subgraph ToolsAcquire["Tool layer — acquisition & normalization"]
        T1["fetch_drugsfda(limit, skip)"]
        T2["extract_results / extract_record"]
        T3["AP foundation DataFrame\n(AP rows, year, kind, sponsor)"]
    end

    subgraph Store["Grounding source (structured, in-memory)"]
        MEM["Application records + filtered AP rows\n= retrieval target for prompts\n(no vector DB / no embeddings)"]
    end

    subgraph OrchDash["Agentic orchestration — dashboard path"]
        direction TB
        O1["Step 1: User filters\n(year, NDA/ANDA/BLA, sponsor opts, trend opts)"]
        O2["Step 2: filtered_approved_for_charts()"]
        O3["Step 3: Charts (Plotly)\n+ parallel CONTEXT build"]
        O4["aggregate_full_dashboard_context(df, …)\nRAG-style: inject same aggregates as charts into prompt"]
        O5["summarize_dashboard_charts(CONTEXT)\n_llm_chat → Ollama / OpenAI"]
        O1 --> O2 --> O3
        O2 --> O4 --> O5
    end

    subgraph OrchDrug["Agentic orchestration — drug info path"]
        direction TB
        D1["Step 1: Select application (dropdown)"]
        D2["Step 2: Structured UI\n(cards / tables from record)"]
        D3["compact_record_for_ai(record)\ntruncate submissions; JSON blob"]
        D4["summarize_drug_application(record)\n_llm / chat API → Ollama / OpenAI"]
        D1 --> D2
        D1 --> D3 --> D4
    end

    FDA --> T1 --> T2 --> T3 --> MEM
    MEM --> OrchDash
    MEM --> OrchDrug

    subgraph LLM["LLM backends (config: AI_BACKEND, keys)"]
        L1["Dashboard: Template 3 analyst prompt\n(business audience, CONTEXT-only)"]
        L2["Drug info: regulatory analyst prompt\n(JSON-grounded, no invented claims)"]
    end

    O5 --> L1
    D4 --> L2
```

## Legend

| Term in rubric | What this app does |
|----------------|-------------------|
| **Data flow** | openFDA → fetch/extract → AP-level records → filters → charts and/or drug detail. |
| **Agentic orchestration** | **Sequential pipeline**: filter → aggregate/context **or** compact record → single LLM call per button (Explain chart trends / Generate AI summary). Not a multi-agent swarm; aligns with a **single analyst step** after deterministic prep. |
| **RAG** | **Structured grounding**: prompts are filled from **retrieved fields** (same `DataFrame` metrics as charts; compact JSON for one application). **No** chunk retrieval from a vector index. |
| **Tool calling** | **Programmatic tools** (`fetch_drugsfda`, aggregation, summarizers) invoked by **Shiny/app code**, not by the model choosing functions at runtime. |

To render: paste the `flowchart` block into [Mermaid Live](https://mermaid.live) or any Markdown viewer with Mermaid support.
