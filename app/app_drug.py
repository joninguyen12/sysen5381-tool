# Drugs@FDA Explorer — Shiny for Python
# Uses api_drug.py against openFDA. Light dashboard style with tabbed sections.
#
# Run: shiny run app_drug.py
# Deps: pip install shiny shinyswatch plotly pandas requests python-dotenv
# AI: ai_drug.py — Drug info summary (draft + validator). agents_drug.py — dashboard chart explanation.
# (Plotly charts are embedded as HTML — no shinywidgets — to avoid Plotly 6 / ipywidgets comm issues.)

from __future__ import annotations

import env_load  # noqa: F401 — OPENAI / OPENFDA keys from app directory .env

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import shinyswatch

from agents_drug import aggregate_full_dashboard_context, summarize_dashboard_charts
from ai_drug import summarize_drug_application
from api_drug import extract_results, fetch_drugsfda
from shiny import reactive, render
from shiny import ui as shiny_ui
from shiny.express import input, ui

# Color palette (hex / RGB aligned to spec)
# #ff0000 (255,0,0)  # #00ff00 (0,255,0)  # #0000ff (0,0,255)  # #eeeeee (238,238,238)  # #1a1a1a (26,26,26)
PALETTE_RED = "#ff0000"
PALETTE_GREEN = "#00ff00"
PALETTE_BLUE = "#0000ff"
PALETTE_LIGHT = "#eeeeee"
PALETTE_DARK = "#1a1a1a"

DARK_PALETTE = [
    PALETTE_RED,
    PALETTE_GREEN,
    PALETTE_BLUE,
    PALETTE_LIGHT,
    PALETTE_DARK,
]

# Multi-category chart fills (light UI: use dark for 4th slice so it reads on white)
CHART_FILL_COLORS = [PALETTE_RED, PALETTE_GREEN, PALETTE_BLUE, PALETTE_DARK]

KIND_LABELS = {
    "NDA": "NDA (brand)",
    "ANDA": "ANDA (generic)",
    "BLA": "BLA (biologic)",
    "Other": "Other / unknown",
}


def _latest_ap_submission_date(submissions: list[dict]) -> str | None:
    best = None
    for s in submissions or []:
        if s.get("submission_status") != "AP":
            continue
        dt = _parse_fda_date(s.get("submission_status_date"))
        if pd.isna(dt):
            continue
        if best is None or dt > best:
            best = dt
    return best.strftime("%Y-%m-%d") if best is not None else None


def _count_ap_submissions(submissions: list[dict]) -> int:
    return sum(1 for s in (submissions or []) if s.get("submission_status") == "AP")


def _identity_headline(products: list[dict]) -> str:
    if not products:
        return "—"
    p = products[0]
    bn = (p.get("brand_name") or "").strip()
    gn = (p.get("generic_name") or "").strip()
    if gn and bn:
        return f"{gn} ({bn})"
    return bn or gn or "—"


def _collect_marketing_statuses(products: list[dict]) -> str:
    seen: list[str] = []
    for p in products or []:
        m = p.get("marketing_status")
        if m and str(m) not in seen:
            seen.append(str(m))
    return ", ".join(seen) if seen else "—"


def _flatten_active_ingredients(products: list[dict]) -> list[dict]:
    rows = []
    for prod in products or []:
        bn = prod.get("brand_name") or "—"
        for ai in prod.get("active_ingredients") or []:
            if not isinstance(ai, dict):
                continue
            name = ai.get("name") or "—"
            strength = ai.get("strength")
            rows.append(
                {
                    "product_brand": bn,
                    "name": name,
                    "strength": "—" if strength is None or strength == "" else str(strength),
                }
            )
    return rows


def _parse_fda_date(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return pd.NaT
    s = str(val).strip()
    if len(s) >= 8 and s[:8].isdigit():
        return pd.to_datetime(s[:8], format="%Y%m%d", errors="coerce")
    return pd.NaT


def _format_display_date(val) -> str:
    dt = _parse_fda_date(val)
    if pd.isna(dt):
        return str(val).strip() if val else "—"
    return dt.strftime("%Y-%m-%d")


def _latest_submission_by_date(submissions: list[dict]) -> dict | None:
    best = None
    best_dt = None
    for s in submissions or []:
        dt = _parse_fda_date(s.get("submission_status_date"))
        if pd.isna(dt):
            continue
        if best_dt is None or dt > best_dt:
            best_dt = dt
            best = s
    return best


def _dataframe_table_html(df: pd.DataFrame, empty_msg: str = "No rows."):
    if df.empty:
        return ui.p(empty_msg, class_="text-muted mb-0")
    html = df.to_html(
        index=False,
        escape=True,
        classes="table table-sm table-striped text-start drug-info-pandas-table",
    )
    return ui.HTML(f'<div class="table-responsive drug-info-table-wrap">{html}</div>')


def _ai_markdown_output(txt: str):
    """Render model text as Markdown (bullets, bold, etc.) inside the AI answer panel."""
    s = (txt or "").strip()
    if not s:
        return ui.p("(Empty response)", class_="app-placeholder-hint mb-0")
    return ui.div(ui.markdown(s), class_="app-ai-output mb-0")


def _classify_application_kind(app_no) -> str:
    """Infer brand vs generic vs biologic from openFDA application_number prefix."""
    if app_no is None or (isinstance(app_no, float) and pd.isna(app_no)):
        return "Other"
    s = str(app_no).strip().upper()
    if s.startswith("NDA"):
        return "NDA"
    if s.startswith("ANDA"):
        return "ANDA"
    if s.startswith("BLA"):
        return "BLA"
    return "Other"


def _sponsor_key(name: str, normalize: bool) -> str:
    """Optional normalization: collapse whitespace + case-fold for grouping."""
    raw = " ".join(str(name or "").split()).strip() or "Unknown"
    if normalize:
        return raw.upper()
    return raw


def _build_approved_submissions_df(records: list[dict]) -> pd.DataFrame:
    """
    Foundation dataset: only submission rows with submission_status == AP and parseable dates.
    One row per AP event (multiple per drug/application possible — for trends).
    """
    rows = []
    for r in records:
        app = r.get("application_number")
        sp = r.get("sponsor_name") or "Unknown"
        kind = _classify_application_kind(app)
        for sub in r.get("submissions") or []:
            if sub.get("submission_status") != "AP":
                continue
            dt = _parse_fda_date(sub.get("submission_status_date"))
            if pd.isna(dt):
                continue
            rows.append(
                {
                    "application_number": app,
                    "sponsor_name": sp,
                    "submission_status_date": sub.get("submission_status_date"),
                    "dt": dt,
                    "year": int(dt.year),
                    "application_kind": kind,
                    "submission_type": sub.get("submission_type"),
                }
            )
    return pd.DataFrame(rows)


def _filter_approved_kind(df: pd.DataFrame, kind_mode: str) -> pd.DataFrame:
    if df.empty:
        return df
    if kind_mode == "nda":
        return df[df["application_kind"] == "NDA"]
    if kind_mode == "anda":
        return df[df["application_kind"] == "ANDA"]
    if kind_mode == "bla":
        return df[df["application_kind"] == "BLA"]
    return df


def _filter_year(df: pd.DataFrame, y0: int, y1: int) -> pd.DataFrame:
    if df.empty:
        return df
    return df[(df["year"] >= y0) & (df["year"] <= y1)]


def _fig_html(fig: go.Figure, height_px: int = 460) -> ui.HTML:
    """Embed Plotly: fixed height, full width of parent (autosize + responsive)."""
    fig.update_layout(
        height=height_px,
        autosize=True,
        width=None,
    )
    html = pio.to_html(
        fig,
        include_plotlyjs="cdn",
        full_html=False,
        config={
            "displayModeBar": True,
            "displaylogo": False,
            "responsive": True,
        },
        default_width="100%",
        default_height=f"{height_px}px",
    )
    return ui.HTML(html)


def _empty_fig(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14, color=PALETTE_DARK),
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(t=40, b=40),
    )
    return fig


def _chart_layout(fig: go.Figure, title: str) -> None:
    """Plot styling for light backgrounds (white cards)."""
    fig.update_layout(
        title=dict(text=title, font=dict(color=PALETTE_DARK, size=18)),
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        font=dict(color=PALETTE_DARK),
        xaxis=dict(
            gridcolor="rgba(26,26,26,0.12)",
            tickfont=dict(color=PALETTE_DARK),
        ),
        yaxis=dict(
            gridcolor="rgba(26,26,26,0.12)",
            tickfont=dict(color=PALETTE_DARK),
        ),
        margin=dict(t=48, b=80),
        legend=dict(font=dict(color=PALETTE_DARK)),
    )


ui.page_opts(
    title="Drugs@FDA — Approved Drugs Dashboard",
    fillable=False,
    # `fluid=False` makes `page_navbar()` wrap the sidebar + tab content in a centered
    # Bootstrap `.container` (instead of edge-to-edge `.container-fluid`).
    fluid=False,
    theme=shinyswatch.theme.flatly,
)

# Streamlit-inspired polish: airy layout, blue accent, soft cards, info/tip callouts, larger type.
ui.tags.style(
    """
    :root {
        --app-accent: #2563eb;
        --app-accent-dark: #1d4ed8;
        --app-accent-soft: #eff6ff;
        --app-page-bg: #f0f4f8;
        --app-text: #1e293b;
        --app-muted: #64748b;
        --app-success-soft: #ecfdf5;
        --app-success-border: #059669;
    }
    body { background-color: var(--app-page-bg) !important; }
    .bslib-page-main {
        overflow-x: auto;
        overflow-y: visible;
        padding: 1.1rem 1.25rem 2.75rem;
        background-color: var(--app-page-bg) !important;
        font-size: 1.125rem;
        color: var(--app-text);
        line-height: 1.55;
    }
    .bslib-page-main h1 {
        font-size: clamp(1.9rem, 2.8vw, 2.45rem) !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
        color: var(--app-text) !important;
    }
    .bslib-page-main h4, .bslib-page-main h5 { font-size: 1.25rem; font-weight: 600; color: var(--app-text); }
    .bslib-page-main .small,
    .bslib-page-main .text-secondary.small {
        font-size: 1.05rem !important;
    }
    .bslib-page-main > .tab-content {
        padding: 1.25rem 1.2rem 2rem;
    }
    .bslib-page-navbar > .container {
        max-width: 1480px;
    }
    .bslib-sidebar, .sidebar {
        background: linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%) !important;
        border-right: 1px solid rgba(37, 99, 235, 0.08) !important;
        font-size: 1.05rem;
    }
    .sidebar-panel-title {
        font-size: 1.38rem !important;
        font-weight: 800 !important;
        color: var(--app-text) !important;
        letter-spacing: -0.02em;
        margin-bottom: 0.35rem !important;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(37, 99, 235, 0.2);
    }
    .sidebar-section-title {
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        color: var(--app-text) !important;
        margin-top: 0.35rem;
        letter-spacing: -0.01em;
    }
    .bslib-card {
        overflow: visible !important;
        min-height: unset !important;
        background-color: #ffffff !important;
        border: 1px solid rgba(15, 23, 42, 0.08) !important;
        border-radius: 14px !important;
        box-shadow: 0 2px 14px rgba(15, 76, 129, 0.07);
    }
    .bslib-card .card-body { overflow: visible !important; padding: 1.15rem 1.35rem !important; }
    .bslib-card .card-header {
        font-size: 1.22rem !important;
        font-weight: 650 !important;
        color: var(--app-text) !important;
        border-bottom: 1px solid rgba(37, 99, 235, 0.12) !important;
        padding: 0.95rem 1.25rem !important;
        border-radius: 14px 14px 0 0 !important;
        background: linear-gradient(90deg, var(--app-accent-soft) 0%, #ffffff 42%) !important;
    }
    .bslib-card .plotly-graph-div {
        width: 100% !important;
        max-width: 100% !important;
    }
    .nav-underline .nav-link {
        color: var(--app-muted) !important;
        font-weight: 600;
        font-size: 1.12rem;
        padding: 0.55rem 1rem;
    }
    .nav-underline .nav-link.active {
        color: var(--app-accent-dark) !important;
        border-bottom-color: var(--app-accent) !important;
        border-bottom-width: 3px !important;
        font-weight: 700;
    }
    .navbar .nav.nav-underline {
        width: 100%;
        border-bottom: 1px solid rgba(30, 41, 59, 0.12);
        padding-bottom: 0.2rem;
        margin-bottom: 0;
    }
    .btn-primary {
        background-color: var(--app-accent) !important;
        border-color: var(--app-accent-dark) !important;
        font-weight: 600;
        font-size: 1.05rem;
        padding: 0.5rem 1rem;
        border-radius: 10px;
    }
    .btn-primary:hover {
        background-color: var(--app-accent-dark) !important;
        border-color: #1e40af !important;
    }
    .btn-outline-secondary {
        border-radius: 10px;
        font-weight: 600;
        font-size: 1.05rem;
        padding: 0.45rem 1rem;
    }
    .app-hero {
        background: linear-gradient(125deg, var(--app-accent-soft) 0%, #ffffff 52%, #f8fafc 100%);
        border: 1px solid rgba(37, 99, 235, 0.18);
        border-radius: 16px;
        padding: 1.4rem 1.65rem 1.35rem;
        margin-bottom: 1.85rem;
        box-shadow: 0 2px 16px rgba(37, 99, 235, 0.08);
    }
    .app-hero-lead {
        font-size: 1.22rem;
        font-weight: 500;
        color: var(--app-text);
        line-height: 1.5;
        margin-bottom: 0.65rem;
    }
    .app-feature-list {
        list-style: none;
        padding-left: 0;
        margin: 0;
    }
    .app-feature-list li {
        position: relative;
        padding: 0.45rem 0 0.45rem 2rem;
        font-size: 1.08rem;
        color: var(--app-text);
        border-bottom: 1px solid rgba(148, 163, 184, 0.25);
    }
    .app-feature-list li:last-child { border-bottom: none; }
    .app-feature-list li::before {
        content: "✓";
        position: absolute;
        left: 0.15rem;
        top: 0.42rem;
        color: var(--app-accent);
        font-weight: 800;
        font-size: 1.1rem;
    }
    .app-callout-info {
        background: var(--app-accent-soft);
        border-left: 4px solid var(--app-accent);
        padding: 1rem 1.3rem;
        border-radius: 0 12px 12px 0;
        margin: 0.5rem 0 0;
        color: var(--app-text);
        font-size: 1.06rem;
        line-height: 1.55;
    }
    .app-callout-tip {
        background: var(--app-success-soft);
        border-left: 4px solid var(--app-success-border);
        padding: 0.95rem 1.25rem;
        border-radius: 0 12px 12px 0;
        margin-top: 1rem;
        color: var(--app-text);
        font-size: 1.05rem;
        line-height: 1.55;
    }
    .app-tab-info {
        background: linear-gradient(118deg, #fffbeb 0%, #ffffff 55%, #f8fafc 100%);
        border: 1px solid rgba(245, 158, 11, 0.28);
        border-left: 5px solid #f59e0b;
        border-radius: 14px;
        padding: 1.05rem 1.35rem 1.1rem;
        margin-bottom: 1.35rem;
        color: var(--app-text);
        font-size: 1.08rem;
        line-height: 1.58;
        box-shadow: 0 1px 10px rgba(245, 158, 11, 0.07);
    }
    .app-tab-info h4 {
        font-size: 1.28rem !important;
        font-weight: 700 !important;
        color: var(--app-text) !important;
        margin-bottom: 0.55rem !important;
    }
    .app-tab-info p:last-child { margin-bottom: 0; }
    .app-section-lead {
        color: var(--app-muted) !important;
        font-size: 1.08rem !important;
        line-height: 1.55;
        margin-bottom: 0.75rem;
    }
    .app-subtitle {
        font-size: 1.12rem !important;
        color: var(--app-muted) !important;
        font-weight: 500;
    }
    .app-ai-output {
        font-size: 1.15rem !important;
        line-height: 1.65 !important;
        white-space: normal;
        word-break: break-word;
        background: #ffffff !important;
        border: 1px solid rgba(37, 99, 235, 0.2) !important;
        border-left: 5px solid var(--app-accent) !important;
        border-radius: 12px !important;
        padding: 1.25rem 1.4rem !important;
        color: var(--app-text) !important;
        margin: 0 !important;
    }
    .app-ai-output p { margin-bottom: 0.65rem; }
    .app-ai-output p:last-child { margin-bottom: 0; }
    .app-ai-output ul, .app-ai-output ol {
        margin-bottom: 0.65rem;
        padding-left: 1.4rem;
    }
    .app-ai-output li { margin-bottom: 0.3rem; }
    .app-ai-output h1, .app-ai-output h2, .app-ai-output h3, .app-ai-output h4 {
        font-size: 1.2rem;
        font-weight: 700;
        margin-top: 0.75rem;
        margin-bottom: 0.5rem;
        color: var(--app-text);
    }
    .app-ai-output h1:first-child, .app-ai-output h2:first-child, .app-ai-output h3:first-child {
        margin-top: 0;
    }
    .app-ai-output code {
        font-size: 0.95em;
        background: var(--app-accent-soft);
        padding: 0.1rem 0.35rem;
        border-radius: 4px;
    }
    .app-ai-output pre {
        white-space: pre-wrap;
        font-size: 0.98rem;
        background: #f1f5f9;
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 8px;
        padding: 0.75rem 1rem;
    }
    .app-about-block {
        background: #ffffff;
        border-radius: 14px;
        border: 1px solid rgba(15, 23, 42, 0.08);
        padding: 1.25rem 1.45rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 10px rgba(15, 23, 42, 0.05);
    }
    .app-about-block h4 {
        font-size: 1.35rem;
        font-weight: 700;
        color: var(--app-accent-dark);
        margin-bottom: 0.75rem;
    }
    .dashboard-approvals-chart {
        width: 100% !important;
        max-width: 100% !important;
        min-height: 640px;
        box-sizing: border-box;
    }
    .dashboard-approvals-chart > div {
        width: 100% !important;
        max-width: 100% !important;
    }
    .dashboard-approvals-chart .plotly-graph-div {
        width: 100% !important;
        max-width: none !important;
    }
    .dashboard-approvals-chart .svg-container {
        width: 100% !important;
    }
    .dashboard-approvals-chart .main-svg {
        width: 100% !important;
    }
    /* Dashboard top row: equal-height columns; KPIs stacked to mirror pie card height */
    .dashboard-top-row {
        align-items: stretch !important;
    }
    .dashboard-top-row > * {
        min-height: 100%;
    }
    .dashboard-approved-drugs-card.card,
    .dashboard-approval-types-card.card {
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    .dashboard-approved-drugs-card .card-body,
    .dashboard-approval-types-card .card-body {
        display: flex;
        flex-direction: column;
        flex: 1 1 auto;
    }
    .foundation-kpi-stack {
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
        flex: 1 1 auto;
    }
    .foundation-kpi-stack .card {
        flex: 1 1 auto;
    }
    /* Drug info: pandas HTML tables — left-align headers and cells (theme defaults often center) */
    .drug-info-stack .drug-info-table-wrap table,
    .drug-info-stack .drug-info-pandas-table {
        text-align: left !important;
    }
    .drug-info-stack .drug-info-table-wrap th,
    .drug-info-stack .drug-info-table-wrap td {
        text-align: left !important;
        vertical-align: top;
        font-size: 1.05rem;
        padding: 0.55rem 0.65rem;
    }
    .drug-info-stack .drug-info-table-wrap th {
        background: var(--app-accent-soft);
        color: var(--app-accent-dark);
        font-weight: 650;
    }
    .form-label, .shiny-input-container > label {
        font-weight: 600 !important;
        color: var(--app-text) !important;
        font-size: 1.05rem !important;
        margin-bottom: 0.35rem;
    }
    .app-placeholder-hint {
        font-size: 1.05rem;
        color: var(--app-muted);
    }
    """
)

ui.h1("💊 Drugs@FDA — Approved Drugs Dashboard", class_="text-center mb-3")
ui.div(
    ui.p(
        "A compact analytics workspace for approved-drug activity: filter a time window, "
        "compare approval types, inspect yearly momentum, and spotlight sponsors — then drill into a single application.",
        class_="app-hero-lead",
    ),
    ui.tags.ul(
        ui.tags.li("📊 Dashboard — KPIs, approval mix (NDA / ANDA / BLA), yearly AP trend with optional rolling average, top sponsors"),
        ui.tags.li("💊 Drug info — structured identity, approval history, active ingredients, and product rows for one application"),
        ui.tags.li("🤖 AI summaries — optional narrative for the current chart context or the selected application (Ollama by default)"),
        class_="app-feature-list",
    ),
    ui.div(
        ui.markdown(
            "**What you get:** reproducible charts tied to the same filtered AP rows, plus transparent links to openFDA fields."
        ),
        class_="app-callout-info",
    ),
    ui.div(
        ui.markdown(
            "**👉 Start here**\n\n"
            "1. Open the **⚙️ Control Panel** in the sidebar.\n"
            "2. Click **Refresh from openFDA** to load the sample from openFDA.\n"
            "3. Use the **📊 Dashboard**, **💊 Drug Info**, and **ℹ️ About** tabs to explore charts, a single application, and documentation."
        ),
        class_="app-callout-tip",
    ),
    class_="app-hero",
)

with ui.sidebar(width=380, gap="0.85rem", padding="1rem"):
    ui.h4("⚙️ Control Panel", class_="sidebar-panel-title")
    ui.h5("📥 Data", class_="sidebar-section-title")
    ui.input_slider(
        "fetch_limit",
        "Applications To Fetch (Max 1000)",
        min=50,
        max=1000,
        value=300,
        step=50,
    )
    ui.input_action_button("refresh", "Refresh From OpenFDA", class_="btn-primary w-100")
    ui.hr()
    ui.h5("🎛️ Filters (Charts)", class_="sidebar-section-title")
    ui.input_slider(
        "year_range",
        "Year Range",
        min=1990,
        max=2030,
        value=[2000, 2025],
        step=1,
    )
    ui.input_radio_buttons(
        "app_kind_filter",
        "Application Type",
        {
            "all": "All",
            "nda": "NDA Only (Brand)",
            "anda": "ANDA Only (Generic)",
            "bla": "BLA Only (Biologic)",
        },
        selected="all",
    )
    ui.input_checkbox(
        "normalize_sponsors",
        "Normalize Sponsor Names (Trim + Uppercase For Grouping)",
        value=False,
    )
    ui.input_radio_buttons(
        "sponsor_metric",
        "Top Sponsors Count",
        {
            "events": "Each AP Event",
            "distinct_apps": "Distinct Applications",
        },
        selected="events",
    )
    ui.input_radio_buttons(
        "top_n_sponsors",
        "Top Sponsors",
        {"10": "Top 10", "20": "Top 20"},
        selected="10",
        inline=True,
    )
    ui.hr()
    ui.h5("📉 Trend Chart", class_="sidebar-section-title")
    ui.input_checkbox("show_rolling_avg", "Show Rolling Average", value=True)
    ui.input_slider("roll_window", "Rolling Window (Years)", min=1, max=7, value=3, step=1)
    with ui.div(class_="app-placeholder-hint mt-2"):
        @render.text
        def fetch_status():
            st = drugs_state()
            if not st["ok"]:
                return f"Error: {st['error']}"
            n = len(st["records"])
            meta = st.get("meta") or {}
            total = meta.get("results", {}).get("total")
            extra = f" (openFDA total matching: {total})" if total is not None else ""
            return f"{n} applications loaded{extra}."


def _tab_info_section(markdown_body: str):
    """Short ‘💡 Info’ blurb at the top of each main tab."""
    return ui.div(
        ui.h4("💡 Info"),
        ui.markdown(markdown_body),
        class_="app-tab-info",
    )


with ui.navset_underline(id="main_tabs", selected="dashboard"):
    with ui.nav_panel("📊 Dashboard", value="dashboard"):
        _tab_info_section(
            "The **Dashboard** summarizes approved-drug activity from openFDA **Drugs@FDA** using only **AP** "
            "(approved) submissions with parseable dates. After you load data in the **Control Panel**, you get "
            "headline KPIs, an **approval-type** pie (NDA / ANDA / BLA), an **approvals-per-year** trend (with an optional rolling average), "
            "and a **top sponsors** bar chart—every visual uses the **same filtered rows** so the story stays consistent. "
            "Click **Explain chart trends** when you want an AI narrative grounded in those charts."
        )
        with ui.layout_columns(
            col_widths=[6, 6],
            fill=False,
            fillable=False,
            class_="dashboard-top-row g-3",
        ):
            with ui.card(full_screen=False, fill=False, class_="dashboard-approved-drugs-card"):
                ui.card_header("Approved Drugs — Foundation Dataset")
                ui.div(
                    ui.markdown(
                        "This card gives you a quick head count for the **same filtered slice** as the charts: sidebar **Year Range** "
                        "and **Application Type** are applied here too, so the totals move when you change those controls."
                    ),
                    class_="app-section-lead",
                )

                @render.ui
                def foundation_summary():
                    base = approved_ap_df()
                    df = filtered_approved_for_charts()
                    if base.empty:
                        return ui.p(
                            "No AP rows in sample — fetch data or widen the application pull.",
                            class_="text-warning",
                        )
                    if df.empty:
                        return ui.p(
                            "No AP rows match your sidebar filters — try widening the year range or choosing “All” application types.",
                            class_="text-warning",
                        )
                    n_events = len(df)
                    n_apps = df["application_number"].nunique()
                    y0, y1 = int(df["year"].min()), int(df["year"].max())
                    return ui.div(
                        shiny_ui.card(
                            shiny_ui.card_header("AP Approval Events"),
                            ui.p(str(n_events), class_="h3 mb-0", style=f"color: {PALETTE_RED};"),
                            ui.p("Rows with status AP", class_="app-placeholder-hint mb-0"),
                            fill=False,
                        ),
                        shiny_ui.card(
                            shiny_ui.card_header("Distinct Applications"),
                            ui.p(str(n_apps), class_="h3 mb-0", style=f"color: {PALETTE_GREEN};"),
                            ui.p("Unique application_number in AP rows", class_="app-placeholder-hint mb-0"),
                            fill=False,
                        ),
                        shiny_ui.card(
                            shiny_ui.card_header("Years Covered"),
                            ui.p(f"{y0} – {y1}", class_="h3 mb-0", style=f"color: {PALETTE_BLUE};"),
                            ui.p("From submission_status_date", class_="app-placeholder-hint mb-0"),
                            fill=False,
                        ),
                        class_="foundation-kpi-stack",
                    )

            with ui.card(full_screen=False, fill=False, class_="dashboard-approval-types-card"):
                ui.card_header("🧾 Approval Type — NDA Vs ANDA Vs BLA")
                ui.p(
                    "This pie chart answers a simple question: in your current filter, are approvals mostly brand-name drugs, generics, or biologics? "
                    "Each slice is a share of approval events so you can compare the mix at a glance.",
                    class_="app-section-lead",
                )

                @render.ui
                def plot_approval_kind_pie():
                    st = drugs_state()
                    if not st["ok"]:
                        return _fig_html(_empty_fig(st.get("error", "Error")), height_px=360)
                    df = filtered_approved_for_charts()
                    if df.empty:
                        return _fig_html(_empty_fig("No AP rows in range / filters."), height_px=360)
                    vc = df["application_kind"].value_counts()
                    labels = [KIND_LABELS.get(k, k) for k in vc.index]
                    fig = go.Figure(
                        data=[
                            go.Pie(
                                labels=labels,
                                values=vc.values,
                                hole=0.35,
                                marker=dict(
                                    colors=[
                                        CHART_FILL_COLORS[i % len(CHART_FILL_COLORS)]
                                        for i in range(len(vc))
                                    ],
                                    line=dict(color="#ffffff", width=1),
                                ),
                                textinfo="label+percent",
                                hovertemplate="%{label}<br>n=%{value}<br>%{percent}<extra></extra>",
                            )
                        ]
                    )
                    _chart_layout(fig, "Brand vs generic vs biologic (AP events)")
                    return _fig_html(fig, height_px=420)

        with ui.layout_columns(col_widths=[12], fill=False, fillable=False):
            with ui.card(full_screen=False, fill=False):
                ui.card_header("📈 Approvals Per Year (Trend)")
                ui.p(
                    "This line chart shows how many approvals happened in each calendar year so you can see ups, downs, and busy years. "
                    "You can turn on a rolling average (dashed line) to smooth year-to-year noise; the chart also highlights the peak year.",
                    class_="app-section-lead",
                )

                @render.ui
                def plot_approvals_per_year():
                    st = drugs_state()
                    if not st["ok"]:
                        return ui.div(
                            _fig_html(_empty_fig(st.get("error", "Error")), height_px=400),
                            class_="dashboard-approvals-chart",
                        )
                    df = filtered_approved_for_charts()
                    if df.empty:
                        return ui.div(
                            _fig_html(_empty_fig("No AP rows in range / filters."), height_px=400),
                            class_="dashboard-approvals-chart",
                        )
                    yearly = df.groupby("year", as_index=False).size()
                    yearly = yearly.rename(columns={"size": "n"}).sort_values("year")
                    if yearly.empty:
                        return ui.div(
                            _fig_html(_empty_fig("Nothing to plot."), height_px=400),
                            class_="dashboard-approvals-chart",
                        )
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=yearly["year"],
                            y=yearly["n"],
                            mode="lines+markers",
                            name="AP approvals",
                            line=dict(color=PALETTE_RED, width=2),
                            marker=dict(size=8),
                        )
                    )
                    if input.show_rolling_avg() and len(yearly) >= 1:
                        w = int(input.roll_window())
                        roll = yearly["n"].rolling(window=w, min_periods=1).mean()
                        fig.add_trace(
                            go.Scatter(
                                x=yearly["year"],
                                y=roll,
                                mode="lines",
                                name=f"{w}-year rolling avg",
                                line=dict(color="#888888", dash="dash"),
                            )
                        )
                    imax = yearly["n"].idxmax()
                    peak_y = int(yearly.loc[imax, "year"])
                    peak_n = int(yearly.loc[imax, "n"])
                    fig.add_annotation(
                        x=peak_y,
                        y=peak_n,
                        text=f"Peak: {peak_n} ({peak_y})",
                        showarrow=True,
                        arrowhead=2,
                        ax=0,
                        ay=-40,
                        font=dict(color=PALETTE_DARK, size=11),
                    )
                    _chart_layout(fig, "FDA AP approvals per year")
                    fig.update_layout(
                        xaxis_title="Year",
                        yaxis_title="Count",
                        margin=dict(l=56, r=32, t=56, b=72),
                    )
                    return ui.div(
                        _fig_html(fig, height_px=640),
                        class_="dashboard-approvals-chart",
                    )

        with ui.layout_columns(col_widths=[12], fill=False, fillable=False):
            with ui.card(full_screen=False, fill=False):
                ui.card_header("🏢 Top Sponsors")
                ui.p(
                    "This bar chart ranks the companies that show up most often in your filtered data so you can see who is driving the activity. "
                    "You choose whether each bar counts every approval event or one row per drug application, and whether sponsor names are grouped more aggressively.",
                    class_="app-section-lead",
                )

                @render.ui
                def plot_top_sponsors():
                    st = drugs_state()
                    if not st["ok"]:
                        return _fig_html(_empty_fig(st.get("error", "Error")))
                    df = filtered_approved_for_charts()
                    if df.empty:
                        return _fig_html(_empty_fig("No AP rows in range / filters."))
                    norm = input.normalize_sponsors()
                    df = df.copy()
                    df["_skey"] = df["sponsor_name"].map(lambda x: _sponsor_key(x, norm))
                    first_label = df.groupby("_skey", as_index=False)["sponsor_name"].first()
                    metric = input.sponsor_metric()
                    top_n = int(input.top_n_sponsors())
                    if metric == "distinct_apps":
                        sub = df.drop_duplicates(["_skey", "application_number"])
                        counts = sub.groupby("_skey").size()
                    else:
                        counts = df.groupby("_skey").size()
                    total = counts.sum()
                    top = counts.sort_values(ascending=False).head(top_n).iloc[::-1]
                    if top.empty:
                        return _fig_html(_empty_fig("No sponsor counts."))
                    labels = []
                    hover = []
                    for k in top.index:
                        lab = first_label.loc[first_label["_skey"] == k, "sponsor_name"].iloc[0]
                        labels.append(str(lab)[:60])
                        pct = 100.0 * float(top[k]) / float(total) if total else 0.0
                        hover.append(f"{lab}<br>Count: {int(top[k])}<br>Share: {pct:.1f}%")
                    fig = go.Figure(
                        go.Bar(
                            x=top.values,
                            y=labels,
                            orientation="h",
                            marker_color=PALETTE_GREEN,
                            text=[f"{int(v)} ({100*v/total:.1f}%)" for v in top.values] if total else [],
                            textposition="auto",
                            hovertext=hover,
                            hoverinfo="text",
                        )
                    )
                    _chart_layout(fig, "Top sponsors (filtered period & type)")
                    fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="Count")
                    n_bars = len(top)
                    h = max(420, 44 * n_bars + 120)
                    return _fig_html(fig, height_px=min(h, 900))

        with ui.layout_columns(col_widths=[12], fill=False, fillable=False):
            with ui.card(full_screen=False, fill=False):
                ui.card_header("📊 Chart Trends — AI Summary")
                ui.p(
                    "Click the button for a short plain-English recap of what the charts already show: the brand vs generic vs biologic mix, how approvals vary by year, and which sponsors dominate. "
                    "Power users can turn on a richer multi-step AI path via environment variables—see **About** for what that does.",
                    class_="app-section-lead",
                )
                ui.input_action_button(
                    "dashboard_chart_ai_btn",
                    "Explain Chart Trends",
                    class_="btn-outline-secondary mb-3",
                )

                @render.ui
                def dashboard_chart_ai_panel():
                    if input.dashboard_chart_ai_btn() == 0:
                        return ui.p(
                            'Click “Explain Chart Trends” after reviewing the charts above.',
                            class_="app-placeholder-hint mb-0",
                        )
                    df = filtered_approved_for_charts()
                    ctx = aggregate_full_dashboard_context(
                        df,
                        sponsor_normalize=bool(input.normalize_sponsors()),
                        sponsor_metric=input.sponsor_metric(),
                        top_n_sponsors=int(input.top_n_sponsors()),
                        show_rolling_avg=bool(input.show_rolling_avg()),
                        roll_window=int(input.roll_window()),
                    )
                    txt = summarize_dashboard_charts(ctx, df=df)
                    return _ai_markdown_output(txt)

    with ui.nav_panel("💊 Drug Info", value="drugs"):
        _tab_info_section(
            "**Drug info** is where you inspect **one application** end-to-end after a sidebar **Refresh**. "
            "Pick an **Application** from the dropdown to load identity (brand / generic), sponsor, latest **AP** approval date, "
            "application type, marketing status, **active ingredients**, and **product** rows straight from the API payload. "
            "Use **Generate AI summary** for a short natural-language recap of that same structured context."
        )
        with ui.layout_columns(col_widths=[12], fill=False, fillable=False):
            with ui.card(full_screen=False, fill=False):
                ui.card_header("Select An Application")
                ui.div(
                    ui.markdown(
                        "Choose one drug application from your latest **Refresh**; this filter controls everything below "
                        "(structured cards and the optional AI summary)."
                    ),
                    class_="app-section-lead mb-2",
                )
                ui.input_select(
                    "selected_app",
                    "",
                    choices={"": "Load Data Using Refresh"},
                    selected="",
                )

            with ui.card(full_screen=False, fill=False):
                ui.card_header("🤖 AI Summary — Selected Application")
                ui.div(
                    ui.markdown(
                        "Short narrative from openFDA Drugs@FDA fields for the application chosen above. "
                        "Uses **Ollama** by default, or **OpenAI** when `OPENAI_API_KEY` is set (see `ai_drug.py`). "
                        "Each run uses a **draft plus validator** pass against the same application JSON; you see the reviewed narrative only.",
                    ),
                    class_="app-section-lead",
                )
                ui.input_action_button(
                    "ai_drug_summary_btn",
                    "Generate AI Summary",
                    class_="btn-outline-secondary mb-3",
                )

                @render.ui
                def drug_ai_summary_panel():
                    if input.ai_drug_summary_btn() == 0:
                        return ui.p(
                            'Click “Generate AI Summary” (may take a few seconds).',
                            class_="app-placeholder-hint mb-0",
                        )
                    st = drugs_state()
                    sel = input.selected_app()
                    if not st["ok"]:
                        return ui.p("Load data with Refresh first.", class_="text-warning app-placeholder-hint mb-0")
                    if not sel:
                        return ui.p(
                            "Select an application in the Select An Application card above.",
                            class_="app-placeholder-hint mb-0",
                        )
                    rec = None
                    for r in st["records"]:
                        if str(r.get("application_number")) == str(sel):
                            rec = r
                            break
                    if rec is None:
                        return ui.p(
                            "Application not found in the loaded sample.",
                            class_="text-warning app-placeholder-hint mb-0",
                        )
                    txt = summarize_drug_application(rec)
                    return _ai_markdown_output(txt)

            @render.ui
            def drug_info_panel():
                st = drugs_state()
                sel = input.selected_app()
                if not st["ok"]:
                    return ui.p("Load data with Refresh first.", class_="text-warning app-placeholder-hint")
                if not sel:
                    return ui.p(
                        "Select an application in the Select An Application card above.",
                        class_="app-placeholder-hint",
                    )
                rec = None
                for r in st["records"]:
                    if str(r.get("application_number")) == str(sel):
                        rec = r
                        break
                if rec is None:
                    return ui.p(
                        "Application not found in the loaded sample.",
                        class_="text-warning app-placeholder-hint",
                    )

                products = rec.get("products") or []
                subs = rec.get("submissions") or []
                app_no = rec.get("application_number")
                sponsor = rec.get("sponsor_name") or "—"
                kind = _classify_application_kind(app_no)
                kind_label = KIND_LABELS.get(kind, KIND_LABELS["Other"])

                headline = _identity_headline(products)
                if products:
                    gn = (products[0].get("generic_name") or "").strip()
                    bn = (products[0].get("brand_name") or "").strip()
                else:
                    gn, bn = "", ""

                latest_ap = _latest_ap_submission_date(subs) or "—"
                n_ap = _count_ap_submissions(subs)
                mstat = _collect_marketing_statuses(products)

                latest_sub = _latest_submission_by_date(subs)
                if latest_sub:
                    sub_status = str(latest_sub.get("submission_status") or "—")
                    sub_date = _format_display_date(latest_sub.get("submission_status_date"))
                else:
                    sub_status, sub_date = "—", "—"

                ing_rows = _flatten_active_ingredients(products)
                if ing_rows:
                    ing_df = pd.DataFrame(ing_rows).rename(
                        columns={
                            "product_brand": "Product (brand)",
                            "name": "Ingredient",
                            "strength": "Strength",
                        }
                    )
                else:
                    ing_df = pd.DataFrame(columns=["Product (brand)", "Ingredient", "Strength"])

                detail_rows = []
                for p in products:
                    detail_rows.append(
                        {
                            "Brand name": p.get("brand_name") or "—",
                            "Generic name": p.get("generic_name") or "—",
                            "Classification": kind_label,
                            "Reference drug": p.get("reference_drug")
                            if p.get("reference_drug") not in (None, "")
                            else "—",
                            "Marketing status": p.get("marketing_status") or "—",
                        }
                    )
                detail_df = pd.DataFrame(detail_rows)

                identity_card = shiny_ui.card(
                    shiny_ui.card_header("💊 Drug Identity"),
                    ui.h3(headline, class_="mb-3"),
                    ui.p(ui.strong("Generic: "), gn or "—"),
                    ui.p(ui.strong("Application: "), str(app_no) if app_no is not None else "—"),
                    ui.p(ui.strong("Sponsor: "), sponsor),
                    fill=False,
                )

                approval_card = shiny_ui.card(
                    shiny_ui.card_header("📊 Approval Summary"),
                    shiny_ui.layout_columns(
                        shiny_ui.card(
                            shiny_ui.card_header("Approval Date (Latest AP)"),
                            ui.p(str(latest_ap), class_="h4 mb-0", style=f"color: {PALETTE_RED};"),
                            ui.p("submissions.submission_status_date (AP only)", class_="app-placeholder-hint mb-0"),
                            fill=False,
                        ),
                        shiny_ui.card(
                            shiny_ui.card_header("Application Type"),
                            ui.p(kind_label, class_="h4 mb-0", style=f"color: {PALETTE_GREEN};"),
                            ui.p(
                                "NDA (brand) / ANDA (generic) / BLA from application_number",
                                class_="app-placeholder-hint mb-0",
                            ),
                            fill=False,
                        ),
                        shiny_ui.card(
                            shiny_ui.card_header("# Of Approvals"),
                            ui.p(str(n_ap), class_="h4 mb-0", style=f"color: {PALETTE_BLUE};"),
                            ui.p(
                                "Count of submissions with status AP (incl. supplements)",
                                class_="app-placeholder-hint mb-0",
                            ),
                            fill=False,
                        ),
                        shiny_ui.card(
                            shiny_ui.card_header("Current Marketing Status"),
                            ui.p(mstat, class_="h4 mb-0", style=f"color: {PALETTE_DARK};"),
                            ui.p("products.marketing_status", class_="app-placeholder-hint mb-0"),
                            fill=False,
                        ),
                        col_widths=[3, 3, 3, 3],
                        fill=False,
                        fillable=False,
                        class_="g-2",
                    ),
                    ui.p(
                        ui.strong("Latest Submission (Any Type): "),
                        f"{sub_status} — {sub_date}",
                        class_="app-placeholder-hint mt-3 mb-0",
                    ),
                    fill=False,
                )

                ingredients_card = shiny_ui.card(
                    shiny_ui.card_header("🧪 Active Ingredients"),
                    _dataframe_table_html(
                        ing_df,
                        empty_msg="No active ingredient rows in the API payload for these products.",
                    ),
                    fill=False,
                )

                details_card = shiny_ui.card(
                    shiny_ui.card_header("📦 Product Details"),
                    _dataframe_table_html(
                        detail_df,
                        empty_msg="No product rows for this application.",
                    ),
                    fill=False,
                )

                return ui.div(
                    identity_card,
                    approval_card,
                    ingredients_card,
                    details_card,
                    class_="drug-info-stack",
                )

    with ui.nav_panel("ℹ️ About", value="about"):
        _tab_info_section(
            "This **About** area explains **why the app exists**, **who it serves**, and **how to use the controls** safely. "
            "The goal is to help **analysts, program managers, and decision-makers** turn dense FDA application tables into "
            "a defensible, shareable read on approval mix, timing, and sponsor concentration—without manual spreadsheet prep. "
            "Stakeholders get clearer alignment on scope (AP-only foundation, optional AI), limits (rate limits, not legal advice), "
            "and configuration (openFDA + optional Ollama / OpenAI) so they trust what they are looking at."
        )
        with ui.div(class_="app-about-block"):
            ui.h4("⚙️ Control Panel — Using The Sidebar")
            ui.markdown(
                "Everything in the **⚙️ Control Panel** (left sidebar) loads data or changes how the **Dashboard** "
                "charts and AI summaries interpret the same filtered **AP** (approved) rows.\n\n"
                "**📥 Data**\n"
                "- **Applications to fetch** — How many drug applications openFDA returns in one request (50–1000). "
                "Larger samples cover more history but are slower to load.\n"
                "- **Refresh from openFDA** — Re-fetches the dataset; all charts and the Drug info list rebuild from this pull.\n\n"
                "**🎛️ Filters (charts)** — Applied after the full AP table is built; they affect **every dashboard chart**, "
                "the **Approved Drugs — Foundation Dataset** KPI card, and the **Explain chart trends** AI context.\n"
                "- **Year range** — Keeps only AP events whose `submission_status_date` falls in those calendar years.\n"
                "- **Application type** — **All** shows NDA + ANDA + BLA + Other, or restrict to **NDA only**, **ANDA only**, or **BLA only** (inferred from `application_number` prefix).\n"
                "- **Normalize sponsor names** — When on, sponsor labels are trimmed and uppercased before grouping "
                "(useful when the same sponsor appears with inconsistent spelling).\n"
                "- **Top sponsors count** — **Each AP event** counts every approval row per sponsor; **Distinct applications** "
                "counts each `application_number` once per sponsor (matches how the horizontal bar chart is computed).\n"
                "- **Top sponsors** — Show **Top 10** or **Top 20** sponsors on the bar chart (and in the aggregated AI context).\n\n"
                "**📉 Trend chart**\n"
                "- **Show rolling average** — Adds the dashed rolling mean on the approvals-per-year chart.\n"
                "- **Rolling window (years)** — Window width for that average (1–7 years).\n\n"
                "**Status line** — Under the trend controls, a short line shows how many applications loaded and, when available, "
                "openFDA’s reported total for the query.\n\n"
                "**Note:** The **Drug info** application dropdown lists every application from the last **Refresh** "
                "(up to your fetch limit). **Year range** and **application type** filters apply to **Dashboard** charts "
                "and chart AI only, not to which IDs appear in that dropdown."
            )
        with ui.div(class_="app-about-block"):
            ui.h4("🤝 Stakeholder Alignment")
            ui.markdown(
                "**Who this is for:** analysts and decision-makers who need a fast, repeatable read on approved "
                "Drugs@FDA activity—approval mix, timing, sponsor concentration—and a structured drill-down for a "
                "single application.\n\n"
                "**What decision it supports:** “What changed in this window?” (trend + composition) and "
                "“What is this application?” (identity + approvals + ingredients + products).\n\n"
                "**Does it solve a real problem for your stakeholders?** It does when the bottleneck is turning FDA "
                "tabular records into an explainable dashboard narrative without manual spreadsheet work."
            )
        with ui.div(class_="app-about-block"):
            ui.h4("📌 Scope & Limitations")
            ui.markdown(
                "- **Foundation analytics** use **AP** submissions with parseable `submission_status_date` only.\n"
                "- **AI outputs** are assistive summaries of the on-screen context; they are not regulatory determinations.\n"
                "- **Rate limits:** openFDA works without a key; `OPENFDA_API_KEY` improves limits."
            )
        with ui.div(class_="app-about-block"):
            ui.h4("Quality Control")
            ui.markdown(
                "This application treats AI text as **assistive reporting**, not a source of truth on its own. Quality control is built in through:\n\n"
                "**1. Grounding in real FDA data** — Summaries are always tied to **openFDA Drugs@FDA** payloads or chart metrics you already "
                "loaded in the app, so the model starts from the same facts you can inspect in the UI.\n\n"
                "**2. A second “validator” pass on AI wording** — For both **Explain chart trends** (when the fuller server mode is on) and "
                "**Drug info → AI summary**, the app runs a **draft-then-review** pattern: a first pass writes the narrative, then a **validator** "
                "step edits that draft against the **same structured context**, aiming to strip unsupported numbers, tighten contradictions, "
                "and keep tone appropriate for analysts and decision-makers—not clinical advice.\n\n"
                "**3. Extra transparency on chart AI** — When **Explain chart trends** runs in full server mode, the answer can end with a short "
                "**Quality control** note that says what ran and includes quick automated checks. **Drug info → AI summary** uses the same "
                "draft-and-validator flow but only shows the cleaned narrative in the card (no separate footer).\n\n"
                "Together, these steps exist so teams can **share summaries with more confidence**: aligned with on-screen data, reviewed, "
                "and clearly labeled as **non-regulatory** assistance."
            )
        with ui.div(class_="app-about-block"):
            ui.h4("ℹ️ Data & Configuration")
            ui.markdown(
                "This dashboard queries the openFDA **Drugs@FDA** API (application-level records). "
                "Data refreshes when you click **Refresh from openFDA**. "
                "Dashboard chart AI uses `agents_drug.py` (filtered AP metrics). Drug info AI uses `ai_drug.py` for an application summary "
                "with the same **draft + validator** pattern described under **Quality Control** above. Default LLM: **Ollama**; optional **OpenAI** via `OPENAI_API_KEY`. "
                "For Ollama Cloud (`OLLAMA_HOST=https://ollama.com`), set `OLLAMA_API_KEY`. "
                "Administrators can enable a **full** chart-AI mode with a server flag (`DASHBOARD_AI_ORCHESTRATOR=1`); that turns on "
                "extra data lookups, supporting notes, and detailed activity in server logs for troubleshooting."
            )
        with ui.div(class_="app-about-block"):
            ui.h4("📚 Further Information")
            ui.markdown(
                "- [openFDA **Drugs@FDA** API reference](https://open.fda.gov/apis/drug/drugsfda/) — fields, query syntax, "
                "and examples for `drug/drugsfda.json` (the endpoint this dashboard calls).\n"
                "- [openFDA API overview](https://open.fda.gov/apis/) — general usage, authentication, and rate limits.\n"
                "- [Ollama HTTP API](https://github.com/ollama/ollama/blob/main/docs/api.md) — `/api/chat`, `/api/generate`, "
                "models, and tool-calling behavior used by chart and drug summaries.\n"
                "- [Shiny for Python documentation](https://shiny.posit.co/py/) — reactive UI patterns used in `app_drug.py`.\n"
                "- [Course repository — `app/` source](https://github.com/joninguyen12/sysen5381-tool/tree/main/app) — "
                "this project’s scripts, prompts, and Posit Connect helpers."
            )


@reactive.calc
def drugs_state():
    input.refresh()
    lim = int(input.fetch_limit())
    try:
        payload = fetch_drugsfda(limit=lim)
        records = extract_results(payload)
        return {
            "ok": True,
            "records": records,
            "meta": payload.get("meta"),
            "error": None,
        }
    except Exception as e:
        return {"ok": False, "records": [], "meta": None, "error": str(e)}


@reactive.calc
def approved_ap_df() -> pd.DataFrame:
    """Clean foundation: AP-only rows with year and application kind."""
    st = drugs_state()
    if not st["ok"]:
        return pd.DataFrame()
    return _build_approved_submissions_df(st["records"])


@reactive.calc
def filtered_approved_for_charts() -> pd.DataFrame:
    """Apply year range + NDA/ANDA/BLA filter for visuals."""
    df = approved_ap_df()
    if df.empty:
        return df
    yr = input.year_range()
    y0, y1 = int(yr[0]), int(yr[1])
    df = _filter_year(df, y0, y1)
    mode = input.app_kind_filter()
    return _filter_approved_kind(df, mode)


@reactive.effect
def _sync_year_slider_to_data():
    df = approved_ap_df()
    if df.empty:
        return
    ymin, ymax = int(df["year"].min()), int(df["year"].max())
    if ymin >= ymax:
        ymax = ymin + 1
    ui.update_slider("year_range", min=ymin, max=ymax, value=[ymin, ymax])


@reactive.effect
def _sync_app_select():
    st = drugs_state()
    if not st["ok"] or not st["records"]:
        ui.update_select("selected_app", choices={"": "Load Data Using Refresh"}, selected="")
        return
    choices = {
        str(r["application_number"]): f"{r['application_number']} — {(r.get('sponsor_name') or 'Unknown')[:48]}"
        for r in st["records"]
        if r.get("application_number") is not None
    }
    if not choices:
        ui.update_select("selected_app", choices={"": "No Application IDs"}, selected="")
        return
    keys = list(choices.keys())
    with reactive.isolate():
        cur = input.selected_app()
    selected = cur if cur in choices else keys[0]
    ui.update_select("selected_app", choices=choices, selected=selected)
