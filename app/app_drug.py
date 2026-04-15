# Drugs@FDA Explorer — Shiny for Python
# Uses api_drug.py against openFDA. Light dashboard style with tabbed sections.
#
# Run: shiny run app_drug.py
# Deps: pip install shiny shinyswatch plotly pandas requests python-dotenv
# AI: ai_drug.py — simple application summary (Drug info). agents_drug.py — dashboard chart explanation.
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
        title=dict(text=title, font=dict(color=PALETTE_DARK, size=16)),
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
    title="Drugs@FDA — Approved drugs",
    fillable=False,
    full_width=True,
    theme=shinyswatch.theme.flatly,
)

ui.tags.style(
    """
    body { background-color: #f5f5f5 !important; }
    .bslib-page-main {
        overflow-x: auto;
        overflow-y: visible;
        padding: 0.75rem 1rem 2.5rem;
        background-color: #f5f5f5 !important;
    }
    .bslib-sidebar, .sidebar { background-color: #ececec !important; }
    .bslib-card {
        overflow: visible !important;
        min-height: unset !important;
        background-color: #ffffff !important;
        border: 1px solid #e3e3e3 !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .bslib-card .card-body { overflow: visible !important; }
    .bslib-card .plotly-graph-div {
        width: 100% !important;
        max-width: 100% !important;
    }
    .nav-underline .nav-link { color: #555555 !important; font-weight: 500; }
    .nav-underline .nav-link.active {
        color: #1a1a1a !important;
        border-bottom-color: #ff0000 !important;
        border-bottom-width: 3px !important;
        font-weight: 600;
    }
    .btn-primary {
        background-color: #ff0000 !important;
        border-color: #cc0000 !important;
    }
    .btn-primary:hover {
        background-color: #cc0000 !important;
        border-color: #990000 !important;
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
    }
    """
)

ui.h1("💊 Drugs@FDA — Approved drugs dashboard", class_="text-center mb-2")
ui.p(
    "openFDA Drugs@FDA • AP-only submissions, trends, sponsors, and product details",
    class_="text-center text-muted mb-3 small",
)

with ui.sidebar():
    ui.h5("📥 Data")
    ui.input_slider(
        "fetch_limit",
        "Applications to fetch (max 1000)",
        min=50,
        max=1000,
        value=300,
        step=50,
    )
    ui.input_action_button("refresh", "Refresh from openFDA", class_="btn-primary w-100")
    ui.hr()
    ui.h5("🎛️ Filters (charts)")
    ui.input_slider(
        "year_range",
        "Year range",
        min=1990,
        max=2030,
        value=[2000, 2025],
        step=1,
    )
    ui.input_radio_buttons(
        "app_kind_filter",
        "Application type",
        {
            "all": "All",
            "nda": "NDA only (brand)",
            "anda": "ANDA only (generic)",
            "bla": "BLA only (biologic)",
        },
        selected="all",
    )
    ui.input_checkbox("normalize_sponsors", "Normalize sponsor names (trim + uppercase for grouping)", value=False)
    ui.input_radio_buttons(
        "sponsor_metric",
        "Top sponsors count",
        {
            "events": "Each AP event",
            "distinct_apps": "Distinct applications",
        },
        selected="events",
    )
    ui.input_radio_buttons(
        "top_n_sponsors",
        "Top sponsors",
        {"10": "Top 10", "20": "Top 20"},
        selected="10",
        inline=True,
    )
    ui.hr()
    ui.h5("📉 Trend chart")
    ui.input_checkbox("show_rolling_avg", "Show rolling average", value=True)
    ui.input_slider("roll_window", "Rolling window (years)", min=1, max=7, value=3, step=1)
    with ui.div(class_="small text-muted mt-2"):
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


with ui.navset_underline(id="main_tabs", selected="dashboard"):
    with ui.nav_panel("📊 Dashboard", value="dashboard"):
        with ui.layout_columns(
            col_widths=[6, 6],
            fill=False,
            fillable=False,
            class_="dashboard-top-row g-3",
        ):
            with ui.card(full_screen=False, fill=False, class_="dashboard-approved-drugs-card"):
                ui.card_header("Approved drugs — foundation dataset")
                ui.p(
                    "Only AP (approved) submissions with parseable dates. Downstream charts start from this table; "
                    "trend charts use every approval event; sponsor totals can use distinct application_number instead.",
                    class_="text-secondary small",
                )

                @render.ui
                def foundation_summary():
                    df = approved_ap_df()
                    if df.empty:
                        return ui.p(
                            "No AP rows in sample — fetch data or widen the application pull.",
                            class_="text-warning",
                        )
                    n_events = len(df)
                    n_apps = df["application_number"].nunique()
                    y0, y1 = int(df["year"].min()), int(df["year"].max())
                    return ui.div(
                        shiny_ui.card(
                            shiny_ui.card_header("AP approval events"),
                            ui.p(str(n_events), class_="h4 mb-0", style=f"color: {PALETTE_RED};"),
                            ui.p("Rows with status AP", class_="small", style="color: #555555;"),
                            fill=False,
                        ),
                        shiny_ui.card(
                            shiny_ui.card_header("Distinct applications"),
                            ui.p(str(n_apps), class_="h4 mb-0", style=f"color: {PALETTE_GREEN};"),
                            ui.p("Unique application_number in AP rows", class_="small", style="color: #555555;"),
                            fill=False,
                        ),
                        shiny_ui.card(
                            shiny_ui.card_header("Years covered"),
                            ui.p(f"{y0} – {y1}", class_="h4 mb-0", style=f"color: {PALETTE_BLUE};"),
                            ui.p("From submission_status_date", class_="small", style="color: #555555;"),
                            fill=False,
                        ),
                        class_="foundation-kpi-stack",
                    )

            with ui.card(full_screen=False, fill=False, class_="dashboard-approval-types-card"):
                ui.card_header("🧾 Approval type — NDA vs ANDA vs BLA")
                ui.p(
                    "Parsed from application_number prefix (NDA/ANDA/BLA); unmatched → Other. "
                    "Shares use the same filtered AP rows as other charts.",
                    class_="text-secondary small",
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
                ui.card_header("📈 Approvals per year (trend)")
                ui.p(
                    "Annual count of AP submissions (every approval kept for trends). "
                    "Optional dashed line: rolling average; strongest year annotated.",
                    class_="text-secondary small",
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
                ui.card_header("🏢 Top sponsors")
                ui.p(
                    "Grouped by sponsor (optional name normalization). "
                    "Market share is % of filtered AP events or distinct apps, depending on metric.",
                    class_="text-secondary small",
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
                ui.card_header("📊 Chart trends — AI summary")
                ui.p(
                    "Summarizes the same filtered data shown in the three chart sections above: "
                    "🧾 approval type (NDA/ANDA/BLA), 📈 approvals per year (including peak and rolling average when enabled), "
                    "and 🏢 top sponsors (same metric and Top N as the bar chart). "
                    "2–3 sentences for a business audience.",
                    class_="text-secondary small",
                )
                ui.input_action_button(
                    "dashboard_chart_ai_btn",
                    "Explain chart trends",
                    class_="btn-outline-secondary btn-sm mb-2",
                )

                @render.ui
                def dashboard_chart_ai_panel():
                    if input.dashboard_chart_ai_btn() == 0:
                        return ui.p(
                            'Click “Explain chart trends” after reviewing the charts above.',
                            class_="text-muted small mb-0",
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
                    txt = summarize_dashboard_charts(ctx)
                    return ui.pre(
                        txt,
                        class_="small bg-light p-3 border rounded mb-0",
                        style="white-space: pre-wrap; word-break: break-word;",
                    )

    with ui.nav_panel("💊 Drug info", value="drugs"):
        with ui.layout_columns(col_widths=[12], fill=False, fillable=False):
            ui.p(
                "Structured view for the selected application (identity, approvals, ingredients, products).",
                class_="text-secondary small mb-2",
            )
            ui.input_select(
                "selected_app",
                "Application",
                choices={"": "Load data using Refresh"},
                selected="",
            )

            with ui.card(full_screen=False, fill=False):
                ui.card_header("🤖 AI summary — selected application")
                ui.p(
                    "Short narrative from openFDA Drugs@FDA fields for the application chosen above. "
                    "Uses Ollama by default, or OpenAI when OPENAI_API_KEY is set (see ai_drug.py).",
                    class_="text-secondary small",
                )
                ui.input_action_button(
                    "ai_drug_summary_btn",
                    "Generate AI summary",
                    class_="btn-outline-secondary btn-sm mb-2",
                )

                @render.ui
                def drug_ai_summary_panel():
                    if input.ai_drug_summary_btn() == 0:
                        return ui.p(
                            'Click “Generate AI summary” (may take a few seconds).',
                            class_="text-muted small mb-0",
                        )
                    st = drugs_state()
                    sel = input.selected_app()
                    if not st["ok"]:
                        return ui.p("Load data with Refresh first.", class_="text-warning small mb-0")
                    if not sel:
                        return ui.p("Select an application above.", class_="text-muted small mb-0")
                    rec = None
                    for r in st["records"]:
                        if str(r.get("application_number")) == str(sel):
                            rec = r
                            break
                    if rec is None:
                        return ui.p("Application not found in the loaded sample.", class_="text-warning small mb-0")
                    txt = summarize_drug_application(rec)
                    return ui.pre(
                        txt,
                        class_="small bg-light p-3 border rounded mb-0",
                        style="white-space: pre-wrap; word-break: break-word;",
                    )

            @render.ui
            def drug_info_panel():
                st = drugs_state()
                sel = input.selected_app()
                if not st["ok"]:
                    return ui.p("Load data with Refresh first.", class_="text-warning")
                if not sel:
                    return ui.p("Select an application above.", class_="text-muted")
                rec = None
                for r in st["records"]:
                    if str(r.get("application_number")) == str(sel):
                        rec = r
                        break
                if rec is None:
                    return ui.p("Application not found in the loaded sample.", class_="text-warning")

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
                    shiny_ui.card_header("💊 Drug identity"),
                    ui.h4(headline, class_="mb-3"),
                    ui.p(ui.strong("Generic: "), gn or "—"),
                    ui.p(ui.strong("Application: "), str(app_no) if app_no is not None else "—"),
                    ui.p(ui.strong("Sponsor: "), sponsor),
                    fill=False,
                )

                approval_card = shiny_ui.card(
                    shiny_ui.card_header("📊 Approval summary"),
                    shiny_ui.layout_columns(
                        shiny_ui.card(
                            shiny_ui.card_header("Approval date (latest AP)"),
                            ui.p(str(latest_ap), class_="h5 mb-0", style=f"color: {PALETTE_RED};"),
                            ui.p("submissions.submission_status_date (AP only)", class_="small text-muted"),
                            fill=False,
                        ),
                        shiny_ui.card(
                            shiny_ui.card_header("Application type"),
                            ui.p(kind_label, class_="h5 mb-0", style=f"color: {PALETTE_GREEN};"),
                            ui.p("NDA (brand) / ANDA (generic) / BLA from application_number", class_="small text-muted"),
                            fill=False,
                        ),
                        shiny_ui.card(
                            shiny_ui.card_header("# of approvals"),
                            ui.p(str(n_ap), class_="h5 mb-0", style=f"color: {PALETTE_BLUE};"),
                            ui.p("Count of submissions with status AP (incl. supplements)", class_="small text-muted"),
                            fill=False,
                        ),
                        shiny_ui.card(
                            shiny_ui.card_header("Current marketing status"),
                            ui.p(mstat, class_="h5 mb-0", style=f"color: {PALETTE_DARK};"),
                            ui.p("products.marketing_status", class_="small text-muted"),
                            fill=False,
                        ),
                        col_widths=[3, 3, 3, 3],
                        fill=False,
                        fillable=False,
                        class_="g-2",
                    ),
                    ui.p(
                        ui.strong("Latest submission (any type): "),
                        f"{sub_status} — {sub_date}",
                        class_="small text-secondary mt-3 mb-0",
                    ),
                    fill=False,
                )

                ingredients_card = shiny_ui.card(
                    shiny_ui.card_header("🧪 Active ingredients"),
                    _dataframe_table_html(
                        ing_df,
                        empty_msg="No active ingredient rows in the API payload for these products.",
                    ),
                    fill=False,
                )

                details_card = shiny_ui.card(
                    shiny_ui.card_header("📦 Product details"),
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
        ui.p(
            "This dashboard queries the openFDA Drugs@FDA API (application-level records). "
            "Only submissions with status AP and valid dates are used in the foundation layer. "
            "Data refreshes when you click Refresh from openFDA. "
            "Optional API key: set OPENFDA_API_KEY in a .env file for higher rate limits. "
            "Dashboard chart AI uses agents_drug.py (filtered AP metrics). Drug info AI uses ai_drug.py for a simple application summary. Default LLM: Ollama; optional OpenAI via OPENAI_API_KEY.",
            class_="text-secondary",
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
        ui.update_select("selected_app", choices={"": "Load data using Refresh"}, selected="")
        return
    choices = {
        str(r["application_number"]): f"{r['application_number']} — {(r.get('sponsor_name') or 'Unknown')[:48]}"
        for r in st["records"]
        if r.get("application_number") is not None
    }
    if not choices:
        ui.update_select("selected_app", choices={"": "No application IDs"}, selected="")
        return
    keys = list(choices.keys())
    with reactive.isolate():
        cur = input.selected_app()
    selected = cur if cur in choices else keys[0]
    ui.update_select("selected_app", choices=choices, selected=selected)
