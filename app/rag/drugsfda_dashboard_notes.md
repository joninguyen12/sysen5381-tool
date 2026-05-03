# Drugs@FDA dashboard notes (RAG knowledge base)

These notes are **not FDA regulatory guidance**. They are a compact, stakeholder-facing crib sheet
to help an analyst interpret **what the charts in this app can and cannot say** when the data is
sampled from openFDA and filtered to **AP** submissions with parseable dates.

## What “AP” means in this app

- **AP** is used here as a proxy for an **approval-related milestone** in the Drugs@FDA submissions list.
- The dashboard foundation layer intentionally keeps rows where `submission_status == "AP"` **and**
  `submission_status_date` parses cleanly. Rows that fail parsing are excluded from the **trend/sponsor**
  analytics even if they may still appear elsewhere in raw Drug info fields.

## Interpreting approval-type mix (NDA / ANDA / BLA)

- The app **infers** NDA vs ANDA vs BLA primarily from the **application number prefix** (NDA/ANDA/BLA).
- Treat “Other / unknown” as **data-shape limitations**, not a regulatory category.

## Interpreting “approvals per year”

- The per-year counts are counting **filtered AP events** in the loaded sample, not a guaranteed census
  of all FDA approvals nationwide.
- Peaks/dips can reflect **sampling**, **parsing**, **filtering**, and **API limits**—not only real-world changes.

## Interpreting “top sponsors”

- Sponsor bars can be shown as **event counts** or **distinct applications**, and names may be grouped
  using normalization (trim + uppercase). Changing the metric can reorder the leaderboard.

## Safe language for stakeholders

- Prefer language like: “**In this filtered openFDA sample**, …”.
- Avoid implying completeness unless you have explicitly widened fetch limits and validated coverage.
