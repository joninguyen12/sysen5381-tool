# api_drug.py
# Query the openFDA Drugs@FDA API in Python
# Documentation: https://open.fda.gov/apis/drug/drugsfda/

import json
import os

import env_load  # noqa: F401 — loads .env from app directory
import requests

BASE_URL = "https://api.fda.gov/drug/drugsfda.json"


def _params(limit: int, skip: int = 0) -> dict:
    """Query parameters for drugsfda.json (limit capped at openFDA max 1000)."""
    lim = max(1, min(int(limit), 1000))
    params: dict = {"limit": lim, "skip": max(0, int(skip))}
    key = os.getenv("OPENFDA_API_KEY")
    if key:
        params["api_key"] = key
    return params


def fetch_drugsfda(limit: int = 100, skip: int = 0, timeout: float = 60.0) -> dict:
    """GET drugsfda and return the parsed JSON object."""
    response = requests.get(BASE_URL, params=_params(limit, skip), timeout=timeout)
    response.raise_for_status()
    return response.json()


def extract_record(record: dict) -> dict:
    submissions_out = []
    for sub in record.get("submissions") or []:
        submissions_out.append(
            {
                "submission_status_date": sub.get("submission_status_date"),
                "submission_status": sub.get("submission_status"),
                "submission_type": sub.get("submission_type"),
            }
        )
    products_out = []
    for prod in record.get("products") or []:
        products_out.append(
            {
                "brand_name": prod.get("brand_name"),
                "generic_name": prod.get("generic_name"),
                "dosage_form": prod.get("dosage_form"),
                "route": prod.get("route"),
                "marketing_status": prod.get("marketing_status"),
                "reference_drug": prod.get("reference_drug"),
                "active_ingredients": prod.get("active_ingredients") or [],
            }
        )
    return {
        "sponsor_name": record.get("sponsor_name"),
        "application_number": record.get("application_number"),
        "submissions": submissions_out,
        "products": products_out,
    }


def extract_results(payload: dict) -> list[dict]:
    return [extract_record(r) for r in payload.get("results") or []]


if __name__ == "__main__":
    payload = fetch_drugsfda(limit=1)
    print(payload.get("meta", {}))
    extracted = extract_results(payload)
    out = {"meta": payload.get("meta"), "extracted_results": extracted}
    print(json.dumps(out, indent=2))
