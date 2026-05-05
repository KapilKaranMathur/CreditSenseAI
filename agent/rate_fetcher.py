import logging
from datetime import datetime
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

TREASURY_API_URL = (
    "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/"
    "v2/accounting/od/avg_interest_rates"
)

RELEVANT_SECURITIES = {
    "Treasury Bills",
    "Treasury Notes",
    "Treasury Bonds",
    "Treasury Inflation-Protected Securities (TIPS)",
    "Treasury Floating Rate Notes (FRN)",
}

REQUEST_TIMEOUT = 10


def fetch_treasury_rates() -> Dict:
    try:
        params = {"sort": "-record_date", "page[size]": 20, "format": "json"}
        response = requests.get(TREASURY_API_URL, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        records = response.json().get("data", [])
        if not records:
            return _error_result("No data returned from Treasury API")

        latest_date = records[0].get("record_date", "Unknown")
        rates = []
        for record in records:
            if record.get("record_date") != latest_date:
                continue
            security = record.get("security_desc", "")
            if security not in RELEVANT_SECURITIES:
                continue
            try:
                rate = float(record.get("avg_interest_rate_amt", 0))
            except (TypeError, ValueError):
                continue
            rates.append({
                "security": security,
                "rate": round(rate, 3),
                "type": record.get("security_type_desc", ""),
            })

        rates.sort(key=lambda x: x["rate"], reverse=True)
        logger.info("Fetched %d Treasury rates (date: %s)", len(rates), latest_date)

        return {
            "rates": rates,
            "record_date": latest_date,
            "fetched_at": datetime.now().isoformat(),
            "source": "US Treasury Fiscal Data API",
            "error": None,
        }

    except requests.exceptions.Timeout:
        return _error_result("API request timed out — rates unavailable")
    except requests.exceptions.ConnectionError:
        return _error_result("No internet connection — rates unavailable")
    except requests.exceptions.RequestException as e:
        return _error_result(f"API request failed: {e}")
    except Exception as e:
        return _error_result(f"Unexpected error: {e}")


def _error_result(message: str) -> Dict:
    return {
        "rates": [],
        "record_date": None,
        "fetched_at": datetime.now().isoformat(),
        "source": "US Treasury Fiscal Data API",
        "error": message,
    }


def compare_borrower_rate(borrower_rate: float, treasury_data: Dict) -> Dict:
    rates = treasury_data.get("rates", [])
    if not rates:
        return {
            "borrower_rate": borrower_rate,
            "treasury_avg": None,
            "spread": None,
            "spread_assessment": "Treasury rates unavailable for comparison",
            "comparisons": [],
            "record_date": treasury_data.get("record_date"),
        }

    rate_values = [r["rate"] for r in rates]
    treasury_avg = round(sum(rate_values) / len(rate_values), 3)
    spread = round(borrower_rate - treasury_avg, 3)

    if spread <= 2.0:
        assessment = "Competitive — borrower rate is close to Treasury benchmarks"
    elif spread <= 5.0:
        assessment = "Moderate — borrower rate has a typical risk premium over Treasury rates"
    elif spread <= 10.0:
        assessment = "Elevated — borrower rate is significantly above Treasury benchmarks"
    else:
        assessment = "Very High — borrower rate is far above Treasury benchmarks, suggesting high-risk pricing"

    comparisons = [
        {
            "security": r["security"],
            "treasury_rate": r["rate"],
            "spread": round(borrower_rate - r["rate"], 3),
            "direction": "above" if borrower_rate > r["rate"] else "below" if borrower_rate < r["rate"] else "equal",
        }
        for r in rates
    ]

    return {
        "borrower_rate": borrower_rate,
        "treasury_avg": treasury_avg,
        "spread": spread,
        "spread_assessment": assessment,
        "comparisons": comparisons,
        "record_date": treasury_data.get("record_date"),
    }
