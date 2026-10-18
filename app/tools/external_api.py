"""Tool 3: call a live external API — currency conversion.

Uses Frankfurter (https://frankfurter.dev), a free exchange-rate API with
no key, based on European Central Bank reference rates. It supports every
currency in the eval questions (USD, GBP, EUR, INR, AUD, CAD).

ECB rates are published once per working day, so "today's rate" is the
latest published one; the result includes its date. Any failure comes
back as a short message the model can pass on, never as a crash.
"""

import json

import httpx
from pydantic import BaseModel, Field

from app.core.config import settings

DESCRIPTION = (
    "Convert an amount between currencies at the latest exchange rate (live API, European "
    "Central Bank reference rates). Use it for any question about exchange rates or prices in "
    "a currency other than USD. Shop prices are stored in USD. To get just the rate, use amount 1."
)


class ConvertCurrencyArgs(BaseModel):
    amount: float = Field(description="The amount to convert, e.g. 250")
    from_currency: str = Field(description="3-letter currency code to convert from, e.g. EUR")
    to_currency: str = Field(description="3-letter currency code to convert to, e.g. INR")


def convert(
    amount: float,
    from_currency: str,
    to_currency: str,
    *,
    client: httpx.Client | None = None,
) -> dict:
    src, dst = from_currency.strip().upper(), to_currency.strip().upper()
    if len(src) != 3 or len(dst) != 3 or not (src + dst).isalpha():
        return {"error": "Currency codes must be 3 letters, like USD or GBP."}
    if src == dst:
        return {
            "from": src,
            "to": dst,
            "amount": amount,
            "rate": 1.0,
            "converted": round(amount, 2),
            "rate_date": None,
            "source": "no conversion needed",
        }

    own_client = client is None
    client = client or httpx.Client(timeout=settings.fx_api_timeout)
    try:
        response = client.get(f"{settings.fx_api_url}/latest", params={"base": src, "symbols": dst})
        if response.status_code == 404 or response.status_code == 422:
            return {"error": f"The exchange-rate service doesn't support {src} or {dst}."}
        response.raise_for_status()
        data = response.json()
        rate = float(data["rates"][dst])
    except httpx.TimeoutException:
        return {"error": "The exchange-rate service didn't answer in time. Try again in a moment."}
    except (httpx.HTTPError, KeyError, ValueError):
        return {"error": "The exchange-rate service isn't available right now. Try again later."}
    finally:
        if own_client:
            client.close()

    return {
        "from": src,
        "to": dst,
        "amount": amount,
        "rate": rate,
        "converted": round(amount * rate, 2),
        "rate_date": data.get("date"),
        "source": "Frankfurter API (European Central Bank reference rates)",
    }


def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    return json.dumps(convert(amount, from_currency, to_currency))
