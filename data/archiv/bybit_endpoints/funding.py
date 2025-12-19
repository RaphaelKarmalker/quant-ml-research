import requests
import pandas as pd
from datetime import datetime, timezone

BASE_URL = "https://api.bybit.com/v5/market/funding/history"

def fetch_funding_history(symbol: str, category: str = "linear", limit: int = 200) -> pd.DataFrame:
    params = {
        "category": category,
        "symbol": symbol,
        "limit": limit
    }

    resp = requests.get(BASE_URL, params=params)
    data = resp.json()

    if data["retCode"] != 0:
        raise RuntimeError(data)

    rows = data["result"]["list"]

    # Nur Funding-Rate + Timestamp extrahieren
    df = pd.DataFrame(rows)[
        ["fundingRate", "fundingRateTimestamp"]
    ]

    # Typen sauber konvertieren
    df["funding_rate"] = df["fundingRate"].astype(float)
    df["timestamp"] = pd.to_datetime(
        df["fundingRateTimestamp"].astype(int),
        unit="ms",
        utc=True
    )

    df = df[["timestamp", "funding_rate"]].sort_values("timestamp").reset_index(drop=True)

    return df


if __name__ == "__main__":
    df_funding = fetch_funding_history("BTCUSDT")
    print(df_funding.tail())
