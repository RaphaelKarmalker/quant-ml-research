import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

BASE_URL = "https://api.bybit.com/v5/market/open-interest"

def fetch_open_interest(
    symbol: str,
    category: str = "linear",
    interval_time: str = "5min",
    limit: int = 200
) -> pd.DataFrame:
    """
    interval_time: 5min | 15min | 30min | 1h | 4h | 1d
    """

    params = {
        "category": category,
        "symbol": symbol,
        "intervalTime": interval_time,
        "limit": limit
    }

    resp = requests.get(BASE_URL, params=params)
    data = resp.json()

    if data["retCode"] != 0:
        raise RuntimeError(data)

    rows = data["result"]["list"]

    df = pd.DataFrame(rows)

    # exakt die existierenden Felder verwenden
    df["open_interest"] = df["openInterest"].astype(float)
    df["timestamp"] = pd.to_datetime(
        df["timestamp"].astype(int),
        unit="ms",
        utc=True
    )

    df = df[["timestamp", "open_interest"]]
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


if __name__ == "__main__":
    df_oi = fetch_open_interest("BTCUSDT", interval_time="5min")
    print(df_oi.tail())
