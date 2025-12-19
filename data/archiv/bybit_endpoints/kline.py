import requests
import pandas as pd
import time
from datetime import datetime, timedelta, timezone

BASE_URL = "https://api.bybit.com/v5/market/kline"

symbol = "BTCUSDT"
category = "linear"
interval = "5"  # 5m

end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
start_time = int((datetime.now(timezone.utc) - timedelta(days=30)).timestamp() * 1000)

params = {
    "category": category,
    "symbol": symbol,
    "interval": interval,
    "start": start_time,
    "end": end_time,
    "limit": 1000
}

resp = requests.get(BASE_URL, params=params)
data = resp.json()

if data["retCode"] != 0:
    raise RuntimeError(data)

klines = data["result"]["list"]

df = pd.DataFrame(
    klines,
    columns=[
        "start_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover"
    ]
)

# Typen konvertieren
df["start_time"] = pd.to_datetime(df["start_time"].astype(int), unit="ms", utc=True)
numeric_cols = ["open", "high", "low", "close", "volume", "turnover"]
df[numeric_cols] = df[numeric_cols].astype(float)

df = df.set_index("start_time").sort_index()

print(df.tail())
print("Candles:", len(df))
