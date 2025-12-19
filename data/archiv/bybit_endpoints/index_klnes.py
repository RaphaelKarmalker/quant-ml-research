import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

# -----------------------------------
# INDEX PRICE KLINE ENDPOINT
# -----------------------------------
BASE_URL = "https://api.bybit.com/v5/market/index-price-kline"

symbol = "BTCUSDT"
category = "linear"
interval = "5"  # 5 Minuten

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

df_index = pd.DataFrame(
    klines,
    columns=[
        "start_time",
        "open",
        "high",
        "low",
        "close"
    ]
)

# Typen konvertieren
df_index["start_time"] = pd.to_datetime(
    df_index["start_time"].astype(int),
    unit="ms",
    utc=True
)

price_cols = ["open", "high", "low", "close"]
df_index[price_cols] = df_index[price_cols].astype(float)

df_index = df_index.set_index("start_time").sort_index()

print(df_index.tail())
print("Index-price candles:", len(df_index))
