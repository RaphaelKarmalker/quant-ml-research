import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

BASE_URL = "https://api.bybit.com/v5/market/mark-price-kline"

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

df_mark = pd.DataFrame(
    klines,
    columns=[
        "start_time",
        "open",
        "high",
        "low",
        "close"
    ]
)

df_mark["start_time"] = pd.to_datetime(
    df_mark["start_time"].astype(int),
    unit="ms",
    utc=True
)

price_cols = ["open", "high", "low", "close"]
df_mark[price_cols] = df_mark[price_cols].astype(float)

df_mark = df_mark.set_index("start_time").sort_index()

print(df_mark.tail())
print("Mark-price candles:", len(df_mark))
