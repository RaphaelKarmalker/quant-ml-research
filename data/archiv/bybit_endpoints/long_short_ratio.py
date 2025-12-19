import pandas as pd
import json

# RAW RESPONSE (hier symbolisch – bei dir kommt sie aus requests)
# data = resp.json()

data = {
    "retCode": 0,
    "result": {
        "list": [
            {"symbol": "BTCUSDT", "buyRatio": "0.6441", "sellRatio": "0.3559", "timestamp": "1766158800000"},
            {"symbol": "BTCUSDT", "buyRatio": "0.6445", "sellRatio": "0.3555", "timestamp": "1766158500000"},
            # ...
        ]
    }
}

rows = data["result"]["list"]

# In DataFrame
df_lsr = pd.DataFrame(rows)

# Typen konvertieren
df_lsr["timestamp"] = pd.to_datetime(
    df_lsr["timestamp"].astype(int),
    unit="ms",
    utc=True
)

df_lsr["buyRatio"] = df_lsr["buyRatio"].astype(float)
df_lsr["sellRatio"] = df_lsr["sellRatio"].astype(float)

# Optional aber sehr empfohlen: explizite Long/Short-Ratio
df_lsr["long_short_ratio"] = df_lsr["buyRatio"] / df_lsr["sellRatio"]

# Aufräumen
df_lsr = (
    df_lsr
    .drop(columns=["symbol"])
    .sort_values("timestamp")
    .reset_index(drop=True)
)

print(df_lsr.tail())
