import requests
import pandas as pd

BASE_URL = "https://api.bybit.com/v5/market/instruments-info"

params = {
    "category": "linear",
    "symbol": "WUSDT"
}

resp = requests.get(BASE_URL, params=params)
data = resp.json()

if data["retCode"] != 0:
    raise RuntimeError(data)

inst = data["result"]["list"][0]

# Gezielt relevante Felder extrahieren
filtered = {
    "symbol": inst["symbol"],
    "launchTime": int(inst["launchTime"]),
    "maxLeverage": float(inst["leverageFilter"]["maxLeverage"]),
    "tickSize": float(inst["priceFilter"]["tickSize"]),
    "minNotionalValue": float(inst["lotSizeFilter"]["minNotionalValue"]),
    "fundingInterval": int(inst["fundingInterval"]),
    "upperFundingRate": float(inst["upperFundingRate"]),
    "lowerFundingRate": float(inst["lowerFundingRate"]),
    "priceLimitRatioX": float(inst["riskParameters"]["priceLimitRatioX"]),
    "priceLimitRatioY": float(inst["riskParameters"]["priceLimitRatioY"]),
}

df_static = pd.DataFrame([filtered])

print(df_static)
