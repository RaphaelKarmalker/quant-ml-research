# Quant ML Research
repository for various state-of-the-art approaches in ml time-series forecasting of stock data


## Datensätze
OHLCV von allen coins (ca. 250 coins, 2023/2024-td, freq. 15min, live)
Nur erste 30 Tage der coins
ByBit Brokerdaten (zb. Open interest, evtl orderbuch, BTC/Eth )
Wichtig: Live available daten
Errechnen was ist max capital von der strategie
Kriterium für coins: aktuell keins (alle neuen) ggf Kriterien aufstellen

(/v5/market/instruments-info)
(/v5/market/orderbook)

Preprocessing:
- ggf. Coins Klassifizieren (Meme, Layer2, etc)
- returns
- frac diff (0.4)


## Model / Architektur
- Gradient Boosting
- LSTM später (evtl mit attention)
- Transformer


## Target
1) Wahrscheinlichkeit, dass coin in X stunden um Y% droppt
1.1) Multi class classification oder direkt regression

2) 
2.1) Wahrscheinlichkeit aktuelle bar := unser top (globales top labeln, penalty für entfernung top)
2.2) wenn predicted penalty < thresh => trade

3) Returns predicten (z.b. next day)

4) Direction prediction 

Helbel:  Volatilität predicten