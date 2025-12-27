"""
Run all preprocessing steps in sequence:

0. pre_processing_1_1.py - Major Coin Data Preparation
   - Merged OHLCV.csv + METRICS.csv für BTC, ETH, DOGE, SOL
   - Input: dataset_storage/coin_specific_data/{COIN}USDT-LINEAR/OHLCV.csv + METRICS.csv
   - Output: dataset_storage/coin_specific_data/{COIN}USDT-LINEAR/merged_coin_data.csv

1. preprocessing_1.py - Feature Merging
   - Lädt OHLCV.csv für jedes Symbol aus csv_data_all/
   - Merged METRICS.csv (open_interest, funding_rate, long_short_ratio) via merge_asof
   - Merged Major Coin Data von BTC, ETH, DOGE, SOL aus dataset_storage/coin_specific_data/{COIN}USDT-LINEAR/merged_coin_data.csv
     (Features: close, open_interest, funding_rate, long_short_ratio per coin)
   - Merged FNG (Fear & Greed Index) aus dataset_storage/FNG-INDEX.BYBIT/FNG.csv
   - Output: dataset_storage/step_1/{SYMBOL}/matched_data.csv

2. pre_processing_2.py - Data Cleaning & Transformation
   - Erstellt timestamp Spalte im Format "YYYY-MM-DD HH:MM:SS"
   - Fügt ts_since_listing hinzu (fortlaufende Nummer ab 1)
   - Entfernt timestamp_nano, timestamp_iso, *_timestamp_nano Spalten
   - Sortiert Spalten: timestamp, ts_since_listing, instrument_id, dann Features
   - Output: dataset_storage/step_2/{SYMBOL}/matched_data_filtered.csv

3. pre_processing_3.py - Symbol Consolidation
   - Sammelt alle matched_data_filtered.csv von allen Symbolen
   - Erstellt Union-Schema aller Spalten (fehlende Spalten → 0)
   - Merged streamweise in eine einzige Datei (memory-efficient)
   - Output: dataset_storage/final/all_matched_data.csv
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

import pre_processing_1_1
import preprocessing_1
import pre_processing_2
import pre_processing_3


def main():
    print("=" * 80)
    print("STARTING PREPROCESSING PIPELINE")
    print("=" * 80)
    
    # Step 0: Prepare major coin data
    print("\n[STEP 0/4] Running pre_processing_1_1.py - Preparing major coin data...")
    print("-" * 80)
    try:
        pre_processing_1_1.run()
        print("\n✓ Step 0 completed successfully")
    except Exception as e:
        print(f"\n✗ Step 0 failed: {e}")
        sys.exit(1)
    
    # Step 1: Merge features
    print("\n" + "=" * 80)
    print("[STEP 1/4] Running preprocessing_1.py - Merging features...")
    print("-" * 80)
    try:
        preprocessing_1.run()
        print("\n✓ Step 1 completed successfully")
    except Exception as e:
        print(f"\n✗ Step 1 failed: {e}")
        sys.exit(1)
    
    # Step 2: Filter and transform
    print("\n" + "=" * 80)
    print("[STEP 2/4] Running pre_processing_2.py - Filtering and transforming...")
    print("-" * 80)
    try:
        pre_processing_2.run()
        print("\n✓ Step 2 completed successfully")
    except Exception as e:
        print(f"\n✗ Step 2 failed: {e}")
        sys.exit(1)
    
    # Step 3: Merge all symbols
    print("\n" + "=" * 80)
    print("[STEP 3/4] Running pre_processing_3.py - Merging all symbols...")
    print("-" * 80)
    try:
        pre_processing_3.run()
        print("\n✓ Step 3 completed successfully")
    except Exception as e:
        print(f"\n✗ Step 3 failed: {e}")
        sys.exit(1)
    
    # Success
    print("\n" + "=" * 80)
    print("✓ PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
