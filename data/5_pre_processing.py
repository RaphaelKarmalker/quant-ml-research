"""
Pre-processing Step 5 for Bitget data: Categorical One-Hot Encoding

This script:
1. Identifies all 'categories' columns (e.g., btc_lunar_market_categories, etc.)
2. Aggregates them into a single "Market View" per row (Market Basket)
3. One-hot encodes the top N most frequent categories across the entire market
4. Removes the original category columns

Configuration:
  ENCODE_ALL = False
  ENCODE_TOP_N_CATEGORIES = 10

Input:
  data_storage_bitget/output/intermediate/step_3/all_matched_data_clean.csv
  (Or the output from your previous Step 4)

Output:
  data_storage_bitget/output/intermediate/step_5/all_matched_data_encoded.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
ENCODE_ALL = False
ENCODE_TOP_N_CATEGORIES = 10
# ---------------------

BASE_DATA_DIR = Path(__file__).resolve().parent / "data_storage_bitget"
OUTPUT_DIR = BASE_DATA_DIR / "output"
INTERMEDIATE_DIR = OUTPUT_DIR / "intermediate"

# Adjust input file to match the output of your previous step
INPUT_FILE = INTERMEDIATE_DIR / "step_3" / "all_matched_data_clean.csv" 
OUTPUT_FOLDER = INTERMEDIATE_DIR / "step_5"
OUTPUT_FILE = OUTPUT_FOLDER / "all_matched_data_encoded.csv"

def process_categorical_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parses category columns, aggregates them per row, and 1-hot encodes the Top N.
    """
    print("-" * 100)
    print(" PROCESSING CATEGORIES")
    print("-" * 100)

    # 1. Identify Category Columns
    category_cols = [c for c in df.columns if 'categories' in c]
    print(f"Found {len(category_cols)} category columns:")
    for c in category_cols:
        print(f"  - {c}")

    if not category_cols:
        print("No category columns found. Skipping encoding.")
        return df

    # 2. Clean & Combine: Merge all coin columns into one massive string per row
    #    This creates a "Market/Portfolio State". If 'layer-1' appears in BTC and ETH,
    #    it is counted once for the row.
    print("\nCleaning and aggregating strings...")
    
    # Vectorized string cleaning: Remove brackets, quotes, spaces, and "0.0"
    clean_df = df[category_cols].astype(str).replace(r"[\[\]'\"\s]|0\.0", "", regex=True)
    
    # Combine into one string per row: "layer-1,pow,meme,bsc..."
    combined_text = clean_df.apply(lambda row: ','.join(filter(None, row.values)), axis=1)

    # 3. One-Hot Encode Globally (Vectorized)
    print("Generating one-hot vectors...")
    dummies = combined_text.str.get_dummies(sep=",")
    
    total_unique_cats = len(dummies.columns)
    print(f"Total unique categories found in dataset: {total_unique_cats}")

    # 4. Filter Top N or Select All
    if not ENCODE_ALL:
        # Sum column values to get global frequency
        top_cols = dummies.sum().nlargest(ENCODE_TOP_N_CATEGORIES).index
        print(f"\nFiltering for Top {ENCODE_TOP_N_CATEGORIES} categories:")
        for i, col in enumerate(top_cols, 1):
            count = dummies[col].sum()
            print(f"  {i}. {col} (appears in {count} rows)")
            
        dummies = dummies[top_cols]
    else:
        print(f"\nEncoding ALL {total_unique_cats} categories.")

    # 5. Prefix & Merge
    # Prefix with 'cat_' to avoid name collisions and indicate source
    dummies.columns = [f"cat_{c}" for c in dummies.columns]
    
    print(f"\nAdding {len(dummies.columns)} new feature columns...")
    
    # Drop original complex columns and add new binary features
    df_final = pd.concat([df, dummies], axis=1).drop(columns=category_cols)
    
    return df_final

def run():
    """Main entry point"""
    print("=" * 100)
    print(" Pre-processing Step 5: Categorical Encoding")
    print("=" * 100)
    
    # Check input file
    if not INPUT_FILE.exists():
        # Fallback if the specific 'clean' file doesn't exist, try standard match
        fallback = INTERMEDIATE_DIR / "step_3" / "all_matched_data.csv"
        if fallback.exists():
            print(f"Input '{INPUT_FILE}' not found, utilizing: {fallback}")
            input_path = fallback
        else:
            raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")
    else:
        input_path = INPUT_FILE
    
    print(f"\nLoading: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Process
    df_encoded = process_categorical_data(df)
    
    # Save
    print("\n" + "=" * 100)
    print(" SAVING DATA")
    print("=" * 100)
    
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    df_encoded.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n✓ Saved to: {OUTPUT_FILE}")
    print(f"  Rows: {len(df_encoded):,}")
    print(f"  Columns: {len(df_encoded.columns)}")
    
    # Preview new columns
    print("\nNew Feature Columns:")
    new_cols = [c for c in df_encoded.columns if c.startswith('cat_')]
    print(new_cols)

    print("\n[DONE]")

if __name__ == "__main__":
    run()