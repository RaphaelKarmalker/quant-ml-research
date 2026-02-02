"""
Pre-processing Step 5 for Bitget data: Behavior-Based Market Category Clustering

This script:
1. Maps individual market categories to behavioral clusters
2. Creates binary features for each cluster (0/1)
3. Removes the original category columns

Cluster Strategy:
- Maps granular categories (e.g., "meme", "ai", "gaming") to behavioral groups
- Each row gets 15 new binary columns representing cluster membership
- A coin can belong to multiple clusters if it has multiple categories

Input:
  data_storage_bitget/output/intermediate/step_3/all_matched_data_clean.csv
  (Or the output from your previous Step 4)

Output:
  data_storage_bitget/output/intermediate/step_5/all_matched_data_encoded.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
import json

# --- BEHAVIORAL CLUSTER MAPPING ---
BEHAVIOR_BASED_SHORT_CLUSTERS = {
    "extreme_fade": [
        "meme",
        "gambling",
        "pump-fun",
    ],

    "hype_tech_fade": [
        "ai",
        "ai-agents",
    ],

    "gaming_retail_fade": [
        "gaming",
        "entertainment",
        "fan",
    ],

    "nft_cycle_fade": [
        "nft",
    ],

    "dao_soft_fade": [
        "dao",
    ],

    "defi_extended_fade": [
        "lending-borrowing",
        "real-world-assets",
        "derivatives",
        "perpetuals",
        "stablecoin",
        "btcfi",
        "exchange-tokens",
    ],

    "defi_core_resilient": [
        "defi",
    ],

    "solana_ecosystem": [
        "solana-ecosystem",
    ],

    "base_ecosystem": [
        "base-ecosystem",
    ],

    "arbitrum_infra_ecosystem": [
        "arbitrum",
    ],

    "layer1_mixed": [
        "layer-1",
        "avalanche",
        "cardano",
        "sui",
        "algorand",
        "bsc",
    ],

    "layer2_speculative": [
        "layer-2",
        "zk",
    ],

    "infra_slow_fade": [
        "oracle",
        "interoperability",
        "storage",
    ],

    "data_identity_low_beta": [
        "analytics",
        "privacy",
        "socialfi",
        "desci",
    ],

    "long_tail_death": [
        "energy",
        "liquid-staking-tokens",
        "pow",
        "pos",
        "prediction",
        "sports",
        "dot",
        "inj",
    ],

    "reg_us_narrative": [
        "made-in-usa",
    ],
}
# ---------------------

BASE_DATA_DIR = Path(__file__).resolve().parent / "data_storage_bitget"
OUTPUT_DIR = BASE_DATA_DIR / "output"
INTERMEDIATE_DIR = OUTPUT_DIR / "intermediate"

# Adjust input file to match the output of your previous step
INPUT_FILE = INTERMEDIATE_DIR / "step_4" / "all_matched_data_clean.csv" 
OUTPUT_FOLDER = INTERMEDIATE_DIR / "step_5"
OUTPUT_FILE = OUTPUT_FOLDER / "all_matched_data_encoded.csv"


def parse_categories(cat_str):
    """Parse category string (JSON array format) into list of categories"""
    if pd.isna(cat_str) or cat_str == "" or cat_str == "0" or cat_str == "0.0":
        return []
    
    try:
        # Handle JSON array format like ["solana-ecosystem", "meme"]
        if isinstance(cat_str, str) and cat_str.startswith("["):
            categories = json.loads(cat_str)
            return [c.strip().lower() for c in categories if c and c != "0"]
        else:
            return []
    except:
        return []


def create_reverse_mapping():
    """Create a reverse mapping: category -> list of clusters"""
    reverse_map = {}
    for cluster_name, categories in BEHAVIOR_BASED_SHORT_CLUSTERS.items():
        for category in categories:
            if category not in reverse_map:
                reverse_map[category] = []
            reverse_map[category].append(cluster_name)
    return reverse_map


def map_categories_to_clusters(categories, reverse_map):
    """Map a list of categories to behavioral clusters"""
    clusters = set()
    for cat in categories:
        cat_lower = cat.lower().strip()
        if cat_lower in reverse_map:
            clusters.update(reverse_map[cat_lower])
    return clusters


def process_categorical_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps category columns to behavioral clusters and creates binary features.
    Only processes lunar_market_categories column (not BTC/ETH/etc. categories).
    """
    print("-" * 100)
    print(" PROCESSING CATEGORIES -> BEHAVIORAL CLUSTERS")
    print("-" * 100)

    # 1. Identify ONLY lunar_market_categories column (the coin's own categories, not major coins)
    category_col = 'lunar_market_categories'
    
    if category_col not in df.columns:
        print(f"Column '{category_col}' not found. Skipping encoding.")
        return df
    
    print(f"Processing column: {category_col}")

    # 2. Create reverse mapping
    reverse_map = create_reverse_mapping()
    print(f"\nCreated reverse mapping for {len(reverse_map)} individual categories")
    print(f"Mapping to {len(BEHAVIOR_BASED_SHORT_CLUSTERS)} behavioral clusters")

    # 3. Initialize cluster columns with zeros
    cluster_names = list(BEHAVIOR_BASED_SHORT_CLUSTERS.keys())
    for cluster in cluster_names:
        df[f'cluster_{cluster}'] = 0

    # 4. Process each row
    print("\nProcessing rows and mapping to clusters...")
    
    total_rows = len(df)
    processed_count = 0
    
    for idx, row in df.iterrows():
        # Get categories from lunar_market_categories column only
        categories = parse_categories(row[category_col])
        
        # Map to clusters
        clusters = map_categories_to_clusters(categories, reverse_map)
        
        # Set cluster columns to 1
        for cluster in clusters:
            df.at[idx, f'cluster_{cluster}'] = 1
        
        processed_count += 1
        if processed_count % 10000 == 0:
            print(f"  Processed {processed_count:,} / {total_rows:,} rows...")
    
    print(f"✓ Processed all {total_rows:,} rows")
    
    # 5. Drop original category column
    print(f"\nDropping original '{category_col}' column...")
    df_final = df.drop(columns=[category_col])
    
    # 6. Statistics
    print("\n" + "=" * 100)
    print(" CLUSTER STATISTICS")
    print("=" * 100)
    print(f"{'Cluster Name':<35} {'Rows with Cluster':<20} {'Percentage'}")
    print("-" * 100)
    
    for cluster in cluster_names:
        col_name = f'cluster_{cluster}'
        count = df_final[col_name].sum()
        percentage = (count / len(df_final)) * 100
        print(f"{cluster:<35} {count:<20} {percentage:>6.2f}%")
    
    print("-" * 100)
    print(f"Total cluster columns added: {len(cluster_names)}")
    
    return df_final


def run():
    """Main entry point"""
    print("=" * 100)
    print(" Pre-processing Step 5: Behavioral Cluster Encoding")
    print("=" * 100)
    
    # Check input file
    if not INPUT_FILE.exists():
        # Try alternative path
        alt_input = INTERMEDIATE_DIR / "step_3" / "all_matched_data_clean.csv"
        if alt_input.exists():
            print(f"Input '{INPUT_FILE}' not found, using: {alt_input}")
            input_path = alt_input
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
    print("\nNew Cluster Feature Columns:")
    new_cols = [c for c in df_encoded.columns if c.startswith('cluster_')]
    for col in new_cols:
        print(f"  - {col}")

    print("\n[DONE]")


if __name__ == "__main__":
    run()
