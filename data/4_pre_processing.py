"""
Pre-processing step 4 for Bitget data: Data Quality Check & Fix

This script:
1. Analyzes the final dataset for missing/empty/NaN/zero values per column
2. Fixes empty strings and NaN values by replacing them with 0
3. Reports statistics before and after fixing

Input:
  data_storage_bitget/final/all_matched_data.csv

Output:
  data_storage_bitget/final/all_matched_data_clean.csv
  Console report with statistics
"""

from pathlib import Path
import pandas as pd
import numpy as np

BASE_DATA_DIR = Path(__file__).resolve().parent / "data_storage_bitget"
OUTPUT_DIR = BASE_DATA_DIR / "output"
INPUT_FILE = OUTPUT_DIR / "final" / "all_matched_data.csv"
OUTPUT_FILE = OUTPUT_DIR / "final" / "all_matched_data_clean.csv"


def analyze_column(series: pd.Series) -> dict:
    """Analyze a single column for data quality issues"""
    total = len(series)
    
    # Count different types of "missing" values
    nan_count = series.isna().sum()
    
    # For object/string columns, check for empty strings
    if series.dtype == "object":
        empty_str_count = (series == "").sum()
        none_str_count = (series.astype(str).str.lower() == "none").sum()
    else:
        empty_str_count = 0
        none_str_count = 0
    
    # Count zeros (only for numeric columns)
    if pd.api.types.is_numeric_dtype(series):
        zero_count = (series == 0).sum()
    else:
        zero_count = 0
    
    # Count negative values (for numeric columns that shouldn't be negative)
    if pd.api.types.is_numeric_dtype(series):
        negative_count = (series < 0).sum()
    else:
        negative_count = 0
    
    return {
        "total": total,
        "nan": nan_count,
        "empty_str": empty_str_count,
        "none_str": none_str_count,
        "zeros": zero_count,
        "negative": negative_count,
        "missing_total": nan_count + empty_str_count + none_str_count,
        "dtype": str(series.dtype)
    }


def print_report(df: pd.DataFrame, title: str):
    """Print a detailed report of data quality"""
    print("\n" + "=" * 100)
    print(f" {title}")
    print("=" * 100)
    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    print("-" * 100)
    
    # Header
    print(f"{'Column':<40} {'Type':<12} {'NaN':<10} {'Empty':<10} {'Zeros':<10} {'Negative':<10} {'% Missing':<10}")
    print("-" * 100)
    
    issues_found = []
    
    for col in df.columns:
        stats = analyze_column(df[col])
        missing_pct = (stats["missing_total"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        
        # Highlight columns with issues
        has_issue = stats["missing_total"] > 0 or stats["negative"] > 0
        
        print(f"{col:<40} {stats['dtype']:<12} {stats['nan']:<10} {stats['empty_str']:<10} {stats['zeros']:<10} {stats['negative']:<10} {missing_pct:>8.2f}%")
        
        if has_issue:
            issues_found.append((col, stats))
    
    print("-" * 100)
    
    # Summary
    total_nan = df.isna().sum().sum()
    total_cells = df.size
    print(f"\nTotal NaN values across all columns: {total_nan:,} ({(total_nan/total_cells)*100:.4f}%)")
    
    if issues_found:
        print(f"\nColumns with missing values or negative values: {len(issues_found)}")
        for col, stats in issues_found:
            if stats["missing_total"] > 0:
                print(f"  - {col}: {stats['missing_total']} missing values")
            if stats["negative"] > 0:
                print(f"  - {col}: {stats['negative']} negative values")
    else:
        print("\n✓ No missing values found!")
    
    return issues_found


def fix_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fix missing values and negative placeholder values in the dataframe"""
    df = df.copy()
    fixes_applied = []
    
    for col in df.columns:
        original_nan = df[col].isna().sum()
        
        # Fix empty strings (for object columns)
        if df[col].dtype == "object":
            empty_mask = df[col] == ""
            none_mask = df[col].astype(str).str.lower() == "none"
            
            if empty_mask.sum() > 0:
                df.loc[empty_mask, col] = np.nan
                fixes_applied.append(f"{col}: {empty_mask.sum()} empty strings -> NaN")
            
            if none_mask.sum() > 0:
                df.loc[none_mask, col] = np.nan
                fixes_applied.append(f"{col}: {none_mask.sum()} 'None' strings -> NaN")
        
        # Fix negative placeholder values (like -999999) for numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            negative_mask = df[col] < 0
            if negative_mask.sum() > 0:
                df.loc[negative_mask, col] = 0
                fixes_applied.append(f"{col}: {negative_mask.sum()} negative values -> 0")
        
        # Now fill NaN with 0 for numeric columns, or "unknown" for string columns
        current_nan = df[col].isna().sum()
        if current_nan > 0:
            if pd.api.types.is_numeric_dtype(df[col]) or col not in ["timestamp", "instrument_id", "lunar_market_categories"]:
                # Try to convert to numeric first
                if df[col].dtype == "object":
                    try:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                        df[col] = df[col].fillna(0)
                        fixes_applied.append(f"{col}: {current_nan} NaN -> 0 (converted to numeric)")
                    except:
                        df[col] = df[col].fillna("unknown")
                        fixes_applied.append(f"{col}: {current_nan} NaN -> 'unknown'")
                else:
                    df[col] = df[col].fillna(0)
                    fixes_applied.append(f"{col}: {current_nan} NaN -> 0")
            else:
                # Keep as string for specific columns
                df[col] = df[col].fillna("unknown")
                fixes_applied.append(f"{col}: {current_nan} NaN -> 'unknown'")
    
    return df, fixes_applied


def run():
    """Main entry point"""
    print("=" * 100)
    print(" Pre-processing Step 4: Data Quality Check & Fix")
    print("=" * 100)
    
    # Check input file
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")
    
    print(f"\nLoading: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Analyze BEFORE fixing
    print_report(df, "BEFORE FIXING - Data Quality Report")
    
    # Fix the data
    print("\n" + "=" * 100)
    print(" APPLYING FIXES")
    print("=" * 100)
    
    df_fixed, fixes_applied = fix_data(df)
    
    if fixes_applied:
        print("\nFixes applied:")
        for fix in fixes_applied:
            print(f"  ✓ {fix}")
    else:
        print("\n✓ No fixes needed!")
    
    # Analyze AFTER fixing
    print_report(df_fixed, "AFTER FIXING - Data Quality Report")
    
    # Save the cleaned data
    print("\n" + "=" * 100)
    print(" SAVING CLEANED DATA")
    print("=" * 100)
    
    df_fixed.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✓ Saved to: {OUTPUT_FILE}")
    print(f"  Rows: {len(df_fixed):,}")
    print(f"  Columns: {len(df_fixed.columns)}")
    
    # Final column list
    print("\n" + "-" * 100)
    print("Final columns in cleaned dataset:")
    print("-" * 100)
    for i, col in enumerate(df_fixed.columns, 1):
        dtype = df_fixed[col].dtype
        print(f"  {i:2}. {col:<50} ({dtype})")
    
    print("\n[DONE]")


if __name__ == "__main__":
    run()
