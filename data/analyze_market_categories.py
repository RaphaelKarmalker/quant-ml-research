"""
Market Categories Analysis Script

This script analyzes the lunar_market_categories column from the cleaned dataset.
It extracts individual categories (since instruments can have multiple categories),
creates a histogram, and shows statistics.

Input:
  data_storage_bitget/final/all_matched_data_clean.csv

Output:
  Console report with histogram and statistics
"""

from pathlib import Path
import pandas as pd
import json
from collections import Counter
import matplotlib.pyplot as plt

BASE_DATA_DIR = Path(__file__).resolve().parent / "data_storage_bitget"
OUTPUT_DIR = BASE_DATA_DIR / "output"
INPUT_FILE = OUTPUT_DIR / "final" / "all_matched_data_clean.csv"


def parse_categories(cat_str):
    """Parse category string (JSON array format) into list of categories"""
    if pd.isna(cat_str) or cat_str == "" or cat_str == "0":
        return []
    
    try:
        # Handle JSON array format like ["solana-ecosystem", "meme"]
        if cat_str.startswith("["):
            categories = json.loads(cat_str)
            return [c.strip() for c in categories if c]
        else:
            return []
    except:
        return []


def analyze_categories():
    """Main analysis function"""
    print("=" * 100)
    print(" Market Categories Analysis")
    print("=" * 100)
    
    # Load data
    print(f"\nLoading: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Get unique instruments and their categories (one entry per instrument)
    print("\n" + "-" * 100)
    print(" Extracting unique instruments and their categories")
    print("-" * 100)
    
    instrument_categories = {}
    
    for instrument_id in df['instrument_id'].unique():
        # Get first row for this instrument (all rows have same category)
        cat_str = df[df['instrument_id'] == instrument_id]['lunar_market_categories'].iloc[0]
        categories = parse_categories(cat_str)
        instrument_categories[instrument_id] = categories
    
    print(f"\nTotal unique instruments: {len(instrument_categories)}")
    
    # Count instruments by number of categories
    cat_counts = Counter([len(cats) for cats in instrument_categories.values()])
    print(f"\nInstruments by number of categories:")
    for num_cats in sorted(cat_counts.keys()):
        print(f"  {num_cats} categories: {cat_counts[num_cats]} instruments")
    
    # Flatten all categories (each category counted separately)
    all_categories = []
    for cats in instrument_categories.values():
        all_categories.extend(cats)
    
    print(f"\nTotal category assignments: {len(all_categories)}")
    
    # Count occurrences of each category
    category_counter = Counter(all_categories)
    
    print("\n" + "=" * 100)
    print(" Category Frequency Distribution")
    print("=" * 100)
    print(f"{'Category':<40} {'Count':<10} {'Percentage':<15} {'Bar'}")
    print("-" * 100)
    
    total_assignments = len(all_categories) if all_categories else 1
    
    for category, count in category_counter.most_common():
        percentage = (count / total_assignments) * 100
        bar_length = int(percentage / 2)  # Scale to max 50 chars
        bar = "█" * bar_length
        print(f"{category:<40} {count:<10} {percentage:>6.2f}%        {bar}")
    
    print("-" * 100)
    print(f"Total unique categories: {len(category_counter)}")
    
    # Show instruments with multiple categories
    print("\n" + "=" * 100)
    print(" Instruments with Multiple Categories")
    print("=" * 100)
    
    multi_cat_instruments = {inst: cats for inst, cats in instrument_categories.items() if len(cats) > 1}
    
    if multi_cat_instruments:
        print(f"\nFound {len(multi_cat_instruments)} instruments with multiple categories:\n")
        for inst, cats in sorted(multi_cat_instruments.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"  {inst:<25} ({len(cats)} categories): {', '.join(cats)}")
    else:
        print("\n✓ No instruments with multiple categories found")
    
    # Show instruments without categories
    print("\n" + "=" * 100)
    print(" Instruments without Categories")
    print("=" * 100)
    
    no_cat_instruments = [inst for inst, cats in instrument_categories.items() if len(cats) == 0]
    
    if no_cat_instruments:
        print(f"\nFound {len(no_cat_instruments)} instruments without categories:")
        for inst in sorted(no_cat_instruments):
            print(f"  - {inst}")
    else:
        print("\n✓ All instruments have at least one category")
    
    # Create histogram plots
    print("\n" + "=" * 100)
    print(" Creating Histogram Plots")
    print("=" * 100)
    
    # PLOT 1: Number of categories per instrument
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    cat_count_data = sorted(cat_counts.items())
    x_labels = [f"{n} categories" if n != 0 else "No categories" for n, _ in cat_count_data]
    y_values = [count for _, count in cat_count_data]
    
    colors = ['#d62728' if n == 0 else '#2ca02c' if n == 1 else '#ff7f0e' if n == 2 else '#1f77b4' 
              for n, _ in cat_count_data]
    
    bars1 = ax1.bar(x_labels, y_values, color=colors, edgecolor='black', alpha=0.8)
    
    # Add value labels on bars
    for bar, count in zip(bars1, y_values):
        height = bar.get_height()
        percentage = (count / len(instrument_categories)) * 100
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count}\n({percentage:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_ylabel('Number of Instruments', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Category Count', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution: How Many Categories per Instrument?', 
                 fontsize=14, fontweight='bold', pad=20)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save plot 1
    output_plot1 = OUTPUT_DIR / "final" / "instruments_by_category_count.png"
    plt.savefig(output_plot1, dpi=300, bbox_inches='tight')
    print(f"\n✓ Category count distribution saved to: {output_plot1}")
    
    # PLOT 2: Category frequency histogram (original)
    if category_counter:
        fig2, ax2 = plt.subplots(figsize=(14, 8))
        
        categories = [cat for cat, _ in category_counter.most_common()]
        counts = [count for _, count in category_counter.most_common()]
        
        bars2 = ax2.barh(categories, counts, color='steelblue', edgecolor='black')
        
        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars2, counts)):
            percentage = (count / total_assignments) * 100
            ax2.text(count + 0.5, i, f'{count} ({percentage:.1f}%)', 
                   va='center', fontsize=9)
        
        ax2.set_xlabel('Number of Instruments', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Market Category', fontsize=12, fontweight='bold')
        ax2.set_title('Distribution of Market Categories Across Instruments', 
                    fontsize=14, fontweight='bold', pad=20)
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        # Save plot 2
        output_plot2 = OUTPUT_DIR / "final" / "market_categories_histogram.png"
        plt.savefig(output_plot2, dpi=300, bbox_inches='tight')
        print(f"✓ Category frequency histogram saved to: {output_plot2}")
        
        # Show both plots
        plt.show()
    else:
        print("\n[WARN] No categories found for frequency plot")
    
    # Summary statistics
    print("\n" + "=" * 100)
    print(" Summary Statistics")
    print("=" * 100)
    print(f"Total instruments analyzed:           {len(instrument_categories):>6}")
    print(f"Instruments with categories:          {len([i for i, c in instrument_categories.items() if c]):>6}")
    print(f"Instruments without categories:       {len(no_cat_instruments):>6}")
    print(f"Total unique categories:              {len(category_counter):>6}")
    print(f"Total category assignments:           {len(all_categories):>6}")
    print(f"Average categories per instrument:    {len(all_categories) / len(instrument_categories) if instrument_categories else 0:>6.2f}")
    
    if category_counter:
        most_common = category_counter.most_common(1)[0]
        print(f"Most common category:                 {most_common[0]} ({most_common[1]} instruments)")
    
    print("\n[DONE]")


if __name__ == "__main__":
    analyze_categories()
