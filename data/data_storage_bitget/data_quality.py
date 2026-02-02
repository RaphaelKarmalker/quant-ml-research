import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import random

# Base paths
BASE_PATH = Path(__file__).parent
CSV_DATA_PATH = BASE_PATH / "csv_data_all_bitget"
DEPTH_PATH = BASE_PATH / "Depth_Bitget"

# Time range filter: August 1, 2024 to February 2, 2026
START_DATE = pd.Timestamp('2024-08-01', tz='UTC')
END_DATE = pd.Timestamp('2026-02-02', tz='UTC')
START_TIMESTAMP_NANO = int(START_DATE.timestamp() * 1e9)
END_TIMESTAMP_NANO = int(END_DATE.timestamp() * 1e9)

def get_ohlcv_time_range(ohlcv_file):
    """Extract start and end timestamps from OHLCV file, filtered by date range"""
    try:
        df = pd.read_csv(ohlcv_file)
        if df.empty:
            return None, None
        
        # Filter by date range
        df = df[(df['timestamp_nano'] >= START_TIMESTAMP_NANO) & 
                (df['timestamp_nano'] <= END_TIMESTAMP_NANO)]
        
        if df.empty:
            return None, None
        
        start_ts = df['timestamp_nano'].min()
        end_ts = df['timestamp_nano'].max()
        return start_ts, end_ts
    except Exception as e:
        print(f"Error reading OHLCV {ohlcv_file}: {e}")
        return None, None

def get_lunar_first_nonzero(lunar_file):
    """Find first non-zero entry timestamp in Lunar data, filtered by date range"""
    try:
        df = pd.read_csv(lunar_file)
        if df.empty:
            return None
        
        # Filter by date range
        df = df[(df['timestamp_nano'] >= START_TIMESTAMP_NANO) & 
                (df['timestamp_nano'] <= END_TIMESTAMP_NANO)]
        
        if df.empty:
            return None
        
        # Check for first non-zero close price (or any price field)
        # Assuming close price is the main indicator
        if 'close' in df.columns:
            non_zero_df = df[df['close'] > 0]
            if not non_zero_df.empty:
                return non_zero_df['timestamp_nano'].min()
        
        # If no close column or all zeros, return first timestamp
        return df['timestamp_nano'].min()
    except Exception as e:
        print(f"Error reading Lunar {lunar_file}: {e}")
        return None

def get_depth_first_timestamp(depth_file):
    """Extract first timestamp from Depth file, filtered by date range"""
    try:
        df = pd.read_csv(depth_file)
        if df.empty:
            return None
        
        # Filter by date range
        df = df[(df['timestamp_nano'] >= START_TIMESTAMP_NANO) & 
                (df['timestamp_nano'] <= END_TIMESTAMP_NANO)]
        
        if df.empty:
            return None
        
        return df['timestamp_nano'].min()
    except Exception as e:
        print(f"Error reading Depth {depth_file}: {e}")
        return None

def analyze_data_quality():
    """Main function to analyze data quality across all instruments"""
    
    results = []
    
    # Iterate over all instrument folders in csv_data_all_bitget
    instrument_folders = [f for f in os.listdir(CSV_DATA_PATH) if os.path.isdir(CSV_DATA_PATH / f)]
    
    print(f"Found {len(instrument_folders)} instruments to analyze...")
    
    for idx, instrument_folder in enumerate(instrument_folders):
        if (idx + 1) % 50 == 0:
            print(f"Processing {idx + 1}/{len(instrument_folders)}...")
        
        instrument_path = CSV_DATA_PATH / instrument_folder
        ohlcv_file = instrument_path / "OHLCV.csv"
        lunar_file = instrument_path / "LUNAR.csv"
        
        # Check if OHLCV exists
        if not ohlcv_file.exists():
            continue
        
        # Get OHLCV time range
        ohlcv_start, ohlcv_end = get_ohlcv_time_range(ohlcv_file)
        if ohlcv_start is None:
            continue
        
        # Extract instrument name (remove -LINEAR suffix)
        instrument_name = instrument_folder.replace("-LINEAR", "")
        
        # Check for Lunar data
        has_lunar = lunar_file.exists()
        lunar_start = None
        lunar_delay = None
        
        if has_lunar:
            lunar_start = get_lunar_first_nonzero(lunar_file)
            if lunar_start is not None:
                # Calculate delay (only count if lunar starts AFTER ohlcv)
                if lunar_start > ohlcv_start:
                    lunar_delay = (lunar_start - ohlcv_start) / 1e9 / 3600  # Convert to hours
                else:
                    lunar_delay = 0  # No delay if lunar starts before or at the same time
        
        # Check for Depth data
        depth_folder = DEPTH_PATH / instrument_name
        depth_file = depth_folder / "DEPTH.csv"
        has_depth = depth_file.exists()
        depth_start = None
        depth_delay = None
        
        if has_depth:
            depth_start = get_depth_first_timestamp(depth_file)
            if depth_start is not None:
                # Calculate delay (only count if depth starts AFTER ohlcv)
                if depth_start > ohlcv_start:
                    depth_delay = (depth_start - ohlcv_start) / 1e9 / 3600  # Convert to hours
                else:
                    depth_delay = 0  # No delay if depth starts before or at the same time
        
        results.append({
            'instrument': instrument_folder,
            'ohlcv_start': ohlcv_start,
            'ohlcv_end': ohlcv_end,
            'has_lunar': has_lunar,
            'lunar_start': lunar_start,
            'lunar_delay_hours': lunar_delay,
            'has_depth': has_depth,
            'depth_start': depth_start,
            'depth_delay_hours': depth_delay
        })
    
    return pd.DataFrame(results)

def plot_results(df):
    """Create visualizations for the data quality analysis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Data availability histogram
    ax1 = axes[0, 0]
    availability_data = {
        'OHLCV': len(df),
        'OHLCV + Lunar': len(df[df['has_lunar'] == True]),
        'OHLCV + Depth': len(df[df['has_depth'] == True]),
        'OHLCV + Both': len(df[(df['has_lunar'] == True) & (df['has_depth'] == True)])
    }
    
    ax1.bar(availability_data.keys(), availability_data.values(), color=['blue', 'green', 'orange', 'red'])
    ax1.set_ylabel('Number of Instruments')
    ax1.set_title('Data Availability Across Instruments')
    ax1.tick_params(axis='x', rotation=15)
    
    for i, (key, value) in enumerate(availability_data.items()):
        ax1.text(i, value + 5, str(value), ha='center', fontweight='bold')
    
    # 2. Lunar delay distribution (only positive delays)
    ax2 = axes[0, 1]
    lunar_delays = df[df['lunar_delay_hours'].notna() & (df['lunar_delay_hours'] > 0)]['lunar_delay_hours']
    
    if len(lunar_delays) > 0:
        ax2.hist(lunar_delays, bins=30, color='green', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Delay (hours)')
        ax2.set_ylabel('Number of Instruments')
        ax2.set_title(f'Lunar Data Delay Distribution\nAvg: {lunar_delays.mean():.2f}h, Median: {lunar_delays.median():.2f}h')
        ax2.axvline(lunar_delays.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {lunar_delays.mean():.2f}h')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No Lunar Delays', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Lunar Data Delay Distribution')
    
    # 3. Depth delay distribution (only positive delays)
    ax3 = axes[1, 0]
    depth_delays = df[df['depth_delay_hours'].notna() & (df['depth_delay_hours'] > 0)]['depth_delay_hours']
    
    if len(depth_delays) > 0:
        ax3.hist(depth_delays, bins=30, color='orange', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Delay (hours)')
        ax3.set_ylabel('Number of Instruments')
        ax3.set_title(f'Depth Data Delay Distribution\nAvg: {depth_delays.mean():.2f}h, Median: {depth_delays.median():.2f}h')
        ax3.axvline(depth_delays.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {depth_delays.mean():.2f}h')
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'No Depth Delays', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Depth Data Delay Distribution')
    
    # 4. Coverage comparison
    ax4 = axes[1, 1]
    coverage_stats = {
        'Lunar Coverage': f"{(df['has_lunar'].sum() / len(df) * 100):.1f}%",
        'Depth Coverage': f"{(df['has_depth'].sum() / len(df) * 100):.1f}%",
        'Both Available': f"{((df['has_lunar'] & df['has_depth']).sum() / len(df) * 100):.1f}%"
    }
    
    y_pos = [0, 1, 2]
    percentages = [
        df['has_lunar'].sum() / len(df) * 100,
        df['has_depth'].sum() / len(df) * 100,
        (df['has_lunar'] & df['has_depth']).sum() / len(df) * 100
    ]
    
    bars = ax4.barh(y_pos, percentages, color=['green', 'orange', 'red'])
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(coverage_stats.keys())
    ax4.set_xlabel('Coverage (%)')
    ax4.set_title('Data Coverage Summary')
    ax4.set_xlim(0, 100)
    
    for i, (bar, pct) in enumerate(zip(bars, percentages)):
        ax4.text(pct + 1, bar.get_y() + bar.get_height()/2, 
                f'{pct:.1f}%', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(BASE_PATH / 'output' / 'data_quality_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {BASE_PATH / 'output' / 'data_quality_analysis.png'}")
    plt.show()

def print_sample_overview(df, n_samples=10):
    """Print detailed overview of random samples"""
    
    print("\n" + "="*100)
    print(f"SAMPLE OVERVIEW - {n_samples} Random Instruments")
    print("="*100)
    
    # Select random samples
    if len(df) < n_samples:
        samples = df
    else:
        samples = df.sample(n=n_samples, random_state=42)
    
    for idx, row in samples.iterrows():
        print(f"\n{'─'*100}")
        print(f"Instrument: {row['instrument']}")
        print(f"{'─'*100}")
        
        # OHLCV info
        ohlcv_start_dt = pd.to_datetime(row['ohlcv_start'], unit='ns')
        ohlcv_end_dt = pd.to_datetime(row['ohlcv_end'], unit='ns')
        ohlcv_duration = (row['ohlcv_end'] - row['ohlcv_start']) / 1e9 / 86400  # days
        
        print(f"  OHLCV:")
        print(f"    Start: {ohlcv_start_dt} ({row['ohlcv_start']})")
        print(f"    End:   {ohlcv_end_dt} ({row['ohlcv_end']})")
        print(f"    Duration: {ohlcv_duration:.2f} days")
        
        # Lunar info
        print(f"\n  LUNAR:")
        if row['has_lunar'] and row['lunar_start'] is not None:
            lunar_start_dt = pd.to_datetime(row['lunar_start'], unit='ns')
            print(f"    Available: YES")
            print(f"    First non-zero: {lunar_start_dt} ({int(row['lunar_start'])})")
            if row['lunar_delay_hours'] is not None:
                if row['lunar_delay_hours'] > 0:
                    print(f"    Delay: {row['lunar_delay_hours']:.2f} hours ({row['lunar_delay_hours']/24:.2f} days)")
                else:
                    print(f"    Delay: 0 hours (starts at or before OHLCV)")
        else:
            print(f"    Available: NO")
        
        # Depth info
        print(f"\n  DEPTH:")
        if row['has_depth'] and row['depth_start'] is not None:
            depth_start_dt = pd.to_datetime(row['depth_start'], unit='ns')
            print(f"    Available: YES")
            print(f"    First timestamp: {depth_start_dt} ({int(row['depth_start'])})")
            if row['depth_delay_hours'] is not None:
                if row['depth_delay_hours'] > 0:
                    print(f"    Delay: {row['depth_delay_hours']:.2f} hours ({row['depth_delay_hours']/24:.2f} days)")
                else:
                    print(f"    Delay: 0 hours (starts at or before OHLCV)")
        else:
            print(f"    Available: NO")
    
    print(f"\n{'='*100}\n")

def print_summary_statistics(df):
    """Print overall summary statistics"""
    
    print("\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100)
    print(f"\nDate Range Filter: {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}")
    
    print(f"\nTotal Instruments Analyzed: {len(df)}")
    print(f"\nData Availability:")
    print(f"  - Instruments with Lunar data: {df['has_lunar'].sum()} ({df['has_lunar'].sum()/len(df)*100:.1f}%)")
    print(f"  - Instruments with Depth data: {df['has_depth'].sum()} ({df['has_depth'].sum()/len(df)*100:.1f}%)")
    print(f"  - Instruments with both: {(df['has_lunar'] & df['has_depth']).sum()} ({(df['has_lunar'] & df['has_depth']).sum()/len(df)*100:.1f}%)")
    
    # Lunar delay statistics (only positive delays)
    lunar_delays = df[df['lunar_delay_hours'].notna() & (df['lunar_delay_hours'] > 0)]['lunar_delay_hours']
    print(f"\nLunar Data Delay Statistics:")
    print(f"  - Instruments with delay > 0: {len(lunar_delays)}")
    if len(lunar_delays) > 0:
        print(f"  - Average delay: {lunar_delays.mean():.2f} hours ({lunar_delays.mean()/24:.2f} days)")
        print(f"  - Median delay: {lunar_delays.median():.2f} hours ({lunar_delays.median()/24:.2f} days)")
        print(f"  - Min delay: {lunar_delays.min():.2f} hours ({lunar_delays.min()/24:.2f} days)")
        print(f"  - Max delay: {lunar_delays.max():.2f} hours ({lunar_delays.max()/24:.2f} days)")
    
    # Depth delay statistics (only positive delays)
    depth_delays = df[df['depth_delay_hours'].notna() & (df['depth_delay_hours'] > 0)]['depth_delay_hours']
    print(f"\nDepth Data Delay Statistics:")
    print(f"  - Instruments with delay > 0: {len(depth_delays)}")
    if len(depth_delays) > 0:
        print(f"  - Average delay: {depth_delays.mean():.2f} hours ({depth_delays.mean()/24:.2f} days)")
        print(f"  - Median delay: {depth_delays.median():.2f} hours ({depth_delays.median()/24:.2f} days)")
        print(f"  - Min delay: {depth_delays.min():.2f} hours ({depth_delays.min()/24:.2f} days)")
        print(f"  - Max delay: {depth_delays.max():.2f} hours ({depth_delays.max()/24:.2f} days)")
    
    print(f"\n{'='*100}\n")

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    output_dir = BASE_PATH / 'output'
    output_dir.mkdir(exist_ok=True)
    
    print("Starting data quality analysis...")
    print(f"Date Range: {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}")
    
    # Analyze all instruments
    results_df = analyze_data_quality()
    
    # Save results to CSV
    results_df.to_csv(output_dir / 'data_quality_results.csv', index=False)
    print(f"\nResults saved to: {output_dir / 'data_quality_results.csv'}")
    
    # Print summary statistics
    print_summary_statistics(results_df)
    
    # Print sample overview
    print_sample_overview(results_df, n_samples=10)
    
    # Create visualizations
    plot_results(results_df)
    
    print("\nAnalysis complete!")
