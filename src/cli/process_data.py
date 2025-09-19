#!/usr/bin/env python3
import sys
sys.path.append('../..')  # Add parent directory to path
"""
⚙️ DFS DATA PROCESSOR
Transform raw data into analysis-ready CSV format.

Usage:
    python src/cli/process_data.py --sport mma --pid 466 --site dk
    python src/cli/process_data.py --sport nascar --pid 503 --site dk
    python src/cli/process_data.py --sport nfl --pid 382 --site dk

This script:
    ✅ Loads raw data files (JSON)
    ✅ Applies newsletter signals & adjustments
    ✅ Calculates sport-specific metrics
    ✅ Generates final extracted CSV
    ✅ Provides data summary for review
"""

import argparse
import sys

from src.models.site import SiteCode
from src.processors import create_processor


def main():
    """Main data processing workflow."""
    parser = argparse.ArgumentParser(
        description="⚙️ DFS Data Processor - Transform raw data into analysis-ready CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--sport', type=str, required=True,
                       choices=['mma', 'nascar', 'nfl'],
                       help='Sport to process data for')
    parser.add_argument('--pid', type=str, required=True,
                       help='Contest/event identifier')
    parser.add_argument('--site', type=str, default='dk',
                       choices=[s.value for s in SiteCode],
                       help='DFS site: dk=DraftKings, fd=FanDuel (default: dk)')

    args = parser.parse_args()

    print("⚙️ DFS DATA PROCESSOR")
    print("=" * 60)
    print(f"Sport: {args.sport.upper()}")
    print(f"PID: {args.pid}")
    site_code = SiteCode.DK if args.site == 'dk' else SiteCode.FD
    print(f"Site: {site_code.full_name}")
    print("=" * 60)

    try:
        processor = create_processor(args.sport, args.pid, site_code)

        # Full processing workflow
        df = processor.process_data()
        output_file = processor.save_processed_data(df)
        processor.display_data_summary(df)

        print(f"\\n🎯 NEXT STEPS")
        print("=" * 60)
        print(f"1. Review processed data: {output_file}")
        print(f"2. Run optimization:")
        print(f"   python src/cli/optimize.py --sport {args.sport} --pid {args.pid} --site {args.site}")

        return 0

    except Exception as e:
        print(f"❌ Error during processing: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())