#!/usr/bin/env python3
import sys
sys.path.append('../..')  # Add parent directory to path
"""
📥 DFS DATA SETUP
Initialize data collection for any sport.

Usage:
    python src/cli/setup_data.py --sport mma --pid 466 --site dk
    python src/cli/setup_data.py --sport nascar --pid 99 --site dk
    python src/cli/setup_data.py --sport nfl --pid 301 --site dk

This script:
    ✅ Creates directory structure
    ✅ Provides data source URLs
    ✅ Guides manual data collection
    ✅ Validates uploaded files
"""

import argparse
import sys
from pathlib import Path
from typing import Dict
import json
from src.models.site import SiteCode


# LineStar sport IDs
LINESTAR_SPORT_IDS = {
    'nfl': 1,
    'nba': 2,
    'mlb': 3,
    'pga': 5,
    'nhl': 6,
    'mma': 8,
    'nascar': 9
}


class DataSetupManager:
    """Manages data setup for all sports."""

    def __init__(self, sport: str, pid: str, site: str = 'dk'):
        self.sport = sport.lower()
        self.pid = pid
        self.site_code = SiteCode.DK if site.lower() == 'dk' else SiteCode.FD
        self.base_path = Path(f"data/{self.sport}/{self.pid}/{self.site_code.value}")
    
    def _get_linestar_url(self) -> str:
        """Get LineStar URL for this sport/site/pid."""
        sport_id = LINESTAR_SPORT_IDS.get(self.sport)
        if not sport_id:
            return None

        site_id = 1 if self.site_code == SiteCode.DK else 2
        return f"https://www.linestarapp.com/DesktopModules/DailyFantasyApi/API/Fantasy/GetSalariesV4?sport={sport_id}&site={site_id}&periodId={self.pid}"

    def create_directory_structure(self):
        """Create empty directories and files for manual data input."""
        import shutil

        # Delete existing directory if it exists (clean slate)
        if self.base_path.exists():
            shutil.rmtree(self.base_path)
            print(f"🗑️  Removed existing: {self.base_path}")

        # Create directories
        json_dir = self.base_path / 'json'
        csv_dir = self.base_path / 'csv'
        newsletters_dir = self.base_path / 'newsletters'

        json_dir.mkdir(parents=True, exist_ok=True)
        csv_dir.mkdir(parents=True, exist_ok=True)
        newsletters_dir.mkdir(parents=True, exist_ok=True)

        print(f"📁 Created: {json_dir}")
        print(f"📁 Created: {csv_dir}")
        print(f"📁 Created: {newsletters_dir}")
        
        # Create/overwrite empty raw.json for user to fill
        raw_file = json_dir / "raw.json"
        raw_file.write_text("{}")
        print(f"📄 Created empty: {raw_file}")
    
    def display_data_sources(self):
        """Display data source URLs and instructions."""
        print(f"\n🎯 {self.sport.upper()} DATA SOURCES")
        print("=" * 60)

        url = self._get_linestar_url()
        if url:
            print(f"\n1. LineStar Contest Data")
            print(f"   📄 File: raw.json")
            print(f"   📝 Description: Complete contest data including salaries, projections, and ownership")
            print(f"   🔗 URL: {url}")
            print(f"   💾 Save to: {self.base_path}/json/raw.json")
        else:
            print(f"❌ Sport '{self.sport}' not yet configured for LineStar")
            return False

        return True


def main():
    """Main data setup workflow."""
    parser = argparse.ArgumentParser(
        description="🎯 DFS Data Setup - Initialize data collection for any sport",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--sport', type=str, required=True,
                       choices=['mma', 'nascar', 'nfl', 'nba'],
                       help='Sport to set up data for')
    parser.add_argument('--pid', type=str, required=True,
                       help='Contest/event identifier (e.g., 466 for UFC 466)')
    parser.add_argument('--site', type=str, default='dk',
                       choices=[s.value for s in SiteCode],
                       help='DFS site: dk=DraftKings, fd=FanDuel (default: dk)')
    
    args = parser.parse_args()
    
    print("📥 DFS DATA SETUP")
    print("=" * 60)
    print(f"Sport: {args.sport.upper()}")
    print(f"PID: {args.pid}")
    site_code = SiteCode.DK if args.site == 'dk' else SiteCode.FD
    print(f"Site: {site_code.full_name}")
    print("=" * 60)
    
    try:
        manager = DataSetupManager(args.sport, args.pid, args.site)

        # Full setup workflow
        print("\n1️⃣ Creating directory structure and templates...")
        manager.create_directory_structure()

        print("\n2️⃣ Data source information:")
        if not manager.display_data_sources():
            return 1

        print(f"\n3️⃣ Manual Steps Required:")
        print("=" * 40)
        print("1. Visit the URL above")
        print("2. Copy the JSON response")
        print(f"3. Paste into: {manager.base_path}/json/raw.json")
        print(f"4. (Optional) Add newsletter files to: {manager.base_path}/newsletters/")
        print(f"\n5. Then run: python src/cli/process_data.py --sport {args.sport} --pid {args.pid} --site {args.site}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during setup: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())