#!/usr/bin/env python3
import sys
sys.path.append('../..')  # Add parent directory to path
"""
📥 DFS DATA SETUP
Initialize data collection for any sport.

Usage:
    python setup_data.py --sport mma --pid 466
    python setup_data.py --sport nfl --pid week1  
    python setup_data.py --sport nba --pid 2024-12-15

This script:
    ✅ Creates directory structure
    ✅ Provides data source URLs  
    ✅ Guides manual data collection
    ✅ Validates uploaded files
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List
import json
from src.models.site import SiteCode


class DataSetupManager:
    """Manages data setup for all sports."""
    
    def __init__(self, sport: str, pid: str, site: str = 'draftkings'):
        self.sport = sport.lower()
        self.pid = pid
        self.site = site.lower()
        self.base_path = Path(f"data/{self.sport}/{self.pid}/{self.site}")
        
        # Sport-specific configurations
        self.sport_configs = {
            'mma': {
                'name': 'MMA/UFC',
                'linestar_sport_id': 8,
                'site_ids': {
                    'dk': 1,
                    'fd': 2
                },
                'data_sources': [
                    {
                        'name': 'LineStar Contest Data',
                        'url_template': 'https://www.linestarapp.com/DesktopModules/DailyFantasyApi/API/Fantasy/GetSalariesV4?sport={sport_id}&site={site_id}&periodId={pid}',
                        'file': 'raw.json',
                        'description': 'Complete contest data including salaries, projections, and ownership'
                    }
                ],
                'newsletter_file': 'newsletter_signals.json'
            },
            'nascar': {
                'name': 'NASCAR',
                'linestar_sport_id': 9,  # NASCAR sport ID on LineStar
                'site_ids': {
                    'dk': 1,
                    'fd': 2
                },
                'data_sources': [
                    {
                        'name': 'LineStar NASCAR Contest Data',
                        'url_template': 'https://www.linestarapp.com/DesktopModules/DailyFantasyApi/API/Fantasy/GetSalariesV4?sport={sport_id}&site={site_id}&periodId={pid}',
                        'file': 'raw.json',
                        'description': 'NASCAR driver salaries, projections, ownership, and track data'
                    }
                ],
                'newsletter_file': 'newsletter_signals.json'
            },
            'nfl': {
                'name': 'NFL',
                'linestar_sport_id': 1,  # NFL sport ID on LineStar
                'site_ids': {
                    'dk': 1,
                    'fd': 2
                },
                'data_sources': [
                    {
                        'name': 'LineStar NFL Contest Data',
                        'url_template': 'https://www.linestarapp.com/DesktopModules/DailyFantasyApi/API/Fantasy/GetSalariesV4?sport={sport_id}&site={site_id}&periodId={pid}',
                        'file': 'raw.json',
                        'description': 'NFL player salaries, projections, ownership, and game data'
                    }
                ],
                'newsletter_file': 'newsletter_signals.json'
            },
            'nba': {
                'name': 'NBA',
                'raw_files': ['salaries.csv', 'ownership.csv', 'projections.csv'],
                'data_sources': [
                    {
                        'name': 'DraftKings Salaries',
                        'url_template': 'https://www.draftkings.com/lineup/getavailableplayerscsv?contestTypeId=70&draftGroupId={pid}',
                        'file': 'salaries.csv',
                        'description': 'Player salaries and positions'
                    }
                ],
                'newsletter_file': 'newsletter_signals.json'
            }
        }
    
    def create_directory_structure(self):
        """Create empty directories and files for manual data input."""
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
        
        # Create empty files for user to fill
        raw_file = json_dir / "raw.json"
        
        # Create empty JSON file
        if not raw_file.exists():
            raw_file.write_text("{}")
            print(f"📄 Created empty: {raw_file}")
    
    def display_data_sources(self):
        """Display data source URLs and instructions."""
        config = self.sport_configs.get(self.sport)
        if not config:
            print(f"❌ Sport '{self.sport}' not yet supported")
            return False
        
        print(f"\n🎯 {config['name']} DATA SOURCES")
        print("=" * 60)
        
        for i, source in enumerate(config['data_sources'], 1):
            # Get sport and site IDs for URL template
            sport_id = config.get('linestar_sport_id', '')
            site_id = config.get('site_ids', {}).get(self.site, 1)  # Default to DraftKings
            
            # Substitute all template variables
            filename = source['file'].replace('{pid}', self.pid)
            url = source['url_template'].format(
                pid=self.pid, 
                sport_id=sport_id, 
                site_id=site_id
            )
            
            print(f"\n{i}. {source['name']}")
            print(f"   📄 File: {filename}")  
            print(f"   📝 Description: {source['description']}")
            print(f"   🔗 URL: {url}")
            print(f"   💾 Save to: {self.base_path}/json/{filename}")
        
        
        return True
    
    def _get_newsletter_format(self) -> str:
        """Get newsletter JSON format example."""
        example = {
            "targets": [
                {"name": "Jon Jones", "confidence": 0.8, "reason": "Strong takedown defense vs wrestler"},
                {"name": "Amanda Nunes", "confidence": 0.7, "reason": "Experience advantage"}
            ],
            "fades": [
                {"name": "Conor McGregor", "confidence": 0.6, "reason": "Ring rust concerns"}
            ],
            "volatile": [
                {"name": "Tony Ferguson", "confidence": 0.7, "reason": "High variance fighter"}
            ]
        }
        return json.dumps(example, indent=2)
    
    def create_newsletter_template(self):
        """Create newsletter template file."""
        config = self.sport_configs.get(self.sport, {})
        newsletter_file = config.get('newsletter_file', 'newsletter_signals.json')
        newsletter_path = self.base_path / 'json' / newsletter_file
        
        template = {
            "targets": [],
            "fades": [], 
            "volatile": [],
            "notes": f"Newsletter signals for {self.sport.upper()} {self.pid}"
        }
        
        with open(newsletter_path, 'w') as f:
            json.dump(template, f, indent=2)
        
        print(f"📰 Created newsletter template: {newsletter_path}")
    
    def validate_files(self) -> bool:
        """Validate that required files have been uploaded."""
        config = self.sport_configs.get(self.sport, {})
        json_path = self.base_path / 'json'
        
        missing_files = []
        found_files = []
        
        for source in config.get('data_sources', []):
            # Substitute {pid} in filename
            filename = source['file'].replace('{pid}', self.pid)
            file_path = json_path / filename
            if file_path.exists():
                file_size = file_path.stat().st_size
                found_files.append(f"✅ {filename} ({file_size:,} bytes)")
            else:
                missing_files.append(f"❌ {filename}")
        
        # Check newsletter file (optional)
        newsletter_file = config.get('newsletter_file', '')
        if newsletter_file:
            newsletter_path = json_path / newsletter_file
            if newsletter_path.exists():
                found_files.append(f"📰 {newsletter_file} (optional)")
            else:
                print(f"⚠️  {newsletter_file} not found (optional)")
        
        print(f"\n📋 FILE VALIDATION")
        print("=" * 40)
        
        for file_status in found_files:
            print(f"  {file_status}")
        
        if missing_files:
            print(f"\n❌ Missing required files:")
            for missing in missing_files:
                print(f"  {missing}")
            return False
        else:
            print(f"\n✅ All required files found!")
            return True
    
    def display_next_steps(self):
        """Display next steps after data collection."""
        print(f"\n🎯 NEXT STEPS")
        print("=" * 40)
        print(f"1. Verify your data files in: {self.base_path}/json/")
        print(f"2. Run data processing:")
        print(f"   python process_data.py --sport {self.sport} --pid {self.pid}")
        print(f"3. After reviewing processed data:")
        print(f"   python optimize.py --sport {self.sport} --pid {self.pid}")


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
                       choices=['dk', 'fd'],
                       help='DFS site: dk=DraftKings, fd=FanDuel (default: dk)')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate existing files, skip setup')
    parser.add_argument('--create-newsletter', action='store_true',
                       help='Create newsletter template file')
    
    args = parser.parse_args()
    
    print("📥 DFS DATA SETUP")
    print("=" * 60)
    print(f"Sport: {args.sport.upper()}")
    print(f"PID: {args.pid}")
    print(f"Site: {args.site.upper()}")
    print("=" * 60)
    
    try:
        manager = DataSetupManager(args.sport, args.pid, args.site)
        
        if args.validate_only:
            # Just validate existing files
            success = manager.validate_files()
            if success:
                manager.display_next_steps()
            return 0 if success else 1
        
        # Full setup workflow
        print("\n1️⃣ Creating directory structure...")
        manager.create_directory_structure()
        
        if args.create_newsletter:
            print("\n2️⃣ Creating newsletter template...")
            manager.create_newsletter_template()
        
        print("\n3️⃣ Data source information:")
        if not manager.display_data_sources():
            return 1
        
        print(f"\n4️⃣ Manual Steps Required:")
        print("=" * 40)
        print("1. Visit each URL above")
        print("2. Download/save the data files to the specified locations")
        print("3. Run validation when complete:")
        print(f"   python setup_data.py --sport {args.sport} --pid {args.pid} --validate-only")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during setup: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())