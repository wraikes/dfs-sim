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


class DataSetupManager:
    """Manages data setup for all sports."""
    
    def __init__(self, sport: str, pid: str):
        self.sport = sport.lower()
        self.pid = pid
        self.base_path = Path(f"data/{self.sport}/{self.pid}")
        
        # Sport-specific configurations
        self.sport_configs = {
            'mma': {
                'name': 'MMA/UFC',
                'raw_files': ['salaries.csv', 'ownership.csv', 'projections.csv'],
                'data_sources': [
                    {
                        'name': 'DraftKings Salaries',
                        'url_template': 'https://www.draftkings.com/lineup/getavailableplayerscsv?contestTypeId=21&draftGroupId={pid}',
                        'file': 'salaries.csv',
                        'description': 'Player salaries and basic info'
                    },
                    {
                        'name': 'RotoGrinders Ownership',
                        'url_template': 'https://rotogrinders.com/projected-ownership/UFC-{pid}',
                        'file': 'ownership.csv', 
                        'description': 'Projected ownership percentages'
                    },
                    {
                        'name': 'FantasyLabs Projections',
                        'url_template': 'https://www.fantasylabs.com/api/players/{pid}/UFC',
                        'file': 'projections.json',
                        'description': 'Player projections and metadata'
                    }
                ],
                'newsletter_file': 'newsletter_signals.json'
            },
            'nfl': {
                'name': 'NFL',
                'raw_files': ['salaries.csv', 'ownership.csv', 'projections.csv'],
                'data_sources': [
                    {
                        'name': 'DraftKings Salaries',
                        'url_template': 'https://www.draftkings.com/lineup/getavailableplayerscsv?contestTypeId=1&draftGroupId={pid}',
                        'file': 'salaries.csv',
                        'description': 'Player salaries and positions'
                    },
                    {
                        'name': 'RotoGrinders Ownership', 
                        'url_template': 'https://rotogrinders.com/projected-ownership/NFL-{pid}',
                        'file': 'ownership.csv',
                        'description': 'Projected ownership percentages'
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
        """Create necessary directories."""
        directories = [
            self.base_path / 'raw',
            self.base_path / 'csv', 
            self.base_path / 'output'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created: {directory}")
    
    def display_data_sources(self):
        """Display data source URLs and instructions."""
        config = self.sport_configs.get(self.sport)
        if not config:
            print(f"❌ Sport '{self.sport}' not yet supported")
            return False
        
        print(f"\n🎯 {config['name']} DATA SOURCES")
        print("=" * 60)
        
        for i, source in enumerate(config['data_sources'], 1):
            url = source['url_template'].format(pid=self.pid)
            print(f"\n{i}. {source['name']}")
            print(f"   📄 File: {source['file']}")  
            print(f"   📝 Description: {source['description']}")
            print(f"   🔗 URL: {url}")
            print(f"   💾 Save to: {self.base_path}/raw/{source['file']}")
        
        # Newsletter signals file
        newsletter_path = self.base_path / 'raw' / config['newsletter_file']
        print(f"\n📰 Newsletter Signals (Optional)")
        print(f"   📄 File: {config['newsletter_file']}")
        print(f"   📝 Manual creation of targets/fades/volatile plays")
        print(f"   💾 Save to: {newsletter_path}")
        print(f"   📋 Format: {self._get_newsletter_format()}")
        
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
        newsletter_path = self.base_path / 'raw' / newsletter_file
        
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
        raw_path = self.base_path / 'raw'
        
        missing_files = []
        found_files = []
        
        for source in config.get('data_sources', []):
            file_path = raw_path / source['file']
            if file_path.exists():
                file_size = file_path.stat().st_size
                found_files.append(f"✅ {source['file']} ({file_size:,} bytes)")
            else:
                missing_files.append(f"❌ {source['file']}")
        
        # Check newsletter file (optional)
        newsletter_file = config.get('newsletter_file', '')
        if newsletter_file:
            newsletter_path = raw_path / newsletter_file
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
        print(f"1. Verify your data files in: {self.base_path}/raw/")
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
                       choices=['mma', 'nfl', 'nba'],
                       help='Sport to set up data for')
    parser.add_argument('--pid', type=str, required=True,
                       help='Contest/event identifier (e.g., 466 for UFC 466)')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate existing files, skip setup')
    parser.add_argument('--create-newsletter', action='store_true',
                       help='Create newsletter template file')
    
    args = parser.parse_args()
    
    print("📥 DFS DATA SETUP")
    print("=" * 60)
    print(f"Sport: {args.sport.upper()}")
    print(f"PID: {args.pid}")
    print("=" * 60)
    
    try:
        manager = DataSetupManager(args.sport, args.pid)
        
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
        print("3. (Optional) Fill out newsletter signals JSON")
        print("4. Run validation when complete:")
        print(f"   python setup_data.py --sport {args.sport} --pid {args.pid} --validate-only")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during setup: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())