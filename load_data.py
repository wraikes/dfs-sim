#!/usr/bin/env python3
"""Main script for loading and managing DFS data."""

import sys
from pathlib import Path
from src.data.data_manager import DataManager
from src.data.json_extractor import JsonExtractor


def main():
    """Main data loading workflow."""
    print("\n" + "="*60)
    print("DFS DATA LOADER")
    print("="*60)
    
    # Initialize data manager
    dm = DataManager("data")
    
    # Check for existing data
    available = dm.list_available_data()
    if available:
        print("\nExisting data found:")
        for sport, periods in available.items():
            print(f"\n{sport.upper()}:")
            for period in periods:
                print(f"  - {period}")
    
    # Get user input
    print("\n" + "-"*40)
    sport = input("Enter sport (mma/nascar/nfl): ").lower()
    contest_type = input("Enter contest type (gpp/cash/qualifier): ").lower()
    period_id = input("Enter period ID (or 'current' for latest): ")
    
    # Generate download instructions
    url = dm.generate_download_url(sport, contest_type, "dk", period_id)
    
    # Wait for user to download
    print("\nAfter downloading, press Enter to continue...")
    input()
    
    # Try to load the data
    try:
        players = dm.load_json_data(sport, period_id)
        
        print(f"\n✅ Successfully loaded {len(players)} players!")
        
        # Show summary
        print("\nData Summary:")
        print(f"  Total players: {len(players)}")
        print(f"  Salary range: ${min(p.salary for p in players)} - ${max(p.salary for p in players)}")
        print(f"  Avg projection: {sum(p.projection for p in players) / len(players):.1f}")
        
        # Show top 5 by salary
        top_5 = sorted(players, key=lambda p: p.salary, reverse=True)[:5]
        print("\nTop 5 by salary:")
        for p in top_5:
            print(f"  {p.name:20} ${p.salary:5} {p.projection:6.1f}pts {p.ownership:5.1f}%")
        
        # Save to CSV
        save_csv = input("\nSave to CSV? (y/n): ").lower()
        if save_csv == 'y':
            dm.save_csv(players, sport, period_id)
            print("✅ Saved to CSV!")
        
        # Check for newsletter
        newsletter_path = dm.get_data_path(sport, period_id, "newsletter")
        print(f"\nTo add newsletter insights, paste content into:")
        print(f"  {newsletter_path}")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure you've saved the JSON file to the correct location.")
    except Exception as e:
        print(f"\n❌ Error loading data: {e}")


def quick_load(sport: str, period_id: str):
    """Quick load for testing."""
    dm = DataManager("data")
    
    try:
        players = dm.load_json_data(sport, period_id)
        print(f"Loaded {len(players)} {sport.upper()} players for period {period_id}")
        
        # Load newsletter if exists
        newsletter = dm.load_newsletter(sport, period_id)
        if newsletter:
            print(f"Newsletter loaded ({len(newsletter)} chars)")
        
        return players
        
    except Exception as e:
        print(f"Error: {e}")
        return []


if __name__ == "__main__":
    if len(sys.argv) > 2:
        # Quick load mode: python load_data.py mma 12345
        sport = sys.argv[1]
        period_id = sys.argv[2]
        quick_load(sport, period_id)
    else:
        # Interactive mode
        main()