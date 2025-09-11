#!/usr/bin/env python3
"""Test script for data loading functionality."""

import json
from pathlib import Path
from src.data import JsonExtractor, load_draftkings_csv
from src.models import DRAFTKINGS_NFL

def test_json_extraction():
    """Test JSON data extraction."""
    print("=" * 60)
    print("TESTING JSON EXTRACTION")
    print("=" * 60)
    
    # Test data structure (you'll replace with actual JSON)
    test_json = {
        "SalaryContainerJson": json.dumps({
            "Salaries": [
                {
                    "PID": "12345",
                    "PN": "Fighter One",
                    "PO": "FIGHTER",
                    "S": 8500,
                    "PS": 95.5,
                    "Id": "sal_123",
                    "TN": "",
                    "OP": "Fighter Two",
                    "GT": "Main Card",
                    "Projection": 102.3
                },
                {
                    "PID": "12346",
                    "PN": "Fighter Two",
                    "PO": "FIGHTER",
                    "S": 7700,
                    "PS": 88.2,
                    "Id": "sal_124",
                    "TN": "",
                    "OP": "Fighter One",
                    "GT": "Main Card",
                    "Projection": 95.8
                }
            ]
        }),
        "Ownership": {
            "Projected": {
                "slate_1": [
                    {"PlayerId": "12345", "SalaryId": "sal_123", "Owned": 35.2},
                    {"PlayerId": "12346", "SalaryId": "sal_124", "Owned": 28.5}
                ]
            }
        },
        "MatchupData": [
            {
                "Name": "Vegas Odds",
                "Columns": ["ML", "ITD%", "Dec%"],
                "PlayerMatchups": [
                    {
                        "PlayerId": "12345",
                        "Values": [-150, 45, 30]
                    },
                    {
                        "PlayerId": "12346",
                        "Values": [130, 35, 40]
                    }
                ]
            }
        ]
    }
    
    # Test MMA extraction
    extractor = JsonExtractor('mma')
    players = extractor.extract_from_json(test_json)
    
    print(f"\nExtracted {len(players)} MMA fighters:")
    for player in players:
        print(f"  {player.name}: ${player.salary}, {player.projection:.1f} pts, {player.ownership:.1f}% owned")
        if player.metadata:
            print(f"    Metadata: {list(player.metadata.keys())[:3]}...")
    
    print("\n" + "=" * 60)
    print("TESTING CSV LOADING")
    print("=" * 60)
    
    # Create test CSV file
    test_csv_path = Path("test_draftkings.csv")
    csv_content = """Name,Position,Salary,TeamAbbrev,AvgPointsPerGame,Game Info
Patrick Mahomes,QB,8000,KC,25.5,KC@BUF 08:20PM ET
Josh Allen,QB,7800,BUF,24.8,KC@BUF 08:20PM ET
Tyreek Hill,WR,8500,MIA,18.2,MIA@NYJ 01:00PM ET
Justin Jefferson,WR,8200,MIN,17.5,MIN@GB 04:25PM ET
Buffalo,DST,3000,BUF,9.5,KC@BUF 08:20PM ET"""
    
    with open(test_csv_path, 'w') as f:
        f.write(csv_content)
    
    # Test CSV loading
    try:
        players = load_draftkings_csv(str(test_csv_path), 'nfl')
        print(f"\nLoaded {len(players)} NFL players from CSV:")
        for player in players:
            print(f"  {player.name} ({player.position.value}): ${player.salary}, {player.projection:.1f} pts")
        
        # Test site validation
        site = DRAFTKINGS_NFL
        print(f"\nDraftKings NFL Settings:")
        print(f"  Salary Cap: ${site.salary_cap}")
        print(f"  Roster Slots: {site.roster_slots}")
        
    finally:
        # Clean up test file
        if test_csv_path.exists():
            test_csv_path.unlink()
    
    print("\n✅ Data loading tests complete!")


def test_real_json_file():
    """Test loading a real JSON file if available."""
    print("\n" + "=" * 60)
    print("TESTING REAL JSON FILE")
    print("=" * 60)
    
    # Look for JSON files in data directory
    data_dir = Path("data")
    if not data_dir.exists():
        print("Creating data directory...")
        data_dir.mkdir(parents=True)
        print("Place JSON files in the 'data' directory to test")
        return
    
    json_files = list(data_dir.glob("*.json"))
    if not json_files:
        print("No JSON files found in data directory")
        print("Place JSON files in the 'data' directory to test")
        return
    
    # Test first JSON file found
    json_file = json_files[0]
    print(f"\nTesting file: {json_file}")
    
    # Try to detect sport from filename
    filename = json_file.stem.lower()
    if 'mma' in filename or 'ufc' in filename:
        sport = 'mma'
    elif 'nascar' in filename:
        sport = 'nascar'
    elif 'nfl' in filename:
        sport = 'nfl'
    else:
        sport = 'nfl'  # default
    
    print(f"Detected sport: {sport}")
    
    try:
        extractor = JsonExtractor(sport)
        players = extractor.load_json_file(str(json_file))
        
        print(f"\nExtracted {len(players)} players")
        
        # Show top 10 by salary
        sorted_players = sorted(players, key=lambda p: p.salary, reverse=True)[:10]
        print("\nTop 10 players by salary:")
        for player in sorted_players:
            print(f"  {player.name:20} ${player.salary:5} {player.projection:6.1f}pts {player.ownership:5.1f}%")
        
        # Show some statistics
        total_salary = sum(p.salary for p in players)
        avg_salary = total_salary / len(players) if players else 0
        avg_projection = sum(p.projection for p in players) / len(players) if players else 0
        
        print(f"\nStatistics:")
        print(f"  Total players: {len(players)}")
        print(f"  Average salary: ${avg_salary:.0f}")
        print(f"  Average projection: {avg_projection:.1f}")
        
    except Exception as e:
        print(f"Error loading JSON: {e}")


if __name__ == "__main__":
    # Run tests
    test_json_extraction()
    test_real_json_file()