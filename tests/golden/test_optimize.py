"""Golden Test: optimize.py script functionality."""

import pytest
import sys
import subprocess
import pandas as pd
import json
from pathlib import Path


def test_optimize_mma_golden():
    """Golden Test: MMA optimize.py generates lineups with small simulation count."""
    test_path = Path("data/mma/99/dk")
    
    # Verify processed CSV exists (from process_data golden test)
    csv_file = test_path / "csv/extracted.csv"
    assert csv_file.exists(), f"Processed CSV not found: {csv_file} - run test_process_data.py first"
    
    # Run optimize.py with small simulation count for testing
    result = subprocess.run([
        sys.executable, "src/cli/optimize.py",
        "--sport", "mma",
        "--pid", "99",
        "--site", "dk",
        "--entries", "5",          # Small number of lineups
        "--simulations", "1000",   # Small simulation count for speed
        "--export-format", "csv"
    ], capture_output=True, text=True)
    
    # Check command succeeded
    assert result.returncode == 0, f"optimize.py failed: {result.stderr}"
    
    # Verify output file was created
    output_file = test_path / "lineups/lineups_99.csv"
    assert output_file.exists(), f"Lineup file not created: {output_file}"
    
    # Load and validate lineup CSV
    df = pd.read_csv(output_file)
    
    # Basic lineup validation
    assert len(df) > 0, "Lineup file is empty"
    
    # Should have 5 lineups × 6 fighters = 30 rows
    expected_rows = 5 * 6  # 5 lineups, 6 fighters each
    assert len(df) == expected_rows, f"Expected {expected_rows} rows, got {len(df)}"
    
    # Required columns
    required_cols = [
        'lineup_id', 'position', 'player_name', 'salary', 
        'projection', 'ownership', 'player_id'
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"
    
    # Optional performance columns
    perf_cols = ['gpp_score', 'percentile_95', 'percentile_99']
    for col in perf_cols:
        if col in df.columns:
            assert df[col].notna().all(), f"Performance column {col} has null values"
    
    # Validate lineup structure
    lineup_ids = df['lineup_id'].unique()
    assert len(lineup_ids) == 5, f"Expected 5 unique lineups, got {len(lineup_ids)}"
    
    for lineup_id in lineup_ids:
        lineup_rows = df[df['lineup_id'] == lineup_id]
        
        # Each lineup should have 6 fighters
        assert len(lineup_rows) == 6, f"Lineup {lineup_id} has {len(lineup_rows)} fighters, expected 6"
        
        # Positions should be 1-6
        positions = sorted(lineup_rows['position'].tolist())
        assert positions == [1, 2, 3, 4, 5, 6], f"Invalid positions for lineup {lineup_id}"
        
        # Salary validation
        total_salary = lineup_rows['salary'].sum()
        assert 48800 <= total_salary <= 50000, f"Lineup {lineup_id} salary ${total_salary:,} outside valid range"
        
        # No duplicate players
        player_ids = lineup_rows['player_id'].tolist()
        assert len(player_ids) == len(set(player_ids)), f"Lineup {lineup_id} has duplicate players"
    
    # Check console output
    output = result.stdout
    assert "DFS LINEUP OPTIMIZER" in output, "Missing header"
    assert "Generated 5 lineups" in output, "Missing lineup count"
    assert "MMA LINEUP SUMMARY" in output, "Missing summary"
    
    print(f"✅ Golden optimize test passed - 5 lineups generated with 1000 simulations")
    print(f"   Output file: {output_file}")
    print(f"   Total rows: {len(df)}")


def test_optimize_summary_golden():
    """Golden Test: optimize.py summary mode."""
    # Run summary-only mode
    result = subprocess.run([
        sys.executable, "src/cli/optimize.py",
        "--sport", "mma",
        "--pid", "99",
        "--site", "dk",
        "--summary-only"
    ], capture_output=True, text=True)
    
    assert result.returncode == 0, f"Summary mode failed: {result.stderr}"
    
    # Should show existing lineup count
    output = result.stdout
    assert "Found" in output and "existing lineups" in output, "Summary should show lineup count"
    
    print("✅ Golden optimize summary test passed")


def test_optimize_dk_export_golden():
    """Golden Test: optimize.py DraftKings export format."""
    # Test DK export format
    result = subprocess.run([
        sys.executable, "src/cli/optimize.py",
        "--sport", "mma",
        "--pid", "99",
        "--site", "dk",
        "--entries", "3",
        "--simulations", "500", 
        "--export-format", "dk"
    ], capture_output=True, text=True)
    
    assert result.returncode == 0, f"DK export failed: {result.stderr}"
    
    # Check DK upload file was created
    dk_file = Path("data/mma/99/dk/lineups/dk_upload_99.csv")
    assert dk_file.exists(), f"DK upload file not created: {dk_file}"
    
    # Validate DK format
    df = pd.read_csv(dk_file)
    assert len(df) == 3, "Should have 3 lineups"
    assert 'Entry ID' in df.columns, "Missing Entry ID column"
    
    # Should have Player 1-6 columns
    for i in range(1, 7):
        col = f'Player {i}'
        assert col in df.columns, f"Missing {col} column"
    
    print("✅ Golden DK export test passed")


if __name__ == "__main__":
    test_optimize_mma_golden()
    test_optimize_summary_golden()
    test_optimize_dk_export_golden()