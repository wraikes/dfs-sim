"""Golden Test: optimize.py script functionality for any sport."""

import pytest
import sys
import subprocess
import pandas as pd
import json
import argparse
from pathlib import Path


def test_optimize(sport: str = "mma"):
    """Golden Test: optimize.py generates lineups with small simulation count for any sport."""
    # Always use PID 99 for golden tests
    pid = "99"
    test_path = Path(f"data/{sport}/{pid}/dk")

    # Verify processed CSV exists (from process_data golden test)
    csv_file = test_path / "csv/extracted.csv"
    assert csv_file.exists(), f"Processed CSV not found: {csv_file} - run test_process_data.py --sport {sport} first"

    # Run optimize.py with small simulation count for testing
    result = subprocess.run([
        sys.executable, "src/cli/optimize.py",
        "--sport", sport,
        "--pid", pid,
        "--site", "dk",
        "--entries", "3",          # Small number of lineups
        "--simulations", "1000",   # Small simulation count for speed
        "--export-format", "csv"
    ], capture_output=True, text=True)

    # Check command succeeded
    assert result.returncode == 0, f"optimize.py failed: {result.stderr}"

    # Verify output file was created (path varies by sport)
    output_file = test_path / f"lineups/lineups_{pid}_gpp.csv"
    assert output_file.exists(), f"Lineup file not created: {output_file}"

    # Load and validate lineup CSV
    df = pd.read_csv(output_file)

    # Basic lineup validation
    assert len(df) > 0, "Lineup file is empty"

    # Sport-specific roster sizes
    roster_sizes = {
        'mma': 6,
        'nascar': 6,
        'nfl': 9,
        'nba': 8
    }
    roster_size = roster_sizes.get(sport, 6)

    # Should have 3 lineups × roster_size players
    expected_rows = 3 * roster_size
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
    assert len(lineup_ids) == 3, f"Expected 3 unique lineups, got {len(lineup_ids)}"

    for lineup_id in lineup_ids:
        lineup_rows = df[df['lineup_id'] == lineup_id]

        # Each lineup should have correct number of players
        assert len(lineup_rows) == roster_size, f"Lineup {lineup_id} has {len(lineup_rows)} players, expected {roster_size}"

        # Positions should be 1 to roster_size
        positions = sorted(lineup_rows['position'].tolist())
        expected_positions = list(range(1, roster_size + 1))
        assert positions == expected_positions, f"Invalid positions for lineup {lineup_id}"

        # Salary validation (sport-specific caps)
        salary_caps = {
            'mma': 50000,
            'nascar': 50000,
            'nfl': 50000,
            'nba': 50000
        }
        salary_cap = salary_caps.get(sport, 50000)
        min_salary = salary_cap - 2000  # Allow some flexibility

        total_salary = lineup_rows['salary'].sum()
        assert min_salary <= total_salary <= salary_cap, f"Lineup {lineup_id} salary ${total_salary:,} outside valid range"

        # No duplicate players
        player_ids = lineup_rows['player_id'].tolist()
        assert len(player_ids) == len(set(player_ids)), f"Lineup {lineup_id} has duplicate players"

    # Check console output
    output = result.stdout
    assert "DFS LINEUP OPTIMIZER" in output, "Missing header"
    assert "Generated 3 lineups" in output, "Missing lineup count"
    assert f"{sport.upper()} LINEUP SUMMARY" in output, "Missing summary"

    print(f"✅ {sport.upper()} golden optimize test passed - 3 lineups generated with 1000 simulations")
    print(f"   Output file: {output_file}")
    print(f"   Total rows: {len(df)} ({roster_size} players per lineup)")

    return df


def test_optimize_summary(sport: str = "mma"):
    """Golden Test: optimize.py summary mode for any sport."""
    # Always use PID 99 for golden tests
    pid = "99"

    # Run summary-only mode
    result = subprocess.run([
        sys.executable, "src/cli/optimize.py",
        "--sport", sport,
        "--pid", pid,
        "--site", "dk",
        "--summary-only"
    ], capture_output=True, text=True)

    assert result.returncode == 0, f"Summary mode failed: {result.stderr}"

    # Should show existing lineup count
    output = result.stdout
    assert "Found" in output and "existing lineups" in output, "Summary should show lineup count"

    print(f"✅ {sport.upper()} golden optimize summary test passed")


def test_optimize_dk_export(sport: str = "mma"):
    """Golden Test: optimize.py DraftKings export format for any sport."""
    # Always use PID 99 for golden tests
    pid = "99"

    # Test DK export format
    result = subprocess.run([
        sys.executable, "src/cli/optimize.py",
        "--sport", sport,
        "--pid", pid,
        "--site", "dk",
        "--entries", "2",
        "--simulations", "500",
        "--export-format", "dk"
    ], capture_output=True, text=True)

    assert result.returncode == 0, f"DK export failed: {result.stderr}"

    # Check DK upload file was created
    dk_file = Path(f"data/{sport}/{pid}/dk/lineups/dk_upload_{pid}.csv")
    assert dk_file.exists(), f"DK upload file not created: {dk_file}"

    # Validate DK format
    df = pd.read_csv(dk_file)
    assert len(df) == 2, "Should have 2 lineups"
    assert 'Entry ID' in df.columns, "Missing Entry ID column"

    # Sport-specific player column validation
    roster_sizes = {
        'mma': 6,
        'nascar': 6,
        'nfl': 9,
        'nba': 8
    }
    roster_size = roster_sizes.get(sport, 6)

    # Should have Player 1 to roster_size columns
    for i in range(1, roster_size + 1):
        col = f'Player {i}'
        assert col in df.columns, f"Missing {col} column"

    print(f"✅ {sport.upper()} golden DK export test passed")


# Legacy functions for backward compatibility
def test_optimize_mma_golden():
    """Golden Test: MMA optimize.py (legacy function)."""
    return test_optimize("mma")


def test_optimize_summary_golden():
    """Golden Test: MMA optimize.py summary (legacy function)."""
    return test_optimize_summary("mma")


def test_optimize_dk_export_golden():
    """Golden Test: MMA DK export (legacy function)."""
    return test_optimize_dk_export("mma")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test optimize.py for any sport")
    parser.add_argument("--sport", default="mma", choices=["mma", "nascar", "nfl", "nba"],
                       help="Sport to test")

    args = parser.parse_args()
    test_optimize(args.sport)
    test_optimize_summary(args.sport)
    test_optimize_dk_export(args.sport)