"""Test process_data.py script functionality for any sport."""

import pytest
import sys
import subprocess
import pandas as pd
import argparse
from pathlib import Path


def test_process_data(sport: str = "mma"):
    """Test process_data.py processes raw data and generates CSV for any sport."""
    # Always use PID 99 for golden tests
    pid = "99"
    test_path = Path(f"data/{sport}/{pid}/dk")

    # Verify test data exists (from setup_data test)
    raw_file = test_path / "json/raw.json"

    assert raw_file.exists(), f"Test data not found: {raw_file} - run test_setup_data.py --sport {sport} first"

    # Check for newsletter file (optional)
    newsletter_files = ["linestar.txt", "newsletter_signals.json"]
    newsletter_exists = any((test_path / "newsletters" / f).exists() for f in newsletter_files)

    # Run process_data.py
    result = subprocess.run([
        sys.executable, "src/cli/process_data.py",
        "--sport", sport,
        "--pid", pid,
        "--site", "dk"
    ], capture_output=True, text=True)

    # Check command succeeded
    assert result.returncode == 0, f"process_data.py failed: {result.stderr}"

    # Verify CSV was created
    csv_file = test_path / "csv/extracted.csv"
    assert csv_file.exists(), f"Processed CSV not created: {csv_file}"

    # Load and validate CSV
    df = pd.read_csv(csv_file)

    # Basic data validation
    assert len(df) > 0, "CSV file is empty"

    # Universal required columns (all sports)
    required_cols = [
        'player_id', 'name', 'salary', 'projection', 'floor', 'ceiling', 'ownership',
        'updated_projection', 'updated_ownership', 'updated_ceiling', 'updated_floor'
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"

    # Newsletter signal columns
    newsletter_cols = ['newsletter_signal', 'newsletter_confidence', 'newsletter_reason']
    for col in newsletter_cols:
        assert col in df.columns, f"Missing newsletter column: {col}"

    # Sport-specific column validation
    sport_specific_cols = {
        'mma': [
            'finishing_rate', 'style_score', 'matchup_advantage', 'takedown_matchup',
            'strikes_per_fight', 'strike_accuracy', 'takedowns_per_fight', 'itd_probability'
        ],
        'nascar': [
            'starting_position', 'bristol_avg_finish', 'avg_pass_diff', 'quality_passes_per_race',
            'fastest_laps_per_race', 'practice_lap_time', 'dominator_score', 'synthesized_floor'
        ],
        'nfl': [
            'position', 'team', 'opponent', 'game_total', 'spread'  # Expected NFL columns
        ]
    }

    if sport in sport_specific_cols:
        for col in sport_specific_cols[sport]:
            assert col in df.columns, f"Missing {sport}-specific column: {col}"

    # Data type validation
    assert df['salary'].dtype in ['int64', 'float64'], "Salary should be numeric"
    assert df['updated_projection'].dtype == 'float64', "Projection should be float"
    assert df['updated_ownership'].dtype == 'float64', "Ownership should be float"

    # Value ranges
    assert df['salary'].min() > 0, "Salaries should be positive"
    assert df['updated_ownership'].min() >= 0, "Ownership should be non-negative"
    assert df['updated_projection'].min() >= 0, "Projections should be non-negative"

    # Newsletter signals applied
    signal_counts = df['newsletter_signal'].value_counts()
    assert 'neutral' in signal_counts.index, "Should have neutral signals"

    # Check for newsletter adjustments (targets/fades should exist if signals present)
    non_neutral = df[df['newsletter_signal'] != 'neutral']
    if len(non_neutral) > 0 and newsletter_exists:
        # Should have confidence values
        assert non_neutral['newsletter_confidence'].notna().all(), "Newsletter confidence missing"
        assert (non_neutral['newsletter_confidence'] > 0).all(), "Newsletter confidence should be positive"

    print(f"✅ {sport.upper()} process data test passed - {len(df)} players processed")
    print(f"   Newsletter signals: {dict(signal_counts)}")
    if sport in sport_specific_cols:
        print(f"   {sport.upper()}-specific columns verified: {len(sport_specific_cols[sport])} columns")

    return df


def test_process_data_summary(sport: str = "mma"):
    """Test process_data.py summary display for any sport."""
    # Always use PID 99 for golden tests
    pid = "99"

    # Run summary-only mode
    result = subprocess.run([
        sys.executable, "src/cli/process_data.py",
        "--sport", sport,
        "--pid", pid,
        "--site", "dk",
        "--summary-only"
    ], capture_output=True, text=True)

    assert result.returncode == 0, f"Summary mode failed: {result.stderr}"

    # Check summary contains expected information
    output = result.stdout
    assert f"{sport.upper()} DATA SUMMARY" in output, "Summary header missing"
    assert "Total Players:" in output, "Player count missing"
    assert "Salary Range:" in output, "Salary info missing"
    assert "Newsletter Signals:" in output, "Newsletter summary missing"

    print(f"✅ {sport.upper()} process data summary test passed")


# Legacy functions for backward compatibility
def test_process_data_mma():
    """Test MMA process_data.py (legacy function)."""
    return test_process_data("mma")


def test_process_data_summary_mma():
    """Test MMA process_data.py summary (legacy function)."""
    return test_process_data_summary("mma")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test process_data.py for any sport")
    parser.add_argument("--sport", default="mma", choices=["mma", "nascar", "nfl", "nba"],
                       help="Sport to test")

    args = parser.parse_args()
    test_process_data(args.sport)
    test_process_data_summary(args.sport)