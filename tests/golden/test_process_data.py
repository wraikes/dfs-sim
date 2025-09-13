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

    # CRITICAL FIELD VALIDATION - these are essential for each sport and must not be lost during refactoring
    critical_fields_by_sport = {
        'mma': {
            # Base MMA data extraction
            'ml_odds': 'Moneyline odds from JSON',
            'itd_probability': 'Inside The Distance probability calculation',
            'takedowns_per_fight': 'Takedown statistics from JSON',
            'takedown_defense': 'Takedown defense percentage',
            'strikes_per_fight': 'Striking volume statistics',
            'strike_accuracy': 'Striking accuracy percentage',

            # Our calculated MMA features (critical for optimization)
            'finishing_rate': 'Fight finishing calculation method',
            'style_score': 'Fighting style score calculation',
            'matchup_advantage': 'Head-to-head matchup calculation',
            'takedown_matchup': 'Opponent-based takedown matchup score',
        },
        'nascar': {
            # Essential NASCAR data from 6 JSON tables
            'starting_position': 'Qualifying position from pmatchup table',
            'practice_lap_time': 'Practice performance from pmatchup table',
            'bristol_avg_finish': 'Track-specific performance from tmatchup table',
            'avg_pass_diff': 'Position advancement from dSeason table',
            'quality_passes_per_race': 'Passing ability from dSeason table',
            'fastest_laps_per_race': 'Speed consistency from dSeason table',

            # Our calculated NASCAR features (critical for correlations/variance)
            'position_differential': 'Starting position vs target calculation',
            'dominator_score': 'Lap-leading potential calculation',
            'synthesized_floor': 'DNF risk-based floor calculation',
            'variance_multiplier': 'Position-based variance adjustment',
        },
        'nfl': {
            # Expected NFL critical fields (when implemented)
            'position': 'Player position from roster data',
            'team': 'Team affiliation for stacking',
            'opponent': 'Opposing team for correlation',
            'game_total': 'Vegas game total for environment',
            'target_share': 'Receiving target percentage',
        }
    }

    # Validate critical fields exist and have realistic values
    if sport in critical_fields_by_sport:
        critical_fields = critical_fields_by_sport[sport]
        missing_critical = []

        print(f"\n🔍 Validating {len(critical_fields)} critical {sport.upper()} fields:")

        for field, description in critical_fields.items():
            if field not in df.columns:
                missing_critical.append(f"{field} ({description})")
                print(f"  ❌ MISSING CRITICAL: {field} - {description}")
            else:
                # Check field has meaningful data (not all zeros/nulls)
                non_null_count = df[field].notna().sum()
                has_variation = df[field].nunique() > 1
                sample_val = df[field].iloc[0] if non_null_count > 0 else 'NULL'

                if non_null_count == 0:
                    print(f"  ⚠️  {field}: ALL NULL - {description}")
                elif not has_variation and df[field].iloc[0] == 0:
                    print(f"  ⚠️  {field}: ALL ZEROS - {description}")
                else:
                    print(f"  ✅ {field}: {sample_val} ({non_null_count}/{len(df)} valid) - {description}")

        # Fail test if critical fields are missing
        assert len(missing_critical) == 0, f"CRITICAL {sport.upper()} FIELDS MISSING: {missing_critical}"

        print(f"✅ All {len(critical_fields)} critical {sport.upper()} fields validated!")

    # Additional sport-specific columns (nice to have)
    additional_cols = {
        'mma': ['hteam', 'oteam', 'fight_card_position'],
        'nascar': ['track_type', 'dnf_pct', 'recent_avg_finish'],
        'nfl': ['snap_count', 'red_zone_targets', 'air_yards']
    }

    if sport in additional_cols:
        for col in additional_cols[sport]:
            if col in df.columns:
                print(f"  📊 Bonus field present: {col}")
            # Don't fail test for missing bonus fields

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