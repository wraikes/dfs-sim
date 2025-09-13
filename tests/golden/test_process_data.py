"""Test process_data.py script functionality."""

import pytest
import sys
import subprocess
import pandas as pd
from pathlib import Path


def test_process_data_mma():
    """Test MMA process_data.py processes raw data and generates CSV."""
    test_path = Path("data/mma/99/dk")
    
    # Verify test data exists (from setup_data test)
    raw_file = test_path / "json/raw.json"
    newsletter_file = test_path / "newsletters/linestar.txt"
    
    assert raw_file.exists(), f"Test data not found: {raw_file} - run test_setup_data.py first"
    assert newsletter_file.exists(), f"Newsletter data not found: {newsletter_file}"
    
    # Run process_data.py
    result = subprocess.run([
        sys.executable, "src/cli/process_data.py",
        "--sport", "mma",
        "--pid", "99",
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
    
    # Required columns
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
    
    # Derived metrics columns (our calculated features)
    derived_cols = [
        'finishing_rate', 'style_score', 'matchup_advantage', 'takedown_matchup',
        'strikes_per_fight', 'strike_accuracy', 'takedowns_per_fight', 'itd_probability'
    ]
    for col in derived_cols:
        assert col in df.columns, f"Missing derived metric column: {col}"
    
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
    if len(non_neutral) > 0:
        # Should have confidence values
        assert non_neutral['newsletter_confidence'].notna().all(), "Newsletter confidence missing"
        assert (non_neutral['newsletter_confidence'] > 0).all(), "Newsletter confidence should be positive"
    
    print(f"✅ Process data test passed - {len(df)} players processed with newsletter signals")
    print(f"   Newsletter signals: {dict(signal_counts)}")
    

def test_process_data_summary():
    """Test process_data.py summary display."""
    # Run summary-only mode
    result = subprocess.run([
        sys.executable, "src/cli/process_data.py",
        "--sport", "mma", 
        "--pid", "99",
        "--site", "dk",
        "--summary-only"
    ], capture_output=True, text=True)
    
    assert result.returncode == 0, f"Summary mode failed: {result.stderr}"
    
    # Check summary contains expected information
    output = result.stdout
    assert "MMA DATA SUMMARY" in output, "Summary header missing"
    assert "Total Players:" in output, "Player count missing"
    assert "Salary Range:" in output, "Salary info missing"
    assert "Newsletter Signals:" in output, "Newsletter summary missing"
    
    print("✅ Process data summary test passed")


if __name__ == "__main__":
    test_process_data_mma()
    test_process_data_summary()