"""Test setup_data.py script functionality for any sport."""

import pytest
import sys
import subprocess
import shutil
import argparse
from pathlib import Path


def test_setup_data(sport: str = "mma"):
    """Test setup_data.py creates directories and provides URLs for any sport."""
    # Always use PID 99 for golden tests
    pid = "99"
    test_path = Path(f"data/{sport}/{pid}/dk")

    # Clean up any existing test data
    if test_path.exists():
        shutil.rmtree(test_path)

    # Run setup_data.py for specified sport
    result = subprocess.run([
        sys.executable, "src/cli/setup_data.py",
        "--sport", sport,
        "--pid", pid,
        "--site", "dk"
    ], capture_output=True, text=True)

    # Check command succeeded
    assert result.returncode == 0, f"setup_data.py failed: {result.stderr}"

    # Verify directories were created
    assert (test_path / "json").exists(), "json directory not created"
    assert (test_path / "csv").exists(), "csv directory not created"
    assert (test_path / "newsletters").exists(), "newsletters directory not created"

    # Verify empty file was created
    assert (test_path / "json" / "raw.json").exists(), "empty raw.json not created"

    # Check that output contains URL information
    output = result.stdout
    assert f"{sport.upper()}" in output, f"{sport.upper()} not found in output"
    assert "99" in output, "PID 99 not substituted in URLs"

    # Sport-specific validations
    if sport in ['mma', 'nascar', 'nfl']:
        assert "LineStar" in output, "LineStar URL not displayed"
        assert "https://www.linestarapp.com" in output, "LineStar URL not found"

    # Copy existing real data if available for subsequent tests
    source_paths = {
        'mma': "data/mma/466/dk",
        'nascar': "data/nascar/503/dk",
        'nfl': "data/nfl/382/dk"
    }

    source_path = Path(source_paths.get(sport, f"data/{sport}/test/dk"))

    if source_path.exists():
        # Copy main data file
        if (source_path / "json/raw.json").exists():
            shutil.copy(
                source_path / "json/raw.json",
                test_path / "json/raw.json"
            )

        # Copy newsletter file (try different possible names)
        newsletter_files = ["linestar.txt", "newsletter_signals.json"]
        for newsletter_file in newsletter_files:
            source_newsletter = source_path / "newsletters" / newsletter_file
            if source_newsletter.exists():
                shutil.copy(
                    source_newsletter,
                    test_path / "newsletters" / newsletter_file
                )
                break

    print(f"✅ {sport.upper()} setup test passed - directories created at {test_path}")
    return test_path


# Legacy function for backward compatibility
def test_setup_data_mma():
    """Test MMA setup_data.py (legacy function)."""
    return test_setup_data("mma")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test setup_data.py for any sport")
    parser.add_argument("--sport", default="mma", choices=["mma", "nascar", "nfl", "nba"],
                       help="Sport to test")

    args = parser.parse_args()
    test_setup_data(args.sport)