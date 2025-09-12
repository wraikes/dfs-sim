"""Test setup_data.py script functionality."""

import pytest
import sys
import subprocess
import shutil
from pathlib import Path


def test_setup_data_mma():
    """Test MMA setup_data.py creates directories and provides URLs."""
    # Clean up any existing test data
    test_path = Path("data/mma/dk/99")
    if test_path.exists():
        shutil.rmtree(test_path)
    
    # Run setup_data.py for MMA pid 99
    result = subprocess.run([
        sys.executable, "src/cli/setup_data.py", 
        "--sport", "mma", 
        "--pid", "99",
        "--site", "dk"
    ], capture_output=True, text=True)
    
    # Check command succeeded
    assert result.returncode == 0, f"setup_data.py failed: {result.stderr}"
    
    # Verify raw directory was created
    assert (test_path / "raw").exists(), "raw directory not created"
    
    # Verify empty files were created
    assert (test_path / "raw" / "dk_99.json").exists(), "empty dk_99.json not created"
    assert (test_path / "raw" / "newsletter_signals.json").exists(), "empty newsletter_signals.json not created"
    
    # Check that output contains URL information
    output = result.stdout
    assert "LineStar Contest Data" in output, "LineStar URL not displayed"
    assert "https://www.linestarapp.com" in output, "LineStar URL not found"
    assert "99" in output, "PID not substituted in URLs"
    
    # Copy 466 raw files to 99 for subsequent tests
    source_path = Path("data/mma/dk/466")
    
    # Copy main data file
    if (source_path / "json/dk_466.json").exists():
        shutil.copy(
            source_path / "json/dk_466.json",
            test_path / "raw/dk_99.json"
        )
    
    # Copy newsletter signals
    if (source_path / "json/linestar_newsletter_signals.json").exists():
        shutil.copy(
            source_path / "json/linestar_newsletter_signals.json", 
            test_path / "raw/newsletter_signals.json"
        )
    
    # Verify test files were copied
    assert (test_path / "raw/dk_99.json").exists(), "Test data file not copied"
    assert (test_path / "raw/newsletter_signals.json").exists(), "Newsletter signals not copied"
    
    print(f"✅ Setup test passed - directories created and test data copied to {test_path}")


def test_setup_data_validate():
    """Test validation of uploaded files."""
    # This assumes test_setup_data_mma has run first
    test_path = Path("data/mma/dk/99")
    assert test_path.exists(), "Test data directory doesn't exist - run test_setup_data_mma first"
    
    # Run validation
    result = subprocess.run([
        sys.executable, "src/cli/setup_data.py",
        "--sport", "mma",
        "--pid", "99",
        "--site", "dk",
        "--validate-only"
    ], capture_output=True, text=True)
    
    assert result.returncode == 0, f"Validation failed: {result.stderr}"
    assert "✅ All required files found!" in result.stdout, "File validation failed"
    
    print("✅ Validation test passed")


if __name__ == "__main__":
    test_setup_data_mma()
    test_setup_data_validate()