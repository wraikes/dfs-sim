"""Test setup_data.py script functionality."""

import pytest
import sys
import subprocess
import shutil
from pathlib import Path


def test_setup_data_mma():
    """Test MMA setup_data.py creates directories and provides URLs."""
    # Clean up any existing test data
    test_path = Path("data/mma/99/dk")
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
    
    # Verify directories were created
    assert (test_path / "json").exists(), "json directory not created"
    assert (test_path / "csv").exists(), "csv directory not created"
    assert (test_path / "newsletters").exists(), "newsletters directory not created"
    
    # Verify empty file was created
    assert (test_path / "json" / "raw.json").exists(), "empty raw.json not created"
    
    # Check that output contains URL information
    output = result.stdout
    assert "LineStar Contest Data" in output, "LineStar URL not displayed"
    assert "https://www.linestarapp.com" in output, "LineStar URL not found"
    assert "99" in output, "PID not substituted in URLs"
    
    # Copy 466 data files to 99 for subsequent tests
    source_path = Path("data/mma/466/dk")
    
    # Copy main data file
    if (source_path / "json/raw.json").exists():
        shutil.copy(
            source_path / "json/raw.json",
            test_path / "json/raw.json"
        )
    
    # Copy newsletter file
    if (source_path / "newsletters/linestar.txt").exists():
        shutil.copy(
            source_path / "newsletters/linestar.txt",
            test_path / "newsletters/linestar.txt"
        )
    
    # Verify test files were copied
    assert (test_path / "json/raw.json").exists(), "Test data file not copied"
    assert (test_path / "newsletters/linestar.txt").exists(), "Newsletter file not copied"
    
    print(f"✅ Setup test passed - directories created and test data copied to {test_path}")


if __name__ == "__main__":
    test_setup_data_mma()