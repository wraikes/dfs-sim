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
    
    # Verify json directory was created
    assert (test_path / "json").exists(), "json directory not created"
    
    # Verify empty file was created
    assert (test_path / "json" / "99.json").exists(), "empty 99.json not created"
    
    # Check that output contains URL information
    output = result.stdout
    assert "LineStar Contest Data" in output, "LineStar URL not displayed"
    assert "https://www.linestarapp.com" in output, "LineStar URL not found"
    assert "99" in output, "PID not substituted in URLs"
    
    # Copy 466 json files to 99 for subsequent tests
    source_path = Path("data/mma/dk/466")
    
    # Copy main data file
    if (source_path / "json/466.json").exists():
        shutil.copy(
            source_path / "json/466.json",
            test_path / "json/99.json"
        )
    
    # Verify test file was copied
    assert (test_path / "json/99.json").exists(), "Test data file not copied"
    
    print(f"✅ Setup test passed - directories created and test data copied to {test_path}")


if __name__ == "__main__":
    test_setup_data_mma()