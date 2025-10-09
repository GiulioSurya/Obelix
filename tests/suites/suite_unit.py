"""
Test Suite for All Unit Tests

Executes all unit tests across all modules.
Following best practices from PPT Slide 12-13.

Usage:
    # Run all unit tests
    pytest tests/suites/suite_unit.py -v

    # Or run directly
    python tests/suites/suite_unit.py
"""

import pytest
import sys
from pathlib import Path

# Ensure src is in path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))


def run_unit_tests():
    """
    Execute all unit tests.

    Includes:
    - Message tests
    - Tool tests
    - Provider tests
    - Mapping tests
    """
    pytest.main([
        "tests/unit/",
        "-v",  # Verbose output
        "--tb=short",  # Short traceback
        "-s",  # Show print statements
        "--color=yes"  # Colored output
    ])


if __name__ == "__main__":
    print("=" * 80)
    print("UNIT TESTS SUITE")
    print("=" * 80)
    print()
    run_unit_tests()