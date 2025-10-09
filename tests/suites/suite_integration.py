"""
Test Suite for Integration Tests

Executes all integration tests that verify multiple components working together.
Following best practices from PPT Slide 12-13.

Usage:
    # Run all integration tests
    pytest tests/suites/suite_integration.py -v

    # Or run directly
    python tests/suites/suite_integration.py
"""

import pytest
import sys
from pathlib import Path

# Ensure src is in path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))


def run_integration_tests():
    """
    Execute all integration tests.

    Includes:
    - BaseAgent integration tests
    - Tool execution integration tests
    - Provider integration tests
    - Multi-component workflows
    """
    pytest.main([
        "tests/integration/",
        "-v",  # Verbose output
        "--tb=short",  # Short traceback
        "-s",  # Show print statements
        "--color=yes"  # Colored output
    ])


if __name__ == "__main__":
    print("=" * 80)
    print("INTEGRATION TESTS SUITE")
    print("=" * 80)
    print()
    run_integration_tests()