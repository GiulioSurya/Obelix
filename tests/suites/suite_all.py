"""
Test Suite for All Tests

Executes complete test suite: unit tests + integration tests.
Following best practices from PPT Slide 12-13.

Usage:
    # Run all tests
    pytest tests/suites/suite_all.py -v

    # Or run directly
    python tests/suites/suite_all.py

    # Run with coverage report
    pytest tests/suites/suite_all.py --cov=src --cov-report=html
"""

import pytest
import sys
from pathlib import Path

# Ensure src is in path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))


def run_all_tests():
    """
    Execute complete test suite.

    Includes:
    - All unit tests (messages, tools, providers, mapping)
    - All integration tests (agents, workflows)
    """
    pytest.main([
        "tests/",
        "-v",  # Verbose output
        "--tb=short",  # Short traceback
        "-s",  # Show print statements
        "--color=yes",  # Colored output
        "--ignore=tests/suites"  # Don't recurse into suite files
    ])


def run_all_tests_with_coverage():
    """
    Execute complete test suite with coverage report.
    """
    pytest.main([
        "tests/",
        "-v",
        "--tb=short",
        "-s",
        "--color=yes",
        "--ignore=tests/suites",
        "--cov=src",  # Coverage for src directory
        "--cov-report=html",  # HTML report
        "--cov-report=term"  # Terminal report
    ])


if __name__ == "__main__":
    print("=" * 80)
    print("COMPLETE TEST SUITE")
    print("=" * 80)
    print()
    print("Running all unit and integration tests...")
    print()

    # Run tests with coverage
    run_all_tests_with_coverage()