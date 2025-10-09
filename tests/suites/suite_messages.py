"""
Test Suite for Message System

Executes all message-related tests in the correct order.
Following best practices from PPT Slide 12-13.

Usage:
    # Run entire message suite
    pytest tests/suites/suite_messages.py -v

    # Or run directly
    python tests/suites/suite_messages.py
"""

import pytest
import sys
from pathlib import Path

# Ensure src is in path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))


def run_message_tests():
    """
    Execute all message unit tests.

    Tests are organized by message type for clarity and isolation.
    """
    pytest.main([
        "tests/unit/messages/",
        "-v",  # Verbose output
        "--tb=short",  # Short traceback
        "-s",  # Show print statements
        "--color=yes"  # Colored output
    ])


if __name__ == "__main__":
    print("=" * 80)
    print("MESSAGE SYSTEM TEST SUITE")
    print("=" * 80)
    print()
    run_message_tests()