"""Pytest bootstrap for the test suite.

tests/test_crowd.py uses unittest-style ``self.assertIsNotNone`` assertions
without subclassing ``unittest.TestCase``.  Copy the TestCase assertion API
onto the class at collection time so the test file itself stays untouched.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_crowd import TestCrowdSystem  # noqa: E402

if not issubclass(TestCrowdSystem, unittest.TestCase):
    for name in dir(unittest.TestCase):
        if not hasattr(TestCrowdSystem, name):
            setattr(TestCrowdSystem, name, getattr(unittest.TestCase, name))
