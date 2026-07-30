#!/usr/bin/env python3
"""
Run the full physics & mathematics validation suite.

    python validate_physics.py
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import unittest

import test_physics_math  # noqa: F401
import test_ray_path_integrity  # noqa: F401


def run_all():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromModule(test_physics_math))
    suite.addTests(loader.loadTestsFromModule(test_ray_path_integrity))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(run_all())
