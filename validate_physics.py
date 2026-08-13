#!/usr/bin/env python3
"""
Run the full physics & mathematics validation suite.

    python validate_physics.py
    python validate_physics.py --coverage
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))


def run_all(*, with_coverage: bool = False):
    if with_coverage:
        try:
            import coverage
        except ImportError:
            print("coverage package not installed; pip install coverage")
            return 2
        cov = coverage.Coverage(config_file=os.path.join(ROOT, ".coveragerc"))
        cov.start()

    import unittest

    import test_physics_math  # noqa: F401
    import test_physics_correctness  # noqa: F401
    import test_blockers  # noqa: F401
    import test_optimizer_spill  # noqa: F401
    import test_view3d  # noqa: F401
    import test_ray_path_integrity  # noqa: F401
    import test_features_extended  # noqa: F401
    import test_coverage_gaps  # noqa: F401
    import test_optimizer_panel  # noqa: F401
    import test_cad_tessellation  # noqa: F401
    import test_ui_live_edit  # noqa: F401
    import test_copy_element  # noqa: F401
    import test_target_zoom  # noqa: F401
    import test_cad_flange  # noqa: F401
    import test_laser_calibrate  # noqa: F401
    import test_design_io  # noqa: F401

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromModule(test_physics_math))
    suite.addTests(loader.loadTestsFromModule(test_physics_correctness))
    suite.addTests(loader.loadTestsFromModule(test_blockers))
    suite.addTests(loader.loadTestsFromModule(test_optimizer_spill))
    suite.addTests(loader.loadTestsFromModule(test_view3d))
    suite.addTests(loader.loadTestsFromModule(test_ray_path_integrity))
    suite.addTests(loader.loadTestsFromModule(test_features_extended))
    suite.addTests(loader.loadTestsFromModule(test_coverage_gaps))
    suite.addTests(loader.loadTestsFromModule(test_optimizer_panel))
    suite.addTests(loader.loadTestsFromModule(test_cad_tessellation))
    suite.addTests(loader.loadTestsFromModule(test_ui_live_edit))
    suite.addTests(loader.loadTestsFromModule(test_copy_element))
    suite.addTests(loader.loadTestsFromModule(test_target_zoom))
    suite.addTests(loader.loadTestsFromModule(test_cad_flange))
    suite.addTests(loader.loadTestsFromModule(test_laser_calibrate))
    suite.addTests(loader.loadTestsFromModule(test_design_io))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if with_coverage:
        cov.stop()
        cov.save()
        print("\n── Coverage (non-GUI modules) ──")
        total = cov.report(show_missing=False)
        print(f"TOTAL coverage: {total:.1f}%")
        if total < 75.0:
            print(
                f"WARNING: coverage {total:.1f}% is below the 75% target "
                "(app.py + optional warp_backend.py excluded)."
            )

    print(
        f"\nSummary: ran {result.testsRun} tests — "
        f"{len(result.failures)} failed, {len(result.errors)} errors, "
        f"{len(result.skipped)} skipped"
    )
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="OptiFlux validation suite")
    ap.add_argument(
        "--coverage",
        action="store_true",
        help="Measure line coverage of non-GUI modules (needs 'coverage' package)",
    )
    args = ap.parse_args()
    raise SystemExit(run_all(with_coverage=args.coverage))
