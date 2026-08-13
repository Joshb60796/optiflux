"""
Live pointer interaction: expensive traces must not start while a slider
or plot handle is held; they start only after release.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import LiveEditGate, should_start_trace


class TestLiveEditGate(unittest.TestCase):
    def test_begin_blocks_trace(self):
        g = LiveEditGate()
        self.assertFalse(g.live)
        self.assertTrue(should_start_trace(auto_run=True, live=g.live, handle_drag=False))
        g.begin()
        self.assertTrue(g.live)
        self.assertFalse(should_start_trace(auto_run=True, live=g.live, handle_drag=False))

    def test_end_allows_trace(self):
        g = LiveEditGate()
        g.begin()
        closed = g.end()
        self.assertTrue(closed)
        self.assertFalse(g.live)
        self.assertTrue(should_start_trace(auto_run=True, live=g.live, handle_drag=False))

    def test_begin_increments_cancel_generation(self):
        g = LiveEditGate()
        a = g.begin()
        b = g.begin()
        self.assertGreater(b, a)
        g.end()
        g.end()
        self.assertFalse(g.live)

    def test_handle_drag_blocks_even_when_not_slider(self):
        self.assertFalse(
            should_start_trace(auto_run=True, live=False, handle_drag=True)
        )

    def test_auto_run_off_never_traces(self):
        self.assertFalse(
            should_start_trace(auto_run=False, live=False, handle_drag=False)
        )

    def test_end_when_not_live_is_safe(self):
        g = LiveEditGate()
        self.assertFalse(g.end())
        self.assertFalse(g.live)


if __name__ == "__main__":
    unittest.main()
