"""The keyboard, which is the only way the person expresses an intent to the walker.

A lost press is not visible in any record: the walker simply does not respond, and the person presses
again. There is nothing in the archive that would show it happened, so the property is asserted here
instead.

The listener polls the Windows keyboard API, so `key_down` is replaced by a script of which keys are
held on each tick. What is exercised is the bookkeeping, which is where the fault was.
"""

from __future__ import annotations

import unittest

from support import ROOT  # noqa: F401
from scripts import realsense_shared_control as sw  # noqa: E402


class EdgeDetection(unittest.TestCase):
    """One tick at a time, with the keys held on that tick named explicitly."""

    def setUp(self):
        self.listener = sw.KeyListener()
        self.held: set[str] = set()
        self.original = sw.key_down
        codes = {code: name for name, code in sw.VK.items()}
        sw.key_down = lambda vk: codes[vk] in self.held
        self.addCleanup(setattr, sw, "key_down", self.original)

    def tick(self, *held):
        self.held = set(held)
        return {name for name, fired in self.listener.edges().items() if fired}

    def test_a_press_fires_once_and_not_again_while_held(self):
        self.assertEqual({"M"}, self.tick("M"))
        self.assertEqual(set(), self.tick("M"))
        self.assertEqual(set(), self.tick())
        self.assertEqual({"M"}, self.tick("M"))

    def test_two_keys_pressed_on_one_tick_both_fire(self):
        """The direction and the request are separate things, and pressing both at once is ordinary.
        A short-circuiting chain reported only the first."""
        self.assertEqual({"W", "M"}, self.tick("W", "M"))

    def test_a_press_is_not_lost_when_released_as_another_key_fires(self):
        """The fault this test exists for, found on 24 August 2026.

        The release must fall on the very tick another key is newly pressed. The chain stopped at
        that key and never updated M's stored state, so M was still recorded as held, and the next
        press read as no change and was discarded.
        """
        self.assertEqual({"M"}, self.tick("M"))          # pressed, and held
        self.assertEqual({"W"}, self.tick("W"))          # M released as W is pressed, one tick
        self.assertEqual({"M"}, self.tick("W", "M"))     # pressed again, and it must register

    def test_a_release_on_a_quiet_tick_was_always_recorded(self):
        """The boundary, which is why the fault was hard to hit. With no other key firing, the loop
        reached M and updated its state, so the next press registered even under the old chain.
        Kept to mark where the fault started, so a later change cannot be judged against the tight
        sequence alone."""
        self.assertEqual({"M"}, self.tick("M"))
        self.assertEqual({"W"}, self.tick("W", "M"))     # W newly pressed, M still held
        self.assertEqual(set(), self.tick("W"))          # M released, nothing else firing
        self.assertEqual({"M"}, self.tick("W", "M"))

    def test_every_polled_key_is_reported(self):
        """A key added to VK and forgotten in the loop is silently inert, which is how Y, N and P
        survived after nothing read them."""
        self.assertEqual(set(sw.VK), set(self.listener.edges()))


class PolledKeys(unittest.TestCase):

    def test_only_the_keys_the_client_acts_on_are_polled(self):
        """Y, N and P were polled until 24 August 2026 and read by nothing."""
        self.assertEqual({"W", "A", "S", "D", "SPACE", "Q", "M", "R"}, set(sw.VK))

    def test_the_state_carries_only_what_the_client_reads(self):
        self.assertEqual(
            {"last_edge_dir", "last_press_ms", "quit_requested", "more_detail_edge",
             "reassess_edge"},
            set(sw.IntentState().__dict__))


if __name__ == "__main__":
    unittest.main()
