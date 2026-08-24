"""The browser sink: what it does when the walker stops listening.

The module docstring says publishing is "non-blocking and lossy by design: a browser that cannot
keep up misses intermediate frames rather than delaying the sensing loop behind it". That held for
the frame channel and not for the input channel, which used a blocking put on a queue of 64.

These tests run a real server on a loopback port. Nothing here needs a camera.
"""

from __future__ import annotations

import json
import time
import unittest
import urllib.error
import urllib.request

import support  # noqa: F401
from scripts.hdsg_web_ui import WebInterface  # noqa: E402

PORT = 8407


def post(port, payload, timeout=3.0):
    """Returns (status, body, seconds). A hang shows as the timeout it took to give up."""
    request = urllib.request.Request(f"http://127.0.0.1:{port}/input",
                                     data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read(), time.monotonic() - started
    except urllib.error.HTTPError as error:
        return error.code, error.read(), time.monotonic() - started
    except Exception as error:  # a hang arrives here as a socket timeout
        return type(error).__name__, b"", time.monotonic() - started


class AFullInputQueueRefusesRatherThanHangs(unittest.TestCase):
    """Every control on the page posts to one endpoint, so a blocked put froze the whole page.

    The way in is ordinary rather than exotic. The loop stops draining, a press does nothing, so
    the person presses again, and sixty-four presses is not many when a button looks dead.
    """

    def setUp(self):
        self.ui = WebInterface(host="127.0.0.1", port=PORT).start()
        for _ in range(50):
            if post(PORT, {"probe": True})[0] == 200:
                break
            time.sleep(0.05)
        self.ui.poll_events()

    def tearDown(self):
        self.ui.stop()

    def fill(self):
        for index in range(64):
            post(PORT, {"n": index})

    def test_a_press_beyond_the_queue_is_refused_promptly(self):
        self.fill()
        status, body, seconds = post(PORT, {"intent": "NONE"})
        self.assertEqual(503, status)
        self.assertLess(seconds, 1.0, "the request hung rather than answering")
        self.assertFalse(json.loads(body)["ok"])

    def test_the_refusal_says_why(self):
        self.fill()
        _, body, _ = post(PORT, {"intent": "NONE"})
        self.assertEqual("input queue full", json.loads(body)["reason"])

    def test_refusals_are_counted(self):
        self.fill()
        post(PORT, {"intent": "NONE"})
        post(PORT, {"intent": "NONE"})
        self.assertEqual(2, self.ui.dropped_inputs)

    def test_the_queue_recovers_once_the_loop_drains_it(self):
        """A dropped press the person can repeat is the point; a permanently dead page is not."""
        self.fill()
        self.assertEqual(503, post(PORT, {"intent": "NONE"})[0])
        self.assertEqual(64, len(self.ui.poll_events()))
        self.assertEqual(200, post(PORT, {"intent": "FORWARD"})[0])

    def test_an_ordinary_press_is_accepted_and_reaches_the_loop(self):
        post(PORT, {"intent": "LEFT"})
        self.assertIn({"intent": "LEFT"}, self.ui.poll_events())

    def test_nothing_is_dropped_in_ordinary_use(self):
        for _ in range(5):
            post(PORT, {"intent": "LEFT"})
            self.ui.poll_events()
        self.assertEqual(0, self.ui.dropped_inputs)


class TheBrowserIsToldWhenAPressIsRefused(unittest.TestCase):
    """Asserted against the page, which cannot be driven without a browser.

    Saying nothing was worse than it sounds: the press vanished, so the natural response was to
    press again, which is the pressing that filled the queue.
    """

    def setUp(self):
        self.page = (support.ROOT / "web" / "index.html").read_text(encoding="utf-8")

    def test_the_response_status_is_read(self):
        self.assertIn("if (!response.ok)", self.page)
        self.assertNotIn(".catch(() => {});", self.page)

    def test_a_refusal_produces_a_message(self):
        self.assertIn("The walker did not take that press", self.page)

    def test_the_message_takes_precedence_over_the_run_notice(self):
        self.assertIn("(Date.now() < localNoticeUntil && localNotice) || state.notice",
                      self.page)


class BindingBeyondLoopbackIsAnnounced(unittest.TestCase):
    """No endpoint carries authentication, so a non-loopback host puts the camera and the walker's
    controls on the network. The default is safe and the flag is not restricted; doing it silently
    is what is not acceptable."""

    def test_a_loopback_host_says_nothing(self):
        import io
        import contextlib

        ui = WebInterface(host="127.0.0.1", port=PORT + 1)
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            ui.start()
        ui.stop()
        self.assertEqual("", printed.getvalue())

    def test_any_other_host_warns(self):
        import io
        import contextlib

        ui = WebInterface(host="0.0.0.0", port=PORT + 2)
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            ui.start()
        ui.stop()
        self.assertIn("reachable by anyone on this network", printed.getvalue())


if __name__ == "__main__":
    unittest.main()
