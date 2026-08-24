"""Why a generation failed, as the release records it.

The code chosen here goes into the release and into the Chapter 5 table, so it has to say what
happened rather than what the model server happened to write about it.

Until 25 August 2026 the classifier searched `str(error)` for the words "timeout", "grammar" and
"constraint". `call_vlm` wraps up to 200 characters of the server's own message inside the exception
it raises, so the reported code was decided by a substring search over text written by llama.cpp.
"""

from __future__ import annotations

import unittest

import requests

import support  # noqa: F401
from scripts import hdsg_runtime as hdsg  # noqa: E402
from scripts import realsense_vlm_on_change_qwen as client  # noqa: E402


def server_error(body: str) -> RuntimeError:
    """What `call_vlm` raises for an HTTP error, the server's message included."""
    return RuntimeError(f"VLM request failed: 500 {body}")


class ClassificationIsByType(unittest.TestCase):

    def test_a_timeout_is_reported_as_one(self):
        self.assertEqual("RG_GENERATION_TIMEOUT", hdsg.generation_failure_code(
            hdsg.GenerationTimeout("the model server did not answer in time")))

    def test_the_builtin_timeout_is_still_recognised(self):
        self.assertEqual("RG_GENERATION_TIMEOUT",
                         hdsg.generation_failure_code(TimeoutError()))

    def test_a_missing_constraint_is_reported_as_one(self):
        self.assertEqual("RG_CONSTRAINT_FAILURE", hdsg.generation_failure_code(
            hdsg.ConstraintUnavailable("The approved HDSG generation constraint is unavailable.")))

    def test_a_server_that_is_not_running_is_reported_as_unavailable(self):
        self.assertEqual("RG_MODEL_UNAVAILABLE", hdsg.generation_failure_code(
            requests.exceptions.ConnectionError("Max retries exceeded, actively refused it")))

    def test_a_reply_of_an_unexpected_shape_is_reported_as_unavailable(self):
        """`call_vlm` reads three nested keys out of the reply and guards none of them."""
        self.assertEqual("RG_MODEL_UNAVAILABLE", hdsg.generation_failure_code(KeyError("choices")))


class TheServersWordingDecidesNothing(unittest.TestCase):
    """The two failures that came out wrong, and the true positive given up to fix them.

    A server rejecting the constraint it was sent is no longer told apart from any other server
    error. That is a loss, and the cheaper one: a general code that is true beats a specific code
    that is wrong. Distinguishing it means reading the server's error contract rather than guessing
    at its wording, which is recorded in `LAB_SESSION_CHECKLIST.md` section E.
    """

    def test_an_out_of_memory_error_mentioning_a_timeout_flag_is_not_a_timeout(self):
        self.assertEqual("RG_MODEL_UNAVAILABLE", hdsg.generation_failure_code(
            server_error('{"error":"failed to allocate KV cache; try lowering --timeout"}')))

    def test_a_load_failure_whose_path_contains_grammar_is_not_a_constraint_failure(self):
        self.assertEqual("RG_MODEL_UNAVAILABLE", hdsg.generation_failure_code(
            server_error('{"error":"cannot open /models/grammar/../qwen.gguf"}')))

    def test_the_words_alone_no_longer_decide_anything(self):
        for body in ('{"error":"timeout"}', '{"error":"grammar"}', '{"error":"constraint"}'):
            with self.subTest(body):
                self.assertEqual("RG_MODEL_UNAVAILABLE",
                                 hdsg.generation_failure_code(server_error(body)))


class TheClientRaisesTheseTypes(unittest.TestCase):
    """A classifier keyed on type is worth nothing if the client raises something else."""

    class Args:
        model = "m"
        system = "s"
        temperature = 0.2
        top_p = 0.9
        max_tokens = 64
        _user_txt_for_payload = "text"

    def test_a_missing_constraint_raises_the_named_type(self):
        import numpy as np

        with self.assertRaises(hdsg.ConstraintUnavailable):
            client._build_payload_from_image_array(
                np.zeros((8, 8, 3), dtype=np.uint8), "jpeg", 70, self.Args(), grammar=None)

    def test_a_request_timeout_is_re_raised_as_the_named_type(self):
        original = client._VLM_SESSION.post

        def refuse(*args, **kwargs):
            raise requests.exceptions.ReadTimeout("Read timed out. (read timeout=45)")

        client._VLM_SESSION.post = refuse
        try:
            with self.assertRaises(hdsg.GenerationTimeout):
                client.call_vlm("http://localhost:8080", {}, timeout=1)
        finally:
            client._VLM_SESSION.post = original

    def test_an_http_error_is_not_re_raised_as_a_timeout(self):
        """Only the transport's own timeout counts. A 500 whose body mentions one does not."""
        original = client._VLM_SESSION.post

        class Response:
            status_code = 500
            text = '{"error":"read timeout while loading the grammar"}'

        client._VLM_SESSION.post = lambda *a, **k: Response()
        try:
            with self.assertRaises(RuntimeError) as caught:
                client.call_vlm("http://localhost:8080", {}, timeout=1)
            self.assertNotIsInstance(caught.exception, hdsg.GenerationTimeout)
            self.assertEqual("RG_MODEL_UNAVAILABLE",
                             hdsg.generation_failure_code(caught.exception))
        finally:
            client._VLM_SESSION.post = original


if __name__ == "__main__":
    unittest.main()
