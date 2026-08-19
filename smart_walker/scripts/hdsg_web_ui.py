"""Local web display and input surface for the HDSG smart-walker prototype.

This module replaces the OpenCV window as the display and input sink. It renders nothing
itself: the browser receives the unannotated camera frame on one channel and the structured
deterministic state on another, and draws the sector bands, object boxes, badges and caption
from that structure. The pixels the server sends carry no overlay, so the browser is working
from the same values the telemetry records rather than from a rasterised picture of them.

Three endpoints, all served from the standard library:

    /               the single page
    /stream.mjpg    the camera frame, multipart/x-mixed-replace
    /events         structured state, text/event-stream
    /input          browser input, POST, JSON body

Nothing here participates in the sensing, deterministic decision, generation or release path.
The only release field it ever receives is `content.caption_text`, which is the sole field the
architecture permits to be displayed.
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from queue import Empty, Queue
import threading
import time
from typing import Any, Optional


PAGE_PATH = Path(__file__).resolve().parents[1] / "web" / "index.html"

# Boundary token for the multipart camera stream. Any token works provided the header and the
# part separators agree; this one is spelled out rather than generated so a packet capture is
# readable during debugging.
_MJPEG_BOUNDARY = "hdsgframe"


class _State:
    """Holds the latest frame and state for however many browsers are connected.

    Readers take the most recent value rather than a queued history: a browser that falls
    behind should show the current situation, not replay an old one. This matters for a
    safety display, where a stale frame presented as current is the failure to avoid.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: Optional[bytes] = None
        self._frame_sequence = 0
        self._state: dict = {}
        self._state_encoded: Optional[str] = None
        self._state_sequence = 0
        self._chat_log: list[dict] = []

    def set_frame(self, jpeg: bytes) -> None:
        with self._lock:
            self._frame = jpeg
            self._frame_sequence += 1

    def get_frame(self) -> tuple[Optional[bytes], int]:
        with self._lock:
            return self._frame, self._frame_sequence

    def set_state(self, value: dict) -> None:
        """Records the state, and bumps the sequence only when something actually changed.

        The caller publishes on every pass of its loop, which runs far faster than the sensing
        rate. Comparing the encoded form here means the event stream carries an update per
        genuine change (roughly the inference rate) rather than per loop iteration, and falls
        silent when the scene is static. Encoding happens outside the lock.
        """
        encoded = json.dumps(value, separators=(",", ":"), ensure_ascii=True, sort_keys=True)
        with self._lock:
            if encoded == self._state_encoded:
                return
            self._state = value
            self._state_encoded = encoded
            self._state_sequence += 1

    def append_chat(self, turn: dict) -> None:
        with self._lock:
            self._chat_log.append(turn)
            del self._chat_log[:-50]
            self._state_sequence += 1

    def get_state(self) -> tuple[dict, int]:
        with self._lock:
            payload = dict(self._state)
            payload["chat_log"] = list(self._chat_log)
            return payload, self._state_sequence


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @property
    def _shared(self) -> _State:
        return self.server.shared_state

    @property
    def _inbound(self) -> "Queue[dict]":
        return self.server.inbound

    def log_message(self, format: str, *args: Any) -> None:
        """Silences the per-request stderr logging that would flood the run console."""

    def do_GET(self) -> None:
        route = self.path.split("?", 1)[0]
        if route == "/":
            self._serve_page()
        elif route == "/stream.mjpg":
            self._serve_stream()
        elif route == "/events":
            self._serve_events()
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] != "/input":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
            self.send_error(400)
            return
        if isinstance(payload, dict):
            self._inbound.put(payload)
        body = b'{"ok":true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_page(self) -> None:
        try:
            body = PAGE_PATH.read_bytes()
        except OSError:
            self.send_error(500, "The interface page could not be read.")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={_MJPEG_BOUNDARY}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last_sequence = -1
        try:
            while not self.server.stop_event.is_set():
                frame, sequence = self._shared.get_frame()
                if frame is None or sequence == last_sequence:
                    time.sleep(0.01)
                    continue
                last_sequence = sequence
                header = (
                    f"--{_MJPEG_BOUNDARY}\r\n"
                    f"Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(frame)}\r\n\r\n"
                ).encode("ascii")
                self.wfile.write(header)
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def _serve_events(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        last_sequence = -1
        try:
            while not self.server.stop_event.is_set():
                state, sequence = self._shared.get_state()
                if sequence == last_sequence:
                    time.sleep(0.02)
                    continue
                last_sequence = sequence
                message = json.dumps(state, separators=(",", ":"), ensure_ascii=True)
                self.wfile.write(f"data: {message}\n\n".encode("utf-8"))
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler, shared_state: _State, inbound: "Queue[dict]",
                 stop_event: threading.Event):
        super().__init__(address, handler)
        self.shared_state = shared_state
        self.inbound = inbound
        self.stop_event = stop_event


class WebInterface:
    """Serves the interface and collects browser input.

    The caller publishes the camera frame and the structured state, and drains input events
    on its own loop. Publishing is non-blocking and lossy by design: a browser that cannot
    keep up misses intermediate frames rather than delaying the sensing loop behind it.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8321,
                 max_frame_hz: float = 15.0) -> None:
        self.host = host
        self.port = port
        self._minimum_frame_interval = 1.0 / max(max_frame_hz, 1.0)
        self._last_frame_published = 0.0
        self._shared = _State()
        self._inbound: "Queue[dict]" = Queue(maxsize=64)
        self._stop_event = threading.Event()
        self._server: Optional[_Server] = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def start(self) -> "WebInterface":
        self._server = _Server(
            (self.host, self.port), _Handler, self._shared, self._inbound, self._stop_event
        )
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        self._stop_event.set()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    def should_publish_frame(self) -> bool:
        """Reports whether enough time has passed to justify encoding another frame.

        The caller checks this before running the JPEG encode, so a display rate lower than
        the loop rate costs no encoding work rather than encoding and discarding.
        """
        now = time.monotonic()
        if now - self._last_frame_published < self._minimum_frame_interval:
            return False
        self._last_frame_published = now
        return True

    def publish_frame(self, jpeg: bytes) -> None:
        self._shared.set_frame(jpeg)

    def publish_state(self, state: dict) -> None:
        self._shared.set_state(state)

    def publish_chat_turn(self, question: str, answer: str, route: Optional[str] = None) -> None:
        self._shared.append_chat({
            "question": question,
            "answer": answer,
            "route": route,
            "at": time.strftime("%H:%M:%S"),
        })

    def poll_events(self) -> list[dict]:
        """Returns the browser input received since the previous call."""
        events: list[dict] = []
        while True:
            try:
                events.append(self._inbound.get_nowait())
            except Empty:
                break
        return events
