from __future__ import annotations
import ctypes
import threading
import time
from dataclasses import dataclass

# Windows virtual-key codes
VK = {'W': 0x57, 'A': 0x41, 'S': 0x53, 'D': 0x44,
      'SPACE': 0x20, 'Q': 0x51, 'P': 0x50}
user32 = ctypes.windll.user32


def key_down(vk: int) -> bool:
    # MSB set => key currently down (instantaneous polling)
    return (user32.GetAsyncKeyState(vk) & 0x8000) != 0


@dataclass
class IntentState:
    last_dir: str = "idle"          # "forward"|"backward"|"left"|"right"|"idle"
    # time (ms) when that dir was last edge-pressed
    last_press_ms: int = 0
    quit_requested: bool = False    # Q pressed
    # Print handling (rate-limited in main):
    p_is_down: bool = False


class KeyIntentListener:
    """
    Background thread that watches W/A/S/D/SPACE with **edge detection**.
    - Only updates state when key goes from UP -> DOWN (prevents "held key" flooding).
    - Q sets quit_requested.
    - P state is exposed (main can rate-limit prints).
    """

    def __init__(self, poll_hz: int = 100):
        self.poll_interval = 1.0 / float(max(1, poll_hz))
        self.state = IntentState()
        self._prev = {k: False for k in VK}  # previous key-down state
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> "KeyIntentListener":
        self._thr.start()
        return self

    def stop(self):
        self._stop.set()
        try:
            self._thr.join(timeout=1.0)
        except Exception:
            pass

    def _edge_down(self, vk_code: int, key_name: str) -> bool:
        now_is_down = key_down(vk_code)
        was_down = self._prev[key_name]
        self._prev[key_name] = now_is_down
        return (not was_down) and now_is_down

    def _loop(self):
        while not self._stop.is_set():
            now_ms = int(time.time() * 1000)

            # Edge-detect directional keys
            if self._edge_down(VK['W'], 'W'):
                self._set_intent("forward",  now_ms)
            elif self._edge_down(VK['S'], 'S'):
                self._set_intent("backward", now_ms)
            elif self._edge_down(VK['A'], 'A'):
                self._set_intent("left",     now_ms)
            elif self._edge_down(VK['D'], 'D'):
                self._set_intent("right",    now_ms)
            elif self._edge_down(VK['SPACE'], 'SPACE'):
                self._set_intent("idle", now_ms)

            # Q to request quit (edge)
            if self._edge_down(VK['Q'], 'Q'):
                with self._lock:
                    self.state.quit_requested = True

            # Track P (level) so main can rate-limit how it prints
            with self._lock:
                self.state.p_is_down = key_down(VK['P'])

            time.sleep(self.poll_interval)

    def _set_intent(self, dir_name: str, now_ms: int):
        with self._lock:
            self.state.last_dir = dir_name
            self.state.last_press_ms = now_ms

    def snapshot(self) -> IntentState:
        with self._lock:
            return IntentState(
                last_dir=self.state.last_dir,
                last_press_ms=self.state.last_press_ms,
                quit_requested=self.state.quit_requested,
                p_is_down=self.state.p_is_down,
            )
