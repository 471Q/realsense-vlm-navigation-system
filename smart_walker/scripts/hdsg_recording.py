"""Synchronised RGB-D recording and replay for HDSG evaluation runs.

`HDSG_EVALUATION_CONTRACT.md` section 12 requires an input file or recording identifier on every
evaluated event, and Chapter 5 replays each base scene under several conditions using the same
observation and intent. Without a recording the conditions receive different inputs, the pairing
between them is lost, and a discordant-pair analysis degrades into independent samples. This
module supplies the recording half and the reader that replays it.

Formats follow the convention already used elsewhere in this repository (see
`rgbd_facts_from_pair.py`'s `realsense_mm` preset):

    <observation_id>.color.png      8-bit BGR, lossless
    <observation_id>.depth_mm.png   16-bit single channel, millimetres
    manifest.jsonl                  one line per observation

The colour frame is PNG rather than JPEG deliberately. Replay feeds it back through the detector,
and JPEG artefacts would change detections, which would make a replayed run disagree with the run
it was supposed to reproduce.

Depth is stored in millimetres because the D455f reports depth in integer units of its depth
scale, which is 0.001 m on this device, so the conversion is exact rather than a quantisation.
The scale in force is recorded in the manifest, and `depth_scale_is_exact` records whether that
assumption held for the run.

Writing happens on a worker thread. The sensing loop never blocks on disk, and a queue overflow
is counted rather than ignored, so a run that failed to record every observation is known to be
incomplete instead of appearing complete.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from queue import Empty, Full, Queue
import threading
from typing import Any, Iterator, Optional

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None


MILLIMETRES_PER_METRE = 1000.0
UINT16_MAX = 65535


def colour_filename(observation_id: str) -> str:
    return f"{observation_id}.color.png"


def depth_filename(observation_id: str) -> str:
    return f"{observation_id}.depth_mm.png"


@dataclass
class _PendingWrite:
    observation_id: str
    colour: Any
    depth_m: Any
    timestamp_ms: float


class ObservationRecorder:
    """Writes synchronised RGB and depth frames keyed by observation identifier.

    The recorder is inert until `start` is called, and the caller is expected to create it only
    for evaluation runs. `record` copies nothing: the caller must pass frames it will not mutate,
    which matches how the entry script already copies frames before handing them on.
    """

    def __init__(self, directory: Path, *, depth_scale: float = 0.001,
                 queue_size: int = 64) -> None:
        if cv2 is None or np is None:
            raise RuntimeError("OpenCV and NumPy are required for RGB-D recording.")
        self.directory = Path(directory)
        self.depth_scale = float(depth_scale)
        self._queue: "Queue[Optional[_PendingWrite]]" = Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._written = 0
        self._dropped = 0
        self._failed = 0
        self._manifest_path = self.directory / "manifest.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self._manifest_path

    def start(self) -> "ObservationRecorder":
        self.directory.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()
        return self

    def record(self, observation_id: str, colour: Any, depth_m: Any,
               timestamp_ms: float) -> bool:
        """Queues one observation. Returns False if the queue was full and it was dropped."""
        if self._thread is None:
            return False
        try:
            self._queue.put_nowait(
                _PendingWrite(observation_id, colour, depth_m, float(timestamp_ms))
            )
            return True
        except Full:
            with self._lock:
                self._dropped += 1
            return False

    def stop(self, timeout_s: float = 10.0) -> dict:
        """Drains outstanding writes and returns the run's recording counters."""
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join(timeout=timeout_s)
            self._thread = None
        self._stop.set()
        return self.stats()

    def stats(self) -> dict:
        with self._lock:
            return {
                "written": self._written,
                "dropped": self._dropped,
                "failed": self._failed,
                "complete": self._dropped == 0 and self._failed == 0,
            }

    def _drain(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=0.1)
            except Empty:
                if self._stop.is_set():
                    return
                continue
            if item is None:
                return
            try:
                self._write(item)
            except Exception:
                with self._lock:
                    self._failed += 1

    def _write(self, item: _PendingWrite) -> None:
        colour_path = self.directory / colour_filename(item.observation_id)
        depth_path = self.directory / depth_filename(item.observation_id)

        depth = np.asarray(item.depth_m, dtype=np.float32)
        # Values above the 16-bit ceiling are beyond any distance the guidance policy acts on,
        # and non-finite readings are already treated as absent by the sector logic, so both
        # collapse to zero, the same value the sensor reports for an invalid pixel.
        millimetres = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0) * MILLIMETRES_PER_METRE
        millimetres = np.clip(np.rint(millimetres), 0, UINT16_MAX).astype(np.uint16)

        if not cv2.imwrite(str(colour_path), item.colour):
            raise RuntimeError(f"could not write {colour_path}")
        if not cv2.imwrite(str(depth_path), millimetres):
            raise RuntimeError(f"could not write {depth_path}")

        entry = {
            "observation_id": item.observation_id,
            "sensor_timestamp_ms": item.timestamp_ms,
            "colour_path": colour_filename(item.observation_id),
            "depth_path": depth_filename(item.observation_id),
            "depth_units": "millimetre",
            "depth_scale_m": self.depth_scale,
            "depth_scale_is_exact": abs(self.depth_scale - 0.001) < 1e-9,
            "width": int(item.colour.shape[1]),
            "height": int(item.colour.shape[0]),
        }
        with self._lock:
            with self._manifest_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry, separators=(",", ":"), ensure_ascii=True) + "\n")
            self._written += 1


class ObservationReader:
    """Reads back a recorded run for deterministic replay."""

    def __init__(self, directory: Path) -> None:
        if cv2 is None or np is None:
            raise RuntimeError("OpenCV and NumPy are required for RGB-D replay.")
        self.directory = Path(directory)
        self._manifest_path = self.directory / "manifest.jsonl"
        if not self._manifest_path.is_file():
            raise FileNotFoundError(f"no recording manifest at {self._manifest_path}")

    def entries(self) -> list[dict]:
        """Returns the manifest in recorded order."""
        lines = self._manifest_path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def load(self, entry: dict) -> tuple[Any, Any]:
        """Returns (colour BGR, depth in metres) for one manifest entry."""
        colour_path = self.directory / entry["colour_path"]
        depth_path = self.directory / entry["depth_path"]
        colour = cv2.imread(str(colour_path), cv2.IMREAD_COLOR)
        if colour is None:
            raise FileNotFoundError(f"could not read {colour_path}")
        raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f"could not read {depth_path}")
        # OpenCV returns a single-channel PNG as either (H, W) or (H, W, 1) depending on the
        # build and on which other libraries are loaded in the process: importing the entry
        # script's dependencies is enough to change which form comes back. Both are the same
        # depth image, so the trailing axis is dropped rather than treated as a format error.
        if raw.ndim == 3 and raw.shape[2] == 1:
            raw = raw[:, :, 0]
        if raw.dtype != np.uint16 or raw.ndim != 2:
            raise ValueError(
                f"{depth_path} must be a 16-bit single-channel PNG, got dtype={raw.dtype} "
                f"shape={raw.shape}"
            )
        depth_m = raw.astype(np.float32) / MILLIMETRES_PER_METRE
        return colour, depth_m

    def observations(self) -> Iterator[tuple[dict, Any, Any]]:
        """Yields (entry, colour, depth_m) for every recorded observation, in order."""
        for entry in self.entries():
            colour, depth_m = self.load(entry)
            yield entry, colour, depth_m
