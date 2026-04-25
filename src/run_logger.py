"""
JSONL run logger with crash-safe writes.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np

from src.interfaces import RunLogger as RunLoggerABC


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return super().default(obj)


class JsonlRunLogger(RunLoggerABC):
    """JSONL-based run logger with crash-safe writes."""

    def __init__(self, experiment_name: str, runs_dir: str = "runs"):
        self.experiment_name = experiment_name
        Path(runs_dir).mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%H%M%S")
        self.filepath = os.path.join(runs_dir, f"{ts}_{experiment_name}.jsonl")
        self._file = open(self.filepath, "a")

    def write(self, record: dict) -> None:
        record = record.copy()
        record["timestamp"] = datetime.now().isoformat()
        record["experiment_name"] = self.experiment_name
        line = json.dumps(record, cls=NumpyEncoder)
        self._file.write(line + "\n")
        self._file.flush()
        os.fsync(self._file.fileno())

    def close(self) -> None:
        self._file.close()
