import csv, os, time
from dataclasses import dataclass, asdict

@dataclass
class TrainRow:
    ts: float
    seed: int
    step: int
    q1: float
    q2: float
    pi: float

class MetricsLogger:
    def __init__(self, path: str):
        self.path = path
        self._wrote_header = os.path.exists(path) and os.path.getsize(path) > 0
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def log(self, seed: int, step: int, q1: float, q2: float, pi: float):
        row = TrainRow(time.time(), seed, step, float(q1), float(q2), float(pi))
        write_header = not self._wrote_header
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[k for k in asdict(row).keys()])
            if write_header:
                w.writeheader()
                self._wrote_header = True
            w.writerow(asdict(row))
