import json
import csv
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

# Always save inside the project folder (data/sessions)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
SESSIONS_DIR = _PROJECT_ROOT /  "data" / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class SessionSummary:
    start_time: str
    end_time: str
    duration_sec: float
    total_reps: int
    total_sets: int
    reps_per_set: list
    avg_sec_per_rep: float | None
    params: dict

def save_session(summary: SessionSummary):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base = SESSIONS_DIR / f"session_{timestamp}"

    # JSON
    with open(base.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, ensure_ascii=False, indent=2)

    # CSV (one session per line)
    csv_path = SESSIONS_DIR / "sessions_log.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "start_time","end_time","duration_sec","total_reps","total_sets",
                "reps_per_set","avg_sec_per_rep","params_json"
            ])
        writer.writerow([
            summary.start_time,
            summary.end_time,
            f"{summary.duration_sec:.2f}",
            summary.total_reps,
            summary.total_sets,
            "|".join(map(str, summary.reps_per_set)),
            f"{summary.avg_sec_per_rep:.2f}" if summary.avg_sec_per_rep else "",
            json.dumps(summary.params, ensure_ascii=False)
        ])
