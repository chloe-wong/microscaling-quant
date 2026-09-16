"""mxq.schedule — per-lane accumulator formats for a 16-deep systolic column.

    load(path, window=16) -> [(e, m), ...]     CSV rows `lane,e,m` or `lane,B,e,m`, optional header
                                                (MXQuant complete_integration_e2e `load_schedule`, verbatim rules)
    fixed(e, m, window=16) -> [(e, m)] * window
    HW_FINAL                                    the tapeout schedule (schedule_hw_final.csv / rtl_exact acc_schedule.csv)
"""
import csv
from pathlib import Path
from typing import List, Tuple, Union

__all__ = ["load", "fixed", "HW_FINAL"]

Lane = Tuple[int, int]

#: schedule_hw_final.csv: lanes 0-7 e4m4, 8-9 e4m5, 10-14 e4m6, 15 e8m7
HW_FINAL: List[Lane] = [(4, 4)] * 8 + [(4, 5)] * 2 + [(4, 6)] * 5 + [(8, 7)]


def load(path: Union[str, Path], window: int = 16) -> List[Lane]:
    entries = {}
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader, None)
        rows = []
        if header and header[0].strip().isdigit():
            rows.append(header)
        rows.extend(reader)
        for row in rows:
            row = [c.strip() for c in row if c.strip()]
            if len(row) == 3:
                lane, e, m = int(row[0]), int(row[1]), int(row[2])
            elif len(row) == 4:
                lane, _B, e, m = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            else:
                continue
            entries[lane] = (e, m)
    missing = [i for i in range(window) if i not in entries]
    if missing:
        raise ValueError(f"schedule {path} missing lanes {missing}")
    return [entries[i] for i in range(window)]


def fixed(e: int, m: int, window: int = 16) -> List[Lane]:
    return [(e, m)] * window
