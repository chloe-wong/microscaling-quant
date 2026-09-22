"""mxq.schedule — a list of float(e, m), one per accumulator position of a reducer.

    load(path, rows=16)   -> [(e, m), ...]    CSV rows `position,e,m` or `position,B,e,m`, optional header
                                              (MXQuant complete_integration_e2e `load_schedule`, same rules)
    fixed(e, m, rows=16)  -> [(e, m)] * rows
    HW_FINAL                                  the MX-Gemmini tapeout's 16 lanes

A position is a PE lane of matmul.systolic.
A schedule is fully defined when positions 0..rows-1 are each given once and nothing else is.
"""
import csv
from pathlib import Path
from typing import List, Tuple, Union

__all__ = ["load", "fixed", "HW_FINAL"]

Entry = Tuple[int, int]

#: MX-Gemmini tapeout lanes (schedule_hw_final.csv, rtl_exact/acc_schedule.csv): 0-7 e4m4, 8-9 e4m5, 10-14 e4m6, 15 e8m7
HW_FINAL: List[Entry] = [(4, 4)] * 8 + [(4, 5)] * 2 + [(4, 6)] * 5 + [(8, 7)]


def load(path: Union[str, Path], rows: int = 16) -> List[Entry]:
    entries = {}
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader, None)
        lines = []
        if header and header[0].strip().isdigit():
            lines.append(header)
        lines.extend(reader)
        for row in lines:
            row = [c.strip() for c in row if c.strip()]
            if len(row) == 3:
                pos, e, m = int(row[0]), int(row[1]), int(row[2])
            elif len(row) == 4:
                pos, _B, e, m = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            else:
                continue
            if not 0 <= pos < rows:
                raise ValueError(f"schedule {path}: position {pos} outside 0..{rows - 1}")
            if pos in entries:
                raise ValueError(f"schedule {path}: position {pos} given twice")
            entries[pos] = (e, m)
    missing = [i for i in range(rows) if i not in entries]
    if missing:
        raise ValueError(f"schedule {path}: missing positions {missing}")
    return [entries[i] for i in range(rows)]


def fixed(e: int, m: int, rows: int = 16) -> List[Entry]:
    return [(e, m)] * rows
