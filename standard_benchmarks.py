"""Standard Benchmarks — real data, equal FLOPs, all models.

NOTE: rebuilt — the previous version contained unresolved git conflict markers
and could not run. Delegates to scale_path.py so every dataset uses the same
fair harness: token-keyed LR schedule, multi-seed paired statistics, honest
FLOPs accounting, tokens + wall-clock reported.

Usage:
    python standard_benchmarks.py --data tinystories
    python standard_benchmarks.py --data wikitext --steps 2000 --seeds 0,1,2
    python standard_benchmarks.py --data openwebtext --samples 20000
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scale_path  # noqa: E402

_DATA_MAP = {"tinystories": "tinystories", "wikitext": "wikitext", "openwebtext": "openwebtext"}

if __name__ == "__main__":
    argv = sys.argv[1:]
    data = "tinystories"
    rest = []
    i = 0
    while i < len(argv):
        if argv[i] == "--data" and i + 1 < len(argv):
            data = argv[i + 1]
            i += 2
        else:
            rest.append(argv[i])
            i += 1
    if data not in _DATA_MAP:
        print(f"Unknown --data {data!r}; use one of {sorted(_DATA_MAP)}")
        sys.exit(1)
    sys.argv = [sys.argv[0], "--mode", "equal_flops", "--dataset", _DATA_MAP[data]] + rest
    scale_path.main()
