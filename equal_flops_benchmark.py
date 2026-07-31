"""Equal-FLOPs Benchmark — FAIR comparison at the same compute budget.

NOTE: rebuilt — the previous version contained unresolved git conflict markers
and could not run. All logic now lives in scale_path.py; this script is a thin
wrapper so `python equal_flops_benchmark.py` keeps working with the same args.

Usage:
    python equal_flops_benchmark.py
    python equal_flops_benchmark.py --steps 2000 --samples 20000 --seeds 0,1,2,3,4
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scale_path  # noqa: E402

if __name__ == "__main__":
    argv = [a for a in sys.argv[1:] if not a.startswith("--mode")]
    sys.argv = [sys.argv[0], "--mode", "equal_flops"] + argv
    scale_path.main()
