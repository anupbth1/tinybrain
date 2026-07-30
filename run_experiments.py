"""
Bootstrap script to run NovaCore experiments.

Usage:
    python run_experiments.py --phase 3
    python run_experiments.py --phase 4
    python run_experiments.py --phase 5
    python run_experiments.py --all
"""

import sys
import os
import argparse

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Ensure novacore package is importable (handle case mismatch)
novacore_path = os.path.join(PROJECT_ROOT, 'NovaCore')
if os.path.exists(novacore_path) and 'novacore' not in sys.modules:
    sys.path.insert(0, novacore_path)

# Try import
try:
    from novacore.models.tiny_brain import TinyBrainConfig, ThinkingStep
    print("✅ novacore package imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"   PYTHONPATH: {sys.path[:3]}")
    sys.exit(1)


def run_phase3():
    """Phase 3: 15 mathematical diagnostics."""
    print("\n" + "=" * 70)
    print("PHASE 3: MATHEMATICAL VERIFICATION (15 diagnostics)")
    print("=" * 70)
    
    from novacore.experiments.phase3_verification import run_all
    results = run_all()
    
    # Save results
    import json
    from pathlib import Path
    from datetime import datetime
    out_path = Path(PROJECT_ROOT) / "NovaCore" / "experiments" / "results" / f"phase3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to: {out_path}")
    return results


def run_phase4():
    """Phase 4: 8 toy tasks."""
    print("\n" + "=" * 70)
    print("PHASE 4: TOY TASKS (8 tasks × 4 models)")
    print("=" * 70)
    
    from novacore.experiments.phase4_toy_tasks import run_all_tasks
    results = run_all_tasks(n_steps=200)
    
    import json
    from pathlib import Path
    from datetime import datetime
    out_path = Path(PROJECT_ROOT) / "NovaCore" / "experiments" / "results" / f"phase4_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to: {out_path}")
    return results


def run_phase5():
    """Phase 5: 50M language model comparison."""
    print("\n" + "=" * 70)
    print("PHASE 5: TINY LM (50M) COMPARISON")
    print("=" * 70)
    
    from novacore.experiments.phase5_tiny_lm import run_comparison
    results = run_comparison()
    
    import json
    from pathlib import Path
    from datetime import datetime
    out_path = Path(PROJECT_ROOT) / "NovaCore" / "experiments" / "results" / f"phase5_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to: {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, choices=[3, 4, 5, 0], default=3)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    
    if args.all:
        args.phase = 0
    
    if args.phase in (0, 3):
        run_phase3()
    if args.phase in (0, 4):
        run_phase4()
    if args.phase in (0, 5):
        run_phase5()
    
    print("\n✅ Experiments complete. Check NovaCore/experiments/results/ for data.")