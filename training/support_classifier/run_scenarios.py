"""Moved to ``evaluation/rung2_frozen/run_scenarios.py`` on 2026-09-24 (rung 2 now lives beside rungs 1 and 3).

This alias only keeps the ``python -m training.support_classifier.run_scenarios`` commands recorded in
``results/artifacts/*/RESULTS.md`` runnable. Import from ``evaluation.rung2_frozen.run_scenarios``.
"""

from evaluation.rung2_frozen.run_scenarios import main

if __name__ == "__main__":
    main()
