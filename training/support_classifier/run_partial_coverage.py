"""Moved to ``evaluation/rung2_frozen/run_partial_coverage.py`` on 2026-09-24 (rung 2 now lives beside rungs 1 and 3).

This alias only keeps the ``python -m training.support_classifier.run_partial_coverage`` commands recorded in
``results/artifacts/*/RESULTS.md`` runnable. Import from ``evaluation.rung2_frozen.run_partial_coverage``.
"""

from evaluation.rung2_frozen.run_partial_coverage import main

if __name__ == "__main__":
    main()
