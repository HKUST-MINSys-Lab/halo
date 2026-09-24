"""Moved to ``evaluation/rung2_frozen/sealed_eval.py`` on 2026-09-24 (rung 2 now lives beside rungs 1 and 3).

This alias only keeps the ``python -m training.support_classifier.sealed_eval`` commands recorded in
``results/artifacts/*/RESULTS.md`` runnable. Import from ``evaluation.rung2_frozen.sealed_eval``.
"""

from evaluation.rung2_frozen.sealed_eval import main

if __name__ == "__main__":
    main()
