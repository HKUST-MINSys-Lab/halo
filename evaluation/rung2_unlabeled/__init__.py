"""Rung 2 — unlabelled adaptation: accuracy as a function of the size N of an unlabelled
deployment pool, with the roster known and no label ever provided.

One established transductive method (EM-Dirichlet, Martin et al. CVPR 2024) applied identically to
every encoder; HALO's own arm is the same procedure unrolled inside training (Phase 4). Design:
docs/overview/roadmap.md, rung 2.
"""
