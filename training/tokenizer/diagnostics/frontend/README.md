# Frontend / JEPA diagnostic probes (2026-09-12)

Read-only probes used for the continuous-kernel sweep and JEPA analysis. They read TRAINING
sources and the label-free corpus only; none touches a sealed dataset. Run from `halo/` with the
project interpreter; outputs land in `out/`.

| script | question it answers |
|---|---|
| `knn_probe.py` | subject-disjoint 5-NN BA on 5 training streams for hand-crafted stats vs random / JEPA / adapted encoders (the "band-energy floor") |
| `ck_grad_probe.py` | do the continuous-kernel parameters receive gradients, and are they sign-consistent across batches (trained vs random init) |
| `ck_response_probe.py` | response scale (log1p regime), frozen standardisation, Nyquist-mask fingerprint, gain redundancy |
| `stage_probe_v4.py <fixed|multispan>` | where within-window time-invariance arises (frontend / random trunk / trained / teacher) + predictor controls (mean/wrong/no/shuffled context, persistence) |
| `jepa_specificity_v2.py [ckpt]` | predictor vs own / other-same-window / other-window targets, by horizon and resolution (fixed arm) |

Checkpoint paths inside the scripts point at the 2026-09-12 runs; edit the `CK` constants.
