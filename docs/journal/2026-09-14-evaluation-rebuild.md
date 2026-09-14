# 2026-09-14 — The evaluation was rebuilt: 4/8/16 s windows, multi-device cells, six fairness guarantees

**Status:** built and exercised. This entry records the protocol change itself, because every
number reported from 2026-09-13 onward sits on it and **is not comparable to anything measured
before it**.

Design: `docs/design/EVAL_EXPANSION_PLAN_20260913.md`.

---

## 1. What changed

**Three evidence durations instead of one.** Every model is asked the same question — *given this
N-second recording, name the activity* — with support also N-second recordings, at **N = 4, 8, 16**.
Previously everything was 6 s.

3 s was rejected on evidence: harnet5's trunk *crashes* on 90 samples
(`Calculated padded input size per channel: (4). Kernel size: (5)`). 4 s is the shortest window it
can physically accept.

**Grid paths now encode window length.** `build_grids._save` wrote to
`grids/<alignment>/<stream>/`, so building 4/8/16 s grids of one stream would have silently
overwritten each other. Paths are now `.../<stream>/w<seconds>/`.

**Two sealed datasets gained their simultaneous placements.** Both sources had multi-placement
recordings on disk that we discarded at conversion. RealWorld now emits **forearm, thigh, waist**
(the source has seven positions; it has no `wrist` — forearm is the watch-proxy). Shoaib emits
**left_pocket, right_pocket, wrist, belt**, excluding upper_arm. All placement streams share 100%
of their event ids, so alignment is exact.

**A new cell type.** Each placement is scored on its own, **plus exactly one all-device composite
per dataset** — not all subsets, which would only enable post-hoc selection.

## 2. The six fairness guarantees

Each is enforced in code with a test, not by convention.

| | guarantee |
|---|---|
| **F1** | Window boundaries are computed once per (stream, N) from the grid, **before any model is loaded**, and stored in the manifest. The test hashes the raw pre-resample slice handed to each model and asserts equality. |
| **F2** | Every model resamples from the **native** array, never from another model's resampled copy. Chained resampling compounds filter error and would advantage whoever sits first in the chain. |
| **F3** | Probe each model's real length tolerance; feed natively where it works, chunk only where the architecture genuinely refuses. |
| **F4** | One fan-out pooling helper (LiMU-BERT's length-weighted mean, lifted into `baselines/base.py`) for chunks and devices, used by every model including HALO. |
| **F5** | Padding disclosed per row via `padded` / `padded_fraction`, never hidden as N/A. |
| **F6** | The device set is fixed in the manifest; no model may choose, drop or re-order devices. |

**What the probe found, and it was a real defect:** harnet5's trunk is fully convolutional, so 4 s
and 6 s already collapse to a single 512-d vector. The adapter had been **centre-cropping 6 s down
to 5 s**, discarding a second of evidence for nothing. Replacing `flatten(1)` with a mean over the
trailing time axis lets it consume any window from 4 s upward in one pass. LiMU-BERT genuinely
refuses longer inputs (`pos_embed` is `nn.Embedding(20, H)`) and already chunked correctly.

F5 is visibly working: at 4 s the result rows record `padded=True` for HARNet, UniMTS, NormWear and
part of LiMU-BERT — their native windows exceed the budget — and `padded=False` for HALO on every
cell. Short budgets structurally disadvantage long-context models, which is disclosed rather than
hidden.

## 3. What multi-device showed

Composite versus the mean of its own member placements, 8 s, sealed. HALO uses centred neighbours;
released models use 1-NN.

| dataset | model | mode | k=1 | k=8 | k=32 |
|---|---|---|---:|---:|---:|
| RealWorld | **HALO** | native | **+11.1** | **+9.1** | −6.6 |
| | UniMTS | native | +3.7 | +3.6 | −8.9 |
| | HARNet | per-device-pooled | +8.9 | +9.9 | +3.2 |
| | NormWear | native | −0.3 | +4.1 | +2.1 |
| Shoaib | **HALO** | native | **+8.9** | +5.5 | +4.9 |
| | UniMTS | native | +4.1 | +3.4 | +3.2 |
| | LiMU-BERT-X | per-device-pooled | +13.0 | +5.9 | +4.4 |
| | HARNet | per-device-pooled | +7.9 | +5.0 | +4.1 |

HALO's hierarchical mean — accel and gyro averaged within a device, then devices averaged equally,
entirely parameter-free — gains more at k=1 than UniMTS's native SMPL-skeleton fusion (+11.1/+8.9
against +3.7/+4.1). That is a result worth stating plainly: a mean beat a purpose-built graph.

**Zero-shot benefits most.** HALO classifier at k=0, 8 s: RealWorld composite **63.1** against 44.3
for its mean single placement (+18.8); Shoaib **72.4** against 68.7. More simultaneous devices
sharply improve recognition from label text alone.

The k=32 reversals on RealWorld are a query-subset artefact — high-k availability changes which
queries can form an honest episode — not a real degradation. Read within a k, never across.

## 4. What this invalidates

**Every grid was rebuilt and every manifest fingerprint changed.** The four-arm JEPA ladder in
[2026-09-13-jepa-value-measured.md](2026-09-13-jepa-value-measured.md) was run at 6 s and is
**not comparable** to anything at 4/8/16 s. It stands as a clearly-labelled historical result.

One data change to be aware of: the RealWorld converter now drops any recording lacking a forearm
or thigh accelerometer, so the waist stream's composition changed (11,237 windows / 132 executions
→ 11,266 / 130). Shoaib and UT-Complex were verified row-identical to their legacy grids. RealWorld
remains accelerometer-only in every grid — the session frames carry no gyro columns — although the
raw release does ship gyroscope CSVs, so that is recoverable.

## 5. Coverage cost

Executions surviving each window length, and windows yielded:

| dataset | ≥4 s | ≥8 s | ≥16 s | windows @4 s | @8 s | @16 s |
|---|---:|---:|---:|---:|---:|---:|
| MotionSense | 100% | 99% | 96% | 6,709 | 3,270 | 1,550 |
| RealWorld | 100% | 100% | 100% | 16,824 | 8,378 | 4,163 |
| Shoaib | 100% | 100% | 100% | 3,150 | 1,540 | 770 |
| USC-HAD | 100% | **92%** | **83%** | 6,316 | 2,956 | 1,296 |
| InclusiveHAR | 100% | 100% | 100% | 1,859 | 893 | 415 |
| UT-Complex | 100% | 100% | 100% | 5,850 | 2,860 | 1,430 |

The high-k tail thins at 16 s: `k=128` at 16 s is **MotionSense only** and must never be read as a
six-dataset aggregate.

## 6. Links

- `docs/design/EVAL_EXPANSION_PLAN_20260913.md`
- Results built on this protocol: [2026-09-14-residual-classifier-first-result.md](2026-09-14-residual-classifier-first-result.md)
