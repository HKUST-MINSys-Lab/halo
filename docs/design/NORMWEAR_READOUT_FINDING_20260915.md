# NormWear is fed the wrong tensor for enrolled readouts (2026-09-15)

**Status:** measured, not yet fixed. Affects every NormWear enrolled number in the project. Does
**not** affect its zero-shot row.

## The defect

NormWear has two output paths:

| path | tensor | purpose |
|---|---|---|
| MSiTF aggregator | `(bn, 2048)`, conditioned on a fixed text query, aligned to Clinical TinyLlama, matched by L1 | **zero-shot** label matching |
| backbone | `(bn, nvar, P, 768)` patch tokens | **representation** for downstream probing |

`baselines/normwear/adapter.py: window_features` returns the **MSiTF aggregator output**, and that is
what feeds 1-NN, prototype and ridge in `sealed_eval.py`. NormWear's own downstream benchmark
(`auxiliary_repos/NormWear/downstream_pipeline/model_apis.py`, in `NormWearAPI.get_embedding`) pools
the **backbone** instead:

```python
out, hiddens = self.backbone.get_signal_embedding(spec, hidden_out=True, device=device)  # 1,nvar,P,E
out = torch.mean(out[0, :, :, :], dim=1)   # nvar, E   (mean over patches)
audio_embeddings = out.flatten()           # nvar * E
```

So we push a text-alignment head into a retrieval task. The backbone tensor is already computed
inside `_signal_encode_np` (it is the `sensor_out` that the aggregator consumes), so the fix costs
nothing at inference time.

## Measurement

Probe: one sealed stream, 8 s windows, subject-excluded global 1-NN over all windows. **This is not
the sealed episodic protocol**, so absolute values are not comparable to `RESULTS.md`; the
comparison between feature choices is valid because the protocol is identical across rows.
Effective rank is the repo's own definition (`training/support_classifier/train.py:effective_rank`,
entropy of the centred singular spectrum).

**MotionSense / phone_front_pocket, 2,400 windows**

| features | dim | rank (centred) | rank (uncentred) | mean pairwise cos | 1-NN macro F1 |
|---|---:|---:|---:|---:|---:|
| MSiTF — what we use | 2048 | 18.5 | 3.9 | 0.983 | 50.2 |
| backbone, channel-mean | 768 | 100.9 | 5.2 | 0.996 | 64.4 |
| backbone, flatten (upstream recipe) | 4608 | 614.0 | 58.8 | 0.992 | **66.3** |

**InclusiveHAR / phone_waist, 1,048 windows**

| features | dim | rank (centred) | 1-NN macro F1 |
|---|---:|---:|---:|
| MSiTF — what we use | 2048 | 14.9 | 26.9 |
| backbone, channel-mean | 768 | 91.0 | **34.7** |
| backbone, flatten | 4608 | 377.6 | 30.5 |

Correcting the tensor is worth **+16.1** (MotionSense) and **+7.8** (InclusiveHAR) macro F1.

## Two corrections to the record

`docs/journal/2026-09-14-baseline-failure-analysis.md` states that NormWear's MotionSense embeddings
have "mean pairwise cosine 0.966 and effective rank 1.4 in 2048 dims" and concludes the failure is
"mechanistic rather than weak features". Both parts need revising, and **a journal correction entry
is owed** (past entries are immutable; this file is not that entry).

1. **The rank figure does not reproduce.** Under the repo's own centred definition I measure 18.5,
   not 1.4; uncentred gives 3.9. The cosine figure is close (0.983 measured against 0.966 stated).
2. **Collapse is the wrong mechanism.** The *correct* backbone features have a **higher** mean
   pairwise cosine (0.996) and a similarly low uncentred rank, yet retrieve 14 points better. A high
   mean cosine here is a common-mode offset, which centring removes — which is precisely why our own
   centring step exists. Mean cosine and uncentred rank are therefore not valid evidence that a
   representation carries no information, and should not be used that way again.

## Fix

1. Add a `feature_source` choice to `NormWearAdapter`, defaulting to `backbone`, that mean-pools
   `sensor_out` over patches and reduces over channels. Keep `msitf` available for reproduction of
   the current numbers. Return the backbone tensor from `window_features`; leave `predict` and
   `predict_candidates_from_features` on the MSiTF path unchanged.
2. Decide channel reduction: **mean** over channels gives a stream-independent 768-d vector; the
   upstream **flatten** gives `nvar x 768`, which varies with channel count and so is not comparable
   across streams of differing width. Upstream can flatten because each of their datasets has fixed
   `nvar`. Our cells do not, so **mean is the correct adaptation**; flatten won on MotionSense and
   lost on InclusiveHAR, so this is not a large sacrifice. Record the choice and the reason.
3. **Do not simply swap the tensor — NormWear is the one model whose two paths share one array.**
   `predict_candidates_from_features` (and the `candidate_scores_from_features` hook another agent
   added on 2026-09-15 for the scenario hybrid readout) compares the **same** `features` array
   against 2048-d TinyLlama label embeddings by L1. If `window_features` starts returning 768-d
   backbone vectors, that comparison breaks: wrong dimension and wrong space. The adapter must
   carry both tensors — for example `window_features` returns the backbone representation while the
   MSiTF vector is cached on `state` per stream and used by the text path — and
   `candidate_scores_from_features` must assert the array it receives is the MSiTF one. Coordinate
   with whoever owns that hook before changing either.
4. Bump `FEATURE_CACHE_SCHEMA` in `sealed_eval.py` so every cached NormWear feature array is
   invalidated rather than silently reused.
5. Re-run NormWear rows: the sealed comparison, the scenario runs, and the failure analysis probes.
6. Write the journal correction entry with the re-run numbers.

## Scope of what is currently wrong

* **Wrong (understated):** every NormWear `1nn` / `prototype` / `ridge` / `centred` row at k>=1, in
  `docs/results/RESULTS.md`, the 2026-09-14 journal entries, and any scenario run.
* **Unaffected:** NormWear's `native_zero_support` row at k=0. The MSiTF path is its genuine
  zero-shot mechanism and the adapter implements it faithfully, including the L1 metric and the
  native query and answer templates.
* **Unaffected:** every other model. This is a NormWear-specific adapter defect.
