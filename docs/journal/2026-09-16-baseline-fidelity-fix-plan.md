# Baseline fidelity and runtime fix plan (2026-09-16)

**Status:** implemented and smoke-verified on 2026-09-16. The interrupted
`scenarios_baselines_equal_fusion_k0_64_20260916` directory remains diagnostic only; a new full
evaluation is still required before promoting baseline numbers.

Implementation evidence:

- RealWorld was rebuilt atomically with explicit part identities and common acc/gyro time
  coverage. The final audit contains 132 sessions; waist has measured acc+gyro in all 132 and the
  forearm/thigh each have 131 because one source recording is absent upstream.
- LiMU-BERT-X now uses 10 Hz, 20-sample two-second clips. HARNet-5 and HARNet-10 are separate
  released-checkpoint providers. UniMTS uses its released 20 Hz / 200-frame contract and
  source-style label dictionaries. NormWear has distinct enrollment and native-zero-shot
  representations and a torch CWT validated against the released SciPy definition.
- The eight classification sources now have exact native 4-, 8-, and 16-second materializations
  and duration-specific quality screens. Stream discovery refuses to advertise a missing duration.
- The sealed and scenario evaluators use duration-specific banks, split capability disclosures,
  float32 features, reusable classwise neighbor scores, bounded caches, strict row validation, and
  fail-closed completion metadata. A HARNet-5 4-second matched cross-placement smoke produced two
  valid rows, zero failures, and `complete=true`.
- The full training recipe profiles at 50.9 ms/step with 16 loader workers on the RTX 4090,
  projecting 29.7 minutes for 35,000 optimizer steps before validation/checkpoint overhead.

## 1. Verified findings and corrections

### Result blockers

1. **RealWorld gyroscope pairing is broken.** The current materialization has a complete waist
   gyroscope in 15 of 132 sessions. The converter pairs independently sorted accelerometer and
   gyroscope part lists by position even though their part-number conventions differ. Repair the
   pairing by explicit part identity, rebuild RealWorld sessions and every derived 4/8/16-second
   grid, regenerate quality artifacts, and invalidate dependent feature caches.
2. **LiMU-BERT-X uses the wrong physical clock.** The paper specifies 10 Hz; the adapter uses
   20 Hz and interprets its 20 positions as one second instead of two. Change the adapter and its
   cache contract to 10 Hz / two-second clips, then rerun its representation checks.
3. **HARNet gravity compatibility is checked per dataset, not per stream.** The first xrf_v2
   stream is gravity-removed, which excludes five compatible streams from the zero-shot bank.
   Make compatibility stream-aware and persist every bank exclusion with its reason.
4. **The zero-shot reference bank silently uses six-second grids.** Pass the active evaluation
   duration into reference-bank construction and include it in bank/cache provenance.
5. **NormWear uses the zero-shot MSiTF text-alignment vector as its enrollment representation.**
   Upstream downstream evaluation uses backbone patch tokens. Expose separate native-zero-shot and
   common-enrollment feature paths. Upstream flattening is valid only when query/support channel
   counts match; use an upstream-supported fixed-width patch/channel pooling rule for heterogeneous
   cross-stream cells or mark such cells unsupported. Do not silently compare variable-width
   flattened vectors.
6. **NormWear's optimized CWT is not equivalent to its released SciPy path.** A direct probe found
   an extra temporal sample, a nearly dead finest scale (standard deviation 0.00034 versus 2.68),
   and material disagreement at the next scales. Vendor the small released SciPy-equivalent CWT or
   correct and validate the torch implementation on values and output shape before rescoring.

### Protocol and attribution fixes

1. **UniMTS label text:** the external report's bare-label claim is incorrect. Released UniMTS
   joins all strings in each dataset's `label_dictionary`. HALO's eight training-derived
   paraphrases are still not faithful. Register a sealed, source-backed label dictionary per test
   dataset and use exactly that as the primary UniMTS row; report bare-label and HALO-paraphrase
   variants only as named sensitivity analyses.
2. **Published input lengths:** UniMTS declares 200 frames and HARNet declares harnet5/150 samples,
   but both adapters consume the complete 4/8/16-second interval. Choose one primary policy before
   rerunning: published preprocessing is the fidelity default; full-interval inference may be a
   clearly named favourable-to-baseline sensitivity. Artifact fields must report consumed samples,
   padding and truncation actually performed.
3. **HARNet model identity:** results currently use harnet5 while the paper's headline encoder is
   harnet10. Add an explicit harnet10 released-checkpoint arm and name every row `HARNet-5` or
   `HARNet-10`; never report the ambiguous label `HARNet`.
4. **Capability columns:** replace ambiguous `native_few_shot_adaptation` with two disclosures:
   `native_support_conditioning` (accepts labeled demonstrations at inference) and
   `published_few_label_finetuning` (the release publishes gradient-based few-label adaptation).
   This keeps HALO's support-conditioned contribution distinct from ordinary supervised fine-tuning.
5. **LiMU-BERT-X pooling and provenance:** mean hidden-state pooling plus duration-weighted clip
   pooling are HALO evaluation choices, not the released GRU classifier. Record them as such and
   remove the incorrect MotionSense/Shoaib pretraining-overlap claim for the Phase-II checkpoint.
6. **NormWear disclosures:** report backbone, MSiTF and text-tower parameter counts separately;
   distinguish this macro-F1 open-vocabulary protocol from the paper's hand-written-option AUROC.
7. **HARNet overlap and preprocessing:** disclose upstream WISDM/RealWorld use, the chosen
   resampler, and whether the primary row uses the published crop or full-window sensitivity.

## 2. Implementation order

### Phase 1: data and cache identity

1. Fix RealWorld part matching with unit tests covering unnumbered gyro part 1 and numbered later
   parts. Fail conversion when an available co-covered gyroscope part is lost.
2. Rebuild RealWorld atomically, audit all 132 sessions and three placements, then regenerate
   grids and duplicate/implausibility artifacts.
3. Bump the feature-cache schema. Include preprocessing implementation/version, physical rate,
   native clip duration, pooling mode, and evaluation duration in every adapter key.

### Phase 2: adapters

1. Correct LiMU-BERT-X to 10 Hz / 20-sample two-second clips; test resampling duration, final-clip
   weighting, strict checkpoint load, and fixed outputs on a pinned fixture.
2. Change compatibility to accept the concrete stream (including composite members), then repair
   HARNet's gravity guard and bank-exclusion artifact.
3. Implement exact published-window and explicit full-window modes for UniMTS and HARNet. Reject a
   result row whose accounting metadata disagrees with the tensor passed to the model.
4. Split NormWear's native zero-shot feature from its common enrollment feature. Validate the CWT
   against the vendored reference over raw, first-difference and second-difference channels.
5. Add harnet10 as a separately named adapter rather than changing historical harnet5 in place.

### Phase 3: evaluation semantics

1. Pass `window_seconds` into reference-bank construction; persist bank source rows, exclusions,
   label counts and fingerprints.
2. Add source-backed UniMTS label dictionaries and hash their complete text into result provenance.
3. Split the capability columns and update the merger, Markdown, CSV and scenario outputs together.
4. Keep equal-weight normalized fusion as one prediction: independently normalize semantic scores
   over all candidates and support scores over enrolled candidates, add with coefficients 1 and 1,
   then take one argmax.
5. Add a result validator that refuses stale cache schema, ambiguous model identities, missing
   capability fields, mismatched consumed-length accounting, or a partial/interrupted run.

### Phase 4: acceptance before a full sweep

1. Run adapter fixture tests and one 4/8/16-second cell per provider.
2. Verify query/support identity, execution disjointness, candidate rosters and equal manifests
   across providers.
3. Verify RealWorld exposes measured gyroscope to LiMU-BERT-X, HALO and NormWear where raw coverage
   permits it.
4. Run a two-cell `k={0,1,64}` smoke and require zero failed rows, explicit unsupported rows, stable
   fingerprints, finite metrics and a clean rerun from cache.
5. Only then run all baselines for `k=0..64` and publish per-scenario/per-dataset results.

## 3. Efficiency plan

### Measured evaluation bottlenecks

A cached HARNET profile over two partial-coverage cells at `k=64` took 64.1 seconds and peaked at
11.8 GiB host RAM. Cumulative costs were: manifest fingerprinting 20.1 s, quality-artifact
validation 15.5 s, manifest construction 12.3 s, grid fingerprinting 11.5 s, 1-NN 2.6 s, and
equal-weight fusion 1.6 s. Optimize in that order:

1. Replace recursive `dataclasses.asdict`/JSON manifest hashing with an incremental binary/text
   hash over query ids, support ids and labels. Compute it once per task and pass it to scoring and
   artifact writers. This is audit-equivalent and removes millions of deep copies.
2. Cache per-grid content fingerprints keyed by path, size, mtime and ctime; aggregate cached
   per-grid digests instead of reopening and sampling the same arrays for every quality check.
3. Add a validated local stamp for duplicate/implausibility artifacts so an unchanged corpus does
   not spend 15 seconds revalidating on every evaluator process. A changed stat or schema must
   force the complete validation.
4. Replace per-query RNG construction in manifest sampling with a tested batched deterministic
   sampler, while preserving without-replacement selection and seed reproducibility.
5. Preserve float32 in `_normalise`; it currently upcasts every representation to float64 and
   doubles host memory. Require prediction parity tests before changing historical outputs.
6. Reuse classwise maximum neighbor scores for both 1-NN and equal-weight fusion. For high k,
   compute them in bounded GPU chunks using `scatter_reduce`, with the normalized feature matrix
   uploaded once. This prevents the unbounded `[queries,supports,dimension]` CPU temporary.
7. Persist duration-specific zero-shot bank features/centroids. Do not reload all eight training
   corpora merely to rebuild an unchanged bridge.
8. Keep compact audit output for full sweeps, but retain complete source, manifest and feature
   fingerprints. Expanded per-query artifacts are an opt-in diagnostic.

### Measured training bottlenecks

The current four-support-set recipe measured 48.7 ms/step with 8 workers and 43.9 ms/step with 12
workers; 16 workers regressed to 54.8 ms/step. At 12 workers this is about 25.6 minutes for 35k
optimizer steps before validation/checkpoint overhead. Mean components were 10.6 ms loader wait,
15.9 ms encode, 10.8 ms backward and 1.2 ms optimizer; peak allocated VRAM was 1.37 GiB.

1. Fix `profile.py` first: it omits the new augmentation fields and profiles only compatible
   episodes rather than the trainer's acquisition/enrollment mixture. Build profiler and trainer
   settings through one shared recipe function.
2. Re-profile after sampler caching. The 2026-09-16 confirmation measured 12 workers at 62.0 ms/step
   (19.5 ms loader wait; projected 36.2 min/35k) and 16 workers at 50.9 ms/step (6.7 ms loader wait;
   projected 29.7 min/35k) on identical episodes. Use 16 as this machine's default and continue to
   record median/p90 because the sampler is bursty.
3. Cache feasibility by acquisition regime, dataset, label, candidate count and k. The current
   sampler takes a median 55 ms and p90 148 ms to construct four mixed-regime support sets on one
   process after warmup; workers hide most, not all, of it.
4. Fix the curriculum correctness findings C1-C15 before tuning sampler internals. Several caches
   depend on the correct definition of cross-placement, zero support and feasible labels.
5. `torch.compile` is not currently available as an optimization: the bounded probe fails because
   PyTorch 2.9 Inductor reads the legacy TF32 API after the process sets the new API. Resolve that
   API conflict and benchmark prediction parity and wall time before enabling compile.
6. Do not increase episodes per optimizer step solely to occupy VRAM: that changes the effective
   batch/objective. The current low VRAM use reflects a small model and irregular episode CPU path,
   not an out-of-memory constraint.
