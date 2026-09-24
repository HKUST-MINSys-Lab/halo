# Rung 1: how learnable the adaptation may be, and five fixes to the rung-1 build

Date: 2026-09-23. Status: **decision record + code fixes; nothing run.** Branch
`feat/evaluation-package-20260923`. Live plan: [`docs/overview/roadmap.md`](../overview/roadmap.md)
(not yet amended for level B; see "Open").

## 1. The decision: levels A and B, not C

Question (Alex): with one unsupervised adaptation method applied to every encoder, HALO's only room
to adapt better is training. Can the adaptation itself be learnable and trained end to end?

Three levels were laid out:

| level | what is learned | status |
|---|---|---|
| A | encoder + `p_text` trained **through** the fixed EM-Dirichlet, unrolled | **in** (already built) |
| B | EM-Dirichlet unrolled with a few learnable, per-iteration **constants** (temperature, λ, damping, support-vs-pool weight), initialised at the reference values, fitted per encoder for the baselines too | **in**; not built |
| C | a fully learned adapter that reads the pool (ARM-CML, ICRM, TPN, a set transformer) | **out** |

Why C is out, from the 2026-09-23 literature pass (scite, Consensus, alphaXiv, Exa, WebSearch):

- ARM (Zhang et al., arXiv:2007.02931) is mixed on real shifts: on WILDS it is below plain training
  on FMoW worst-case (24.6 vs 32.3) and iWildCam (70.3 vs 71.6).
- ARM collapses with an empty pool: 49.5 vs 79.3 (FEMNIST) and 36.5 vs 94.2 (rotated MNIST) at zero
  context (ICRM paper, Gupta et al., ICLR 2024, Table 2). Our N=0 checks would fail by construction.
- ICRM's curves are flat from 25 context samples to 100: a learned adapter mostly identifies the
  environment — our registered "acquisition fingerprint" threat — rather than exploiting class
  structure.
- A fixed transductive loss on cross-entropy features beats meta-learned transductive modules (TIM,
  Boudiaf et al., arXiv:2008.11297; Tian et al., ECCV 2020).
- Our own history: every learned head over the parameter-free vote collapsed (T1, T3) or fell below
  its floor at high k (v3 −3.6, v4 −0.9 at k=128).

The line between B and C: B's parameters are constants shared across tasks, never functions of the
pool. Once they depend on the pool it is C.

## 2. Fixes made in the same session

1. **HALO is scored through its own `p_text`.** `evaluation.zero_shot` previously scored every HALO
   checkpoint through a ridge bridge refitted on the training bank, while the pool training arm
   trains `p_text` through the unrolled transduction — so level A would have been evaluated with a
   different head from the one trained. `ProviderScorer` now reads the checkpoint's `p_text`
   (`checkpoint_text_projection`) and uses it; the ridge bridge is the fallback for HALO-slot
   checkpoints without a learned head (neighbours control, corpus-matched arms). Each row records
   `zero_shot_route` (`halo_p_text` / `halo_text_bridge`). Test: the evaluator's simplex rows equal
   the trainer's `probability_features_torch` on the same checkpoint (`tests/test_rung1_text_route.py`).
   **This changes the tier-1 HALO row** from the bridge (38.6 at k=0 in the sealed table) to `p_text`;
   no number has been read on either route.
2. **The partition weight λ.** The reference sets `λ = int(C/k_eff)·N` (few-shot) and `int(C/5)·N`
   (zero-shot), where C is its label space (1,000 for ImageNet) and `k_eff` the classes present in a
   task (3–10; 5 is its zero-shot guess). The port applied `int(C/5)·N` to our rosters, switching the
   partition term off for rosters under five classes and doubling it at ten or more; `k_eff` was never
   passed. Our label space is the declared roster and the regime is "roster known", so the non-oracle
   `k_eff` is the roster size: `transduce` now defaults to `λ = N` for every roster and every k.
   `paper_lambda` still reproduces the reference for any explicit `k_eff`. Disclosed in the module
   docstring.
3. **The curve's null.** `run.py` stripped N=0 from the pool sizes, so the summary's "N=0" column was
   the inductive readout — zero-shot arg-max at k=0, embedding prototypes at k>0 — while every other
   column is EM-Dirichlet on text probabilities. "Gain with N" mixed a change of feature space with the
   pool's effect. N=0 now runs the same transductive method over the scored set alone; the inductive
   readout is kept as a separate `inductive` anchor column. The summary is also now dataset-balanced
   (mean over datasets of per-dataset means), matching the sealed table.
4. **bf16 autocast.** The trainer runs under bf16 autocast; the unrolled EM-Dirichlet's einsums and
   log-gamma terms would have run in bf16 there. `transduce` now disables autocast and computes in
   float32, as the reference does (test: identical logits inside and outside autocast).
5. **Smaller:** the pool-minus-supports filter in `ncurve.run_cell` rebuilt a Python set per element
   (quadratic in pool size; now `np.isin`); the trainer's pooled readouts mask supports whose bound is
   −1 instead of clamping them to candidate 0 (unreachable with today's sampler, silent if a future
   one adds off-roster supports); four module docstrings linked a non-existent
   `2026-09-22-rung1-rung1-implementation-plan.md` after the renumbering; `design_of_record.md` still
   described evidence-aware v2 as active and v3 as the default — rewritten for v4.

Suite: 1,132 passed, 2 skipped.

## 3. Open

- **Embedding-affinity term.** EM-Dirichlet sees only text probabilities, so rung 1 does not test the
  embedding geometry where HALO leads in rung 2 (1-NN 71.7 vs UniMTS 66.9 at k=8; bridges at k=0 are
  38.6 / 36.0 / 33.1). Adding a term over embedding neighbours, applied to all six, would change the
  method from the pure published one. Undecided.
- Level B's exact parameterisation, its baseline tier, and ladder step 5 are to be written into the
  roadmap once the affinity question is settled.
- The disjoint-class control yields rows only at k=0 (supports are drawn from a pool that holds only
  held-out classes, so kept classes have none). Intended or not, it should be stated in the roadmap.
