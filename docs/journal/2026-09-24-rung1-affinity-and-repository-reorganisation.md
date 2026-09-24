# Rung 1 gets an embedding-affinity term and is k = 0 only; the repository moves to main-only and is reorganised

Date: 2026-09-24. Status: **decisions + code; nothing run on real data.** Supersedes the "Open"
list of [the 2026-09-23 learnability entry](2026-09-23-rung1-learnability-ceiling-and-fixes.md).
Live plan: [`docs/overview/roadmap.md`](../overview/roadmap.md).

## 1. Rung 1 is k = 0

The roadmap had specified rung 1 as a curve in N "at fixed k ∈ {0, 1, 2, 4, 8}", and the code ran
every pool size at every k, drawing labelled supports for k > 0. The thesis defines rung 1 as "no
label ever". Alex, asked about it: *"i thought for rung 1, any examples or demonstrations we provide
are unlabelled?"* Rung 1 is now k = 0 only (`ncurve.DEFAULT_K = (0,)`); labels-plus-pool stays
computable as a separately named semi-supervised condition and is not in the plan.

## 2. The embedding-affinity term

Alex agreed to add it after the explanation that EM-Dirichlet sees each window only through its
text probabilities, so rung 1 would otherwise compare the six encoders' text heads (bridges at k = 0:
HALO 38.6, HARNet-5 36.0, UniMTS 33.1) rather than their geometry, where HALO leads under 1-NN
(71.7 vs 66.9 at k = 8). *"The encoder decides which recordings belong together; the text decides
what to call each group."*

Each window's EM update gets `μ · log(neighbour vote)`, the vote being the cosine-weighted mean of
the **text probabilities** of its 10 nearest neighbours in the encoder's own embedding space: a
local prior beside the method's global class-proportion prior (cf. LaplacianShot's pairwise term).
μ = 0 is the published method bit for bit; μ = 1 a priori, in the evaluator and in the unrolled
training readout (`--pool-affinity-mu`, `--pool-affinity-knn`).

**The first version was wrong, and a synthetic check caught it.** Voting on the neighbours' current
*assignments* (the Laplacian form) gained on clean clusters, but when neighbourhoods were
uninformative — random, or grouped by an unrelated "subject" — it fed errors back each iteration
and collapsed every window onto one class (six classes: 0.84 → 0.17, chance). That would have
punished an encoder for mediocre geometry far beyond what the geometry deserves, i.e. an unfair
comparison. Voting on the neighbours' fixed *text evidence* instead:

| six-class synthetic, noisy text head | μ = 0 | μ = 1 | μ = 2 |
|---|---:|---:|---:|
| clean class clusters (high neighbour purity) | 0.83–0.84 | 0.88–0.90 | 0.94–0.95 |
| noisy clusters (purity ~0.87) | 0.83–0.84 | 0.88–0.89 | 0.93 |
| random embeddings (purity ~0.25) | 0.83–0.84 | 0.85 | 0.73–0.77 |
| grouped by subject (purity 0.17) | 0.83–0.84 | 0.84 | 0.68–0.83 |

μ = 1 gains where geometry is good and is neutral where it is not; μ = 2 already hurts on bad
geometry, which is why 1 is the a-priori default. These are synthetic numbers for the design
choice only. Every rung-1 row now reports `neighbour_purity` (share of neighbours with the same true
class; a label-using diagnostic, never an input), and the μ = 0 ablation is a registered control.
Readout version `ncurve-v2`.

## 3. Main only, and the repository reorganised

Alex: *"I prefer to always be working on the main branch."* The one-day feature-branch policy of
2026-09-23 is dropped. Done today:

- `feat/evaluation-package-20260923` fast-forwarded into `main`; it and
  `fix/training-eval-readiness-20260918` (already contained) deleted locally and on `origin`.
- `codex/phase-b-decoder-diagnostics-20260811` had a worktree with uncommitted Phase-B work
  (17 files). Committed verbatim and tagged `hist/v2-evidence-engine/decoder-diagnostics-uncommitted-20260811`;
  its 163 MB of checkpoints moved outside Git to `halo_archives/phase-b-decoder-diagnostics-uncommitted-20260811/`;
  worktree and branch removed. An untracked `training/tokenizer/diagnostics/frontend/out/` directory
  of retired-probe outputs was deleted with the diagnostics folder without being inspected first;
  its findings are recorded in the 2026-09-12 journal entries.
- Tag `hist/v3-support-conditioned/pre-cleanup-20260924`, then removed from `main`: the unused M1
  primitives module and its test, two one-off reporting scripts with no importer, and the retired
  Future-JEPA health/monitoring scripts and frontend probes.
- The rung-2 runners moved to `evaluation/rung2_frozen/` (the roadmap's planned move), with
  three-line aliases at the old paths so recorded commands still run; `evaluation/` is now
  packaged in `pyproject.toml` (the rung-1/3 entry points lived in an unpackaged directory).
- Documentation swept against three read-only audits: `CONTRIBUTING.md` (main-only),
  `overview/{roadmap,thesis,architecture,history}.md`, `README.md`, the top of `RESULTS.md` (these
  tables are rung 2), every contract's verified date, `evaluation_protocol.md` (rungs 1 and 3),
  `design_of_record.md` (T7/T8, relation to the overview), `curriculum.md` (rewritten as the living
  v4 + pool-arm curriculum; the 2026-09-16 experiment record archived), two archive files whose
  supersession notice sat at the bottom, `history.md` (a non-existent tag name corrected), and
  stale "v3 is promoted" / "T7 is active" comments in `model/support/factory.py`.

Suite: 1,106 passed, 2 skipped (31 tests left with the archived modules).

## 4. Still open

- Level B (learnable constants, tier B, ladder step 5) — not built.
- The functional golden for the extraction check has not been produced (needs a go: it re-runs
  the old sealed evaluator on one cached cell).
- Launching the corpus-matched arms; MM-Fit in rung 1; an embodied evaluation source; pool-mean
  centring.
