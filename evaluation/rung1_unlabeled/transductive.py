"""EM-Dirichlet transductive inference — Martin, Huang, Shakeri, Pesquet, Ben Ayed,
*Transductive Zero-Shot and Few-Shot CLIP*, CVPR 2024 (10.1109/cvpr52733.2024.02722).

Ported from the released implementation, ``src/methods/{zero_shot,few_shot}/em_dirichlet.py`` at
github.com/SegoleneMartin/transductive-CLIP (default branch ``master``, fetched 2026-09-23), not
re-derived from the paper. This is the ONE implementation used by the rung-1 evaluator for every
encoder and by the trainer's unrolled loop, so what HALO is trained through is provably what the
baselines are scored with.

Inputs are *probability features*: rows on the unit simplex over the C roster entries (softmax of
temperature-scaled zero-shot scores; see ``evaluation.zero_shot.probability_features``). One
Dirichlet component per class, parameter ``alpha[k] ∈ R^C``. Block majorisation–minimisation:

* ``alpha`` update — "MM-quadratic" (paper Algorithm 1): per coordinate, the positive root of a
  quadratic built from digamma / log-gamma terms; no inner Minka iterations;
* ``v`` update — log class proportions (+1), the dual of the partition-complexity term that
  *discourages* overly balanced partitions (``lambda``);
* ``u`` update — softmax of the Dirichlet log-density plus ``lambda * v / N``.

Batched over tasks ``(B, N, C)``; padded roster slots are excluded exactly through ``candidate_mask``
(they never enter a sum, a log-gamma, or a softmax). Every operation is a ``torch`` primitive with
autograd, so with ``early_stop=False`` and fixed iteration counts the whole procedure is a finite
differentiable graph — that is what Phase 4 unrolls.

Disclosed deviation (partition weight, 2026-09-23): the reference sets ``lambda = int(C / k_eff) * N``
at few-shot and ``int(C / 5) * N`` at zero-shot, where ``C`` is the label space the probabilities range
over (1,000 for ImageNet) and ``k_eff`` the number of classes actually present in a task (3-10 in its
sampler; the zero-shot 5 is a fixed guess of it). Our label space *is* the declared roster, and the
rung-1 regime is "roster known, all of it deployed", so the non-oracle value of ``k_eff`` is the roster
size and ``lambda = N`` at every k. Porting the integer-division rule verbatim instead would switch the
partition term off for rosters under five classes and double it at ten or more — a dependence on
roster size that nothing in the method motivates. ``paper_lambda`` still implements the reference
rule for any explicit ``k_eff``.

Added term (embedding affinity, 2026-09-24): EM-Dirichlet sees each window only through its
C-dimensional probability vector, so two windows the encoder places side by side are treated
independently. In the spirit of the Laplacian term of LaplacianShot (Ziko et al., ICML 2020), each
window's assignment update also receives ``mu * log(neighbour vote)``, where the vote is the
cosine-weighted mean of the **text probability vectors** of its ``knn`` nearest neighbours **in the
encoder's own embedding space**. It enters exactly like the partition term: a log prior, local
instead of global, with ``mu = 1`` giving the two equal standing. ``mu = 0`` (or no embeddings) is
the published method bit for bit. This is what makes rung 1 read the encoder's geometry rather
than only its text head.

The vote uses the neighbours' fixed text evidence, not their current assignments. A synthetic
check (2026-09-24; six classes, noisy text head) showed why: voting on the current assignments
gains on clean clusters but, when neighbourhoods are uninformative (random, or grouped by subject),
feeds errors back each iteration and collapses every window onto one class (0.84 -> 0.17, chance),
which would punish an encoder for its geometry far beyond what its geometry deserves. Voting on the
fixed evidence gains +4 to +6 points on clean clusters at ``mu = 1`` and is neutral (-0.1 to +0.4)
on uninformative ones; ``mu = 2`` already costs up to 15 points there, which is why the a-priori
default is 1.

Disclosed deviation (cluster assignment): at zero-shot, the reference initialises ``u`` from the probability features
(component k starts as class k) and *still* re-assigns clusters to classes afterwards by graph
matching on cluster prototypes (paper §4.3). Here ``assign_clusters(mode="identity")`` keeps the
initial identity (differentiable, needed for training) and ``mode="graph"`` / ``"basic"`` port the
reference's post-hoc matching for inference; both are reported on one cell.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

EPS = 1e-15
# config/methods_config/em_dirichlet.yaml: iter 20, iter_mm 1000 (early stop on a 1e-11 relative
# criterion checked every 50 inner steps). Training-time unrolling uses fixed, short counts.
INFERENCE_DEFAULTS = {"n_iter": 20, "n_iter_mm": 1000, "early_stop": True, "mu": 1.0, "knn": 10}
UNROLL_DEFAULTS = {"n_iter": 5, "n_iter_mm": 20, "early_stop": False, "mu": 1.0, "knn": 10}


def paper_lambda(n_classes: torch.Tensor | int, n_query: torch.Tensor | int, *,
                 k_eff: torch.Tensor | int | None = None) -> torch.Tensor:
    """The released code's rule ``int(C / k_eff) * N``, with ``k_eff = 5`` when omitted (its
    zero-shot constant). :func:`transduce` passes ``k_eff = C`` (see the module docstring), so the
    rule as used here is ``lambda = N``; an explicit ``k_eff`` reproduces the reference exactly,
    including its integer division (``C < k_eff`` gives ``lambda = 0``)."""
    c = torch.as_tensor(n_classes, dtype=torch.float32)
    divisor = torch.as_tensor(5 if k_eff is None else k_eff, dtype=torch.float32, device=c.device)
    return torch.floor(c / divisor) * torch.as_tensor(n_query, dtype=torch.float32, device=c.device)


def _curvature(alpha: torch.Tensor, log_gamma_1: torch.Tensor, zero_value: torch.Tensor):
    digam = torch.polygamma(0, alpha + 1)
    curv = torch.where(
        alpha > 1e-11,
        torch.abs(2 * (log_gamma_1 - torch.lgamma(alpha + 1) + digam * alpha) / alpha ** 2),
        zero_value,
    )
    return curv, digam


def _mm_step(alpha: torch.Tensor, y_cst: torch.Tensor, dim_mask: torch.Tensor,
             log_gamma_1: torch.Tensor, zero_value: torch.Tensor) -> torch.Tensor:
    """One MM-quadratic update. Pure elementwise math plus one masked sum."""
    curv, digam = _curvature(alpha, log_gamma_1, zero_value)
    alpha_sum = (alpha * dim_mask).sum(-1, keepdim=True)
    b = digam - torch.polygamma(0, alpha_sum) - curv * alpha - y_cst
    return (-b + torch.sqrt(b * b + 4 * curv)) / (2 * curv)


_COMPILED_MM_STEP = None


def _mm_step_fn(device: torch.device):
    """On CUDA the ~12 tiny kernels of one MM step are fused by ``torch.compile``: the loop is
    kernel-launch-bound (tensors are (B, C, C) with C <= ~20), and fusion cut a step from 0.094 to
    0.034 ms with outputs equal to 5e-7 (2026-09-25 profile: this loop was 88 % of rung-1 training
    wall time). CPU stays eager, so tests and CPU runs need no compiler toolchain."""
    global _COMPILED_MM_STEP
    if device.type != "cuda":
        return _mm_step
    if _COMPILED_MM_STEP is None:
        _COMPILED_MM_STEP = torch.compile(_mm_step, dynamic=True)
    return _COMPILED_MM_STEP


def update_alpha(alpha: torch.Tensor, y_cst: torch.Tensor, dim_mask: torch.Tensor, *,
                 n_iter_mm: int, early_stop: bool, tol: float = 1e-11) -> torch.Tensor:
    """MM-quadratic (Algorithm 1). ``alpha, y_cst: (B, K, C)``; ``dim_mask: (B, 1, C)``."""
    one = torch.ones((), dtype=alpha.dtype, device=alpha.device)
    log_gamma_1 = torch.lgamma(one)
    zero_value = torch.polygamma(1, one)
    step_fn = _mm_step_fn(alpha.device)
    for step in range(int(n_iter_mm)):
        alpha_new = step_fn(alpha, y_cst, dim_mask, log_gamma_1, zero_value)
        if early_stop and step > 0 and step % 50 == 0:
            criterion = (alpha_new - alpha).norm() ** 2 / alpha.norm().clamp_min(EPS) ** 2
            alpha = alpha_new
            if float(criterion) < tol:
                break
        else:
            alpha = alpha_new
    return alpha


def dirichlet_log_density(log_z: torch.Tensor, alpha: torch.Tensor, dim_mask: torch.Tensor) -> torch.Tensor:
    """``(B, N, K)`` log Dirichlet density of each row under each component, padded dims excluded."""
    l1 = torch.lgamma((alpha * dim_mask).sum(-1))                              # (B, K)
    l2 = -(torch.lgamma(alpha) * dim_mask).sum(-1)                              # (B, K)
    l3 = torch.einsum("bnc,bkc->bnk", log_z, (alpha - 1) * dim_mask)            # (B, N, K)
    return l1[:, None, :] + l2[:, None, :] + l3


def knn_affinity(embeddings: torch.Tensor, row_mask: torch.Tensor, knn: int, *,
                 chunk: int = 2048) -> tuple[torch.Tensor, torch.Tensor]:
    """``(B, N, k)`` neighbour indices and non-negative cosine weights among the valid rows.

    A window is never its own neighbour; padded rows are never neighbours and have no neighbours.
    Weights are ``max(cos, 0)`` so dissimilar rows cannot vote; the selection is by cosine rank,
    and gradients reach the embeddings through the weights. Rows are processed in chunks so a
    16k-window cell never materialises an N x N matrix.
    """
    B, N, _ = embeddings.shape
    k = min(int(knn), N - 1)
    if k < 1:
        empty = torch.zeros((B, N, 0), device=embeddings.device)
        return empty.long(), empty
    unit = F.normalize(embeddings.float(), dim=-1)
    valid = row_mask.to(device=embeddings.device, dtype=torch.bool)
    idx_parts, weight_parts = [], []
    for start in range(0, N, chunk):
        stop = min(start + chunk, N)
        cos = torch.einsum("bqd,bnd->bqn", unit[:, start:stop], unit)             # (B, q, N)
        blocked = ~valid[:, None, :].expand_as(cos).clone()
        rows = torch.arange(start, stop, device=cos.device)
        blocked[:, rows - start, rows] = True                                     # no self-loops
        cos = cos.masked_fill(blocked, float("-inf"))
        top, idx = cos.topk(k, dim=-1)
        weight = torch.where(torch.isfinite(top), top.clamp_min(0.0), torch.zeros_like(top))
        weight = weight * valid[:, start:stop, None].to(weight.dtype)
        idx_parts.append(idx)
        weight_parts.append(weight)
    return torch.cat(idx_parts, dim=1), torch.cat(weight_parts, dim=1)


def neighbour_vote(u: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Cosine-weighted mean of the neighbours' assignments, ``(B, N, C)``, and a ``(B, N)`` mask of
    rows that have any positive-weight neighbour."""
    B, N, C = u.shape
    gathered = torch.gather(u, 1, idx.reshape(B, -1, 1).expand(-1, -1, C)).reshape(B, N, idx.shape[-1], C)
    total = weight.sum(-1)
    vote = (weight.unsqueeze(-1) * gathered).sum(2) / total.clamp_min(EPS).unsqueeze(-1)
    return vote, total > 0


@dataclass
class Transduction:
    u: torch.Tensor          # (B, N, C) soft assignments after the last update
    logits: torch.Tensor     # (B, N, C) the pre-softmax scores that produced ``u``
    alpha: torch.Tensor      # (B, C, C) Dirichlet parameters per component
    v: torch.Tensor          # (B, C) log class proportions (+1)
    lam: torch.Tensor        # (B,) the partition-term weight actually used
    neighbours: torch.Tensor | None = None   # (B, N, knn) embedding-affinity indices, if used


def transduce(
    z: torch.Tensor,
    *,
    candidate_mask: torch.Tensor | None = None,
    row_mask: torch.Tensor | None = None,
    support_z: torch.Tensor | None = None,
    support_onehot: torch.Tensor | None = None,
    n_iter: int = 20,
    n_iter_mm: int = 1000,
    early_stop: bool = True,
    lam: torch.Tensor | float | None = None,
    k_eff: torch.Tensor | int | None = None,
    embeddings: torch.Tensor | None = None,
    mu: torch.Tensor | float = 0.0,
    knn: int = 10,
) -> Transduction:
    """Run EM-Dirichlet over a batch of tasks.

    ``z``: ``(B, N, C)`` probability features of the unlabelled pool (rows sum to 1 over valid
    slots). ``candidate_mask``: ``(B, C)`` bool, valid roster slots (default all). ``support_z`` /
    ``support_onehot``: ``(B, S, C)`` labelled supports for the few-shot variant, or ``None`` for
    zero-shot. ``lam``: partition weight; ``None`` applies :func:`paper_lambda` with ``k_eff``, which
    defaults to each task's roster size (``lambda = N``; see the module docstring). ``embeddings``:
    ``(B, N, D)`` encoder features of the same rows; with ``mu != 0`` they add the embedding-affinity
    term over each row's ``knn`` nearest neighbours. The function-level default ``mu = 0`` is the
    published method; the evaluator and trainer defaults (``INFERENCE_DEFAULTS`` /
    ``UNROLL_DEFAULTS``) switch the term on.

    Always computed in float32 with autocast disabled: the reference runs in float32, and the
    log-gamma / digamma terms and the log-density einsums lose the assignment signal in bf16.
    """
    if torch.is_autocast_enabled(z.device.type):
        with torch.autocast(device_type=z.device.type, enabled=False):
            return transduce(
                z.float(), candidate_mask=candidate_mask, row_mask=row_mask,
                support_z=None if support_z is None else support_z.float(),
                support_onehot=None if support_onehot is None else support_onehot.float(),
                n_iter=n_iter, n_iter_mm=n_iter_mm, early_stop=early_stop, lam=lam, k_eff=k_eff,
                embeddings=embeddings, mu=mu, knn=knn,
            )
    z = z.float()
    if support_z is not None:
        support_z = support_z.float()
    if support_onehot is not None:
        support_onehot = support_onehot.float()
    if z.ndim != 3:
        raise ValueError("z must be (B, N, C)")
    B, N, C = z.shape
    if N < 1 or C < 2:
        raise ValueError("a task needs at least one row and two roster slots")
    mask = (torch.ones((B, C), dtype=torch.bool, device=z.device) if candidate_mask is None
            else candidate_mask.to(device=z.device, dtype=torch.bool))
    if mask.shape != (B, C):
        raise ValueError("candidate_mask must be (B, C)")
    dim_mask = mask.to(z.dtype)[:, None, :]                                     # (B, 1, C)
    rows_valid = (torch.ones((B, N), dtype=torch.bool, device=z.device) if row_mask is None
                  else row_mask.to(device=z.device, dtype=torch.bool))
    if rows_valid.shape != (B, N):
        raise ValueError("row_mask must be (B, N)")
    row_w = rows_valid.to(z.dtype)[..., None]                                   # (B, N, 1)
    n_valid = rows_valid.sum(dim=1).clamp_min(1).to(z.dtype)                     # (B,)
    log_z = torch.log(z.clamp_min(0) + EPS) * dim_mask                          # padded dims → 0
    few_shot = support_z is not None
    if few_shot:
        if support_onehot is None or support_z.shape[0] != B or support_onehot.shape != support_z.shape:
            raise ValueError("support_z and support_onehot must be (B, S, C)")
        log_s = torch.log(support_z.clamp_min(0) + EPS) * dim_mask
        y_s_sum = support_onehot.sum(dim=1)                                     # (B, C)
    if lam is None:
        roster = mask.sum(-1)
        lam_t = paper_lambda(roster, n_valid, k_eff=roster if k_eff is None else k_eff) \
            .to(device=z.device, dtype=z.dtype)
    else:
        lam_t = torch.as_tensor(lam, dtype=z.dtype, device=z.device).expand(B).clone()
    neg_inf = torch.finfo(z.dtype).min
    affinity = None
    use_affinity = embeddings is not None and not (isinstance(mu, (int, float)) and float(mu) == 0.0)
    if use_affinity:
        if embeddings.shape[:2] != (B, N):
            raise ValueError("embeddings must be (B, N, D) and aligned with z")
        affinity = knn_affinity(embeddings, rows_valid, knn)
        mu_t = torch.as_tensor(mu, dtype=z.dtype, device=z.device)
        # The neighbours vote with their *text evidence* z, computed once, not with the current
        # assignments u: a vote on u feeds each iteration's errors back into the next and, when the
        # neighbourhoods are uninformative, collapses every row onto one class.
        vote, has_neighbours = neighbour_vote(z * dim_mask * row_w, *affinity)
        local_prior = torch.log(vote + EPS) * has_neighbours.unsqueeze(-1).to(z.dtype)

    v = torch.zeros((B, C), dtype=z.dtype, device=z.device)
    u = z * dim_mask * row_w                                                    # init: features
    alpha = torch.ones((B, C, C), dtype=z.dtype, device=z.device)
    logits = torch.log(u + EPS)
    for _ in range(int(n_iter)):
        u_sum = u.sum(dim=1)                                                    # (B, K)
        if few_shot:
            numerator = torch.einsum("bsk,bsc->bkc", support_onehot.to(z.dtype), log_s) \
                + torch.einsum("bnk,bnc->bkc", u, log_z)
            y_cst = numerator / (y_s_sum + u_sum).clamp_min(EPS)[..., None]
            alpha = update_alpha(alpha, y_cst, dim_mask, n_iter_mm=n_iter_mm, early_stop=early_stop)
        else:
            nonzero = (u_sum > EPS)[..., None]                                  # (B, K, 1)
            y_cst = torch.einsum("bnk,bnc->bkc", u, log_z) / u_sum.clamp_min(EPS)[..., None]
            y_cst = torch.where(nonzero, y_cst, torch.full_like(y_cst, -10.0))
            alpha_new = update_alpha(alpha, y_cst, dim_mask, n_iter_mm=n_iter_mm, early_stop=early_stop)
            alpha = torch.where(nonzero, alpha_new, alpha)                      # empty clusters keep alpha
        v = torch.log(u_sum / n_valid[:, None] + EPS) + 1
        logits = dirichlet_log_density(log_z, alpha, dim_mask) \
            + lam_t[:, None, None] * v[:, None, :] / n_valid[:, None, None]
        if affinity is not None:
            logits = logits + mu_t * local_prior
        logits = logits.masked_fill(~mask[:, None, :], neg_inf)
        u = torch.softmax(logits, dim=-1) * row_w
    return Transduction(u=u, logits=logits, alpha=alpha, v=v, lam=lam_t,
                        neighbours=None if affinity is None else affinity[0])


def assign_clusters(u: torch.Tensor, z: torch.Tensor, *, mode: str = "identity",
                    candidate_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Cluster → class assignment. ``identity`` keeps component k = class k (the text
    initialisation); ``basic`` / ``graph`` port the reference's post-hoc matching on cluster
    prototypes in probability-feature space (``compute_basic_matching`` / ``compute_graph_matching``)."""
    preds = u.argmax(dim=-1)
    if mode == "identity":
        return preds
    if mode not in {"basic", "graph"}:
        raise ValueError(f"unknown assignment mode {mode!r}")
    B, N, C = z.shape
    mask = (torch.ones((B, C), dtype=torch.bool, device=z.device) if candidate_mask is None
            else candidate_mask.to(device=z.device, dtype=torch.bool))
    out = preds.clone()
    for b in range(B):
        onehot = F.one_hot(preds[b], C).to(z.dtype)                             # (N, K)
        sizes = onehot.sum(dim=0)                                               # (K,)
        prototypes = (onehot.T @ z[b]) / sizes.clamp_min(EPS)[:, None]          # (K, C)
        prototypes = prototypes * mask[b].to(z.dtype)[None, :]
        nonempty = (sizes > EPS).nonzero().flatten()
        mapping = torch.arange(C, device=z.device)
        if mode == "basic":
            mapping[nonempty] = prototypes[nonempty].argmax(dim=-1)
        else:
            cost = -prototypes[nonempty][:, mask[b]].detach().cpu().numpy()
            rows, cols = linear_sum_assignment(cost)
            valid_slots = mask[b].nonzero().flatten()
            mapping[nonempty[rows]] = valid_slots[torch.as_tensor(cols, device=z.device)]
        out[b] = mapping[preds[b]]
    return out


def transduce_numpy(z: np.ndarray, *, support_z: np.ndarray | None = None,
                    support_labels: np.ndarray | None = None, assignment: str = "identity",
                    embeddings: np.ndarray | None = None,
                    device: torch.device | None = None, **kwargs) -> tuple[np.ndarray, np.ndarray, dict]:
    """Single-task convenience for the evaluator: ``z (N, C)`` → (predicted ids, soft u, info).
    ``embeddings (N, D)`` enable the affinity term when ``mu != 0``; ``info["neighbours"]`` then
    holds the ``(N, knn)`` neighbour indices (for the neighbour-purity diagnostic)."""
    device = device or torch.device("cpu")
    zt = torch.as_tensor(np.asarray(z, dtype=np.float32), device=device)[None]
    if embeddings is not None:
        kwargs["embeddings"] = torch.as_tensor(np.asarray(embeddings, dtype=np.float32), device=device)[None]
    support_zt = support_onehot = None
    if support_z is not None:
        support_zt = torch.as_tensor(np.asarray(support_z, dtype=np.float32), device=device)[None]
        support_onehot = F.one_hot(
            torch.as_tensor(np.asarray(support_labels, dtype=np.int64), device=device), zt.shape[-1],
        ).to(zt.dtype)[None]
    with torch.no_grad():
        result = transduce(zt, support_z=support_zt, support_onehot=support_onehot, **kwargs)
        preds = assign_clusters(result.u, zt, mode=assignment)
    info = {"lam": float(result.lam[0]), "assignment": assignment,
            "n_iter": int(kwargs.get("n_iter", 20)), "n_iter_mm": int(kwargs.get("n_iter_mm", 1000)),
            "few_shot": support_z is not None,
            "mu": float(kwargs.get("mu", 0.0)) if result.neighbours is not None else 0.0,
            "knn": int(result.neighbours.shape[-1]) if result.neighbours is not None else 0,
            "neighbours": None if result.neighbours is None else result.neighbours[0].cpu().numpy(),
            "cluster_sizes": result.u[0].sum(0).cpu().numpy().round(3).tolist()}
    return preds[0].cpu().numpy(), result.u[0].cpu().numpy(), info
