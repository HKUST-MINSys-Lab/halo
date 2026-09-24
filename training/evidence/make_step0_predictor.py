"""Build the step-0 (untrained) Phase-B control predictor.

This script packages the current relational decoder and learned retriever before any Phase-B
optimization. It provides the matched step-zero arm needed to distinguish Phase-A/retrieval quality
from improvements learned during Phase B.

This script produces the missing arm: the decoder and retriever at initialisation, packaged in the
evaluator's artifact schema so `eval_enrollment` scores it through the identical path. All metadata
(bank fingerprint, vocabulary, fold, backbone, retrieval policy) is cloned from a reference
predictor so the only difference between the two arms is the learned weights.

Scoring this against a trained checkpoint isolates what Phase-B optimization contributes:

    step-0     untrained retriever + untrained relational decoder
    step-N     trained   retriever + trained   relational decoder

There is no closed-form base term: the logit is the readout. So the step-0 arm is genuinely
untrained rather than a repackaged closed-form score, and `--verify-discriminates` only checks that
it still separates candidates at all (a constant predictor would not be a usable control).

Usage:
    python -m training.evidence.make_step0_predictor \
        --reference training/evidence/outputs/.../step_1000.predictor.pt \
        --seed 20260725 --out training/evidence/outputs/.../untrained_step0.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model.evidence.relational_decoder import RelationalDecoderConfig, RelationalEvidenceDecoder
from model.evidence.patch_retrieval import PatchSubspaceRetriever
from training.evidence.policy import (
    PHASE_B_TRAINING_REGIME,
    RETRIEVAL_PROJECTION_EMA,
    RETRIEVAL_SUBSPACE_DIM,
    RETRIEVAL_SUBSPACES,
)
from training.evidence.train_patch_decoder import atomic_torch_save

_REPO = Path(__file__).resolve().parents[2]

# An untrained arm is only a usable control if it still separates candidates. This floor just has
# to exclude a numerically constant readout, not assert any particular quality.
MIN_INIT_LOGIT_SPREAD = 1e-4


def build_untrained_state(
    cfg: dict, seed: int
) -> tuple[dict, RelationalEvidenceDecoder, PatchSubspaceRetriever]:
    """Initialise decoder and retriever exactly as the trainer does at step 0."""
    torch.manual_seed(seed)
    retriever = PatchSubspaceRetriever(
        int(cfg["d_model"]),
        int(cfg.get("n_subspaces", RETRIEVAL_SUBSPACES)),
        int(cfg.get("subspace_dim", RETRIEVAL_SUBSPACE_DIM)),
        float(cfg.get("subspace_ema", RETRIEVAL_PROJECTION_EMA)),
    )
    decoder = RelationalEvidenceDecoder(RelationalDecoderConfig(
        d_model=int(cfg["d_model"]),
        n_layers=int(cfg["n_layers"]),
        n_heads=int(cfg["n_heads"]),
    ))
    state = {
        "decoder": {k: v.detach().cpu().clone() for k, v in decoder.state_dict().items()},
        "retriever": {k: v.detach().cpu().clone() for k, v in retriever.state_dict().items()},
    }
    return state, decoder, retriever


@torch.no_grad()
def candidate_logit_spread(decoder: RelationalEvidenceDecoder, *, seed: int) -> float:
    """Mean best-to-worst candidate logit spread at initialization.

    The decoder used to be a residual over a closed-form base, and this check asserted the residual
    was exactly zero so the step-0 arm reproduced that base. There is no base now — the readout is
    the whole prediction — so the meaningful check is the opposite one: the untrained model must
    still *distinguish* candidates. A spread of zero would mean a constant predictor, which is not
    a usable control arm.
    """
    decoder.eval()
    generator = torch.Generator().manual_seed(seed)
    B, q, k, C, L = 3, 4, 6, 5, 3
    d, text = decoder.cfg.d_model, decoder.cfg.text_dim

    def normal(*shape):
        return torch.randn(*shape, generator=generator)

    zq, zev = normal(B, q, d), normal(B, k, d)
    cand_text, label_text = normal(B, C, text), normal(B, L, text)
    q_mask = torch.ones(B, q, dtype=torch.bool)
    ev_mask = torch.ones(B, k, dtype=torch.bool)
    ev_mask[0, -1] = False
    label_mask = torch.ones(B, L, dtype=torch.bool)
    slot_ids = torch.arange(C + L + 1).unsqueeze(0).expand(B, -1)
    ev_slot = torch.randint(C + L + 1, (B, k), generator=generator)

    logits = decoder(
        cand_text=cand_text, label_text=label_text,
        label_mask=label_mask, slot_ids=slot_ids,
        zq=zq, q_mask=q_mask, zev=zev, ev_mask=ev_mask, ev_slot=ev_slot,
        ev_support_mask=torch.zeros(B, k, dtype=torch.bool),
        ev_score=torch.rand(B, k, generator=generator),
        q_time=torch.rand(B, q, generator=generator),
        ev_time=torch.rand(B, k, generator=generator),
        q_group=torch.zeros(B, q, dtype=torch.long),
        ev_group=torch.zeros(B, k, dtype=torch.long),
    )
    return float((logits.max(1).values - logits.min(1).values).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", type=Path, required=True,
                    help="trained predictor whose metadata (bank, vocab, fold, policy) is cloned")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260725,
                    help="initialisation seed; the retriever's random projection is the only "
                         "stochastic part of the control, so vary this to get an error bar")
    ap.add_argument("--verify-discriminates", action="store_true", default=True)
    ap.add_argument("--no-verify-discriminates", dest="verify_discriminates",
                    action="store_false")
    args = ap.parse_args()

    # weights_only matches how eval_enrollment loads a predictor, so a reference that survives
    # this load is guaranteed to survive the evaluator's.
    reference = torch.load(args.reference, map_location="cpu", weights_only=True)
    if "cfg" not in reference or "decoder" not in reference:
        raise SystemExit(f"{args.reference} is not a Phase-B predictor artifact")
    cfg = dict(reference["cfg"])

    state, decoder, _ = build_untrained_state(cfg, args.seed)
    if set(state["decoder"]) != set(reference["decoder"]):
        missing = set(reference["decoder"]) ^ set(state["decoder"])
        raise SystemExit(
            "untrained decoder does not match the reference parameter schema; the control would "
            f"not be architecture-matched. Differing keys: {sorted(missing)[:8]}"
        )

    spread = None
    if args.verify_discriminates:
        spread = candidate_logit_spread(decoder, seed=args.seed)
        if spread < MIN_INIT_LOGIT_SPREAD:
            raise SystemExit(
                f"untrained decoder emits a near-constant prediction (candidate logit spread "
                f"{spread:.3e} < {MIN_INIT_LOGIT_SPREAD:.0e}); it would not be a usable control arm"
            )

    payload = {
        **{key: value for key, value in reference.items()
           if key not in {"decoder", "retriever", "tokenizer_online", "tokenizer_ema"}},
        **state,
        # An untrained artifact has no training recipe. The regime field is what the evaluator
        # checks for *architecture and protocol* compatibility, so it carries the current regime;
        # the flags below keep the artifact from ever being mistaken for a trained checkpoint.
        "training_regime": PHASE_B_TRAINING_REGIME,
        "untrained_control": True,
        "init_seed": args.seed,
        "init_candidate_logit_spread": spread,
        "reference_predictor": str(args.reference),
        "reference_training_regime": reference.get("training_regime"),
        "checkpoint_step": 0,
        "best_step": 0,
        "checkpoint_metrics": {},
        "best_metrics": {},
        "checkpoint_selection": "untrained_step0_control",
        # A reference artifact may itself have selected the closed-form fallback. This script exists
        # to measure the architecture-matched untrained decoder, so never inherit that choice.
        "predictor_mode": "relational_decoder",
    }
    if reference.get("cfg", {}).get("tokenizer_mode") == "ema_finetune":
        raise SystemExit(
            "reference fine-tuned the tokenizer; a step-0 control would also need the untrained "
            "tokenizer path, which is not implemented here"
        )
    atomic_torch_save(payload, args.out)
    print(f"seed={args.seed} init_logit_spread="
          f"{spread if spread is None else f'{spread:.2e}'} -> {args.out}")


if __name__ == "__main__":
    main()
