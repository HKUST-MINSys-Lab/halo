"""
Augmentation strategies for IMU time series data.

Implements the physically plausible transforms used by HALO Phase-A pretraining.
"""

import re
import torch
import numpy as np
from typing import Optional, List

# Sensor-type token detector for triad location inference. Longest alternative first so
# 'accelerometer'/'accel' win over 'acc' (else 'acc' truncates 'accel' and mis-locates); the
# trailing \d* absorbs dual-range suffixes like pamap2's acc16/acc6.
_SENSOR_TOKEN_RE = re.compile(
    r'(accelerometer|accel|acc|gyroscope|gyro|magnetometer|magnet|mag|orientation|orient|ori)\d*')



# =============================================================================
# Unified, configurable augmentation system (V2)
# =============================================================================
# Every augmentation is switched on/off and tuned from a single
# AugmentationConfig, so it is obvious at a glance which augmentations are
# active. Physics/metadata-changing augmentations (gravity, rate, channel
# dropout) also update the per-sample channel description / sampling rate so the
# model's channel-text conditioning stays consistent with the augmented signal
# (the loader appends the "sampled at NHz" suffix from sample.sampling_rate).

import random as _random
from contextlib import contextmanager
from dataclasses import dataclass, field
from dataclasses import fields as _dc_fields
from fractions import Fraction
from scipy import signal as _sps


# ---- Per-augmentation config specs (each has `enabled` + `p` + its params) ----
@dataclass
class JitterCfg:
    """Additive Gaussian sensor noise, scaled by each channel's local signal std."""
    enabled: bool = False
    p: float = 0.5
    sigma: float = 0.05


@dataclass
class ScaleCfg:
    """Per-channel amplitude scaling (gain/calibration variance)."""
    enabled: bool = False
    p: float = 0.5
    low: float = 0.9
    high: float = 1.1


@dataclass
class GravityCfg:
    """P1 — add/remove gravity. Subtracts a low-pass gravity estimate to
    manufacture the iOS `userAcceleration` (gravity-removed) representation the
    training corpus otherwise lacks; annotates the acc channel text accordingly."""
    enabled: bool = False
    p: float = 0.5
    cutoff_hz: float = 0.4   # gravity is quasi-DC; human motion energy is > ~0.5 Hz
    order: int = 2


@dataclass
class Rotation3dCfg:
    """P2b — full uniform-random SO(3) rotation of every co-located sensor triad
    (acc+gyro+mag share one rotation per body location). This is the placement/
    orientation-invariance lever (cf. UniMTS): the gravity DC vector rotates WITH the
    accel signal, so the model learns 'gravity can point any direction' rather than
    memorizing each dataset's fixed orientation. Principled ONLY because the filterbank
    now carries a signed DC feature (else full SO(3) scrambles an unrepresented gravity
    cue — the reason plain rotation was originally disabled). Applies to ANY real triad: a
    gravity-removed accel is still a proper 3-vector and rotates correctly (its DC≈0, so
    nothing is scrambled), so gravity-removed streams are NOT excluded (F12). The only
    genuinely non-rotatable case — per-axis min-max normalized data — is excluded at curation
    time (recgym was dropped), not here."""
    enabled: bool = False
    p: float = 0.5


@dataclass
class RateCfg:
    """P3 — anti-aliased resample to a random sampling rate (teaches rate-invariance).
    Updates sample.sampling_rate; the rate is conveyed NUMERICALLY to the filterbank
    (via the collate's per-sample ``rate``), NOT through text — the channel text carries no Hz
    token (rate/duration were stripped from the template), so descriptions stay byte-identical."""
    enabled: bool = False
    p: float = 0.5
    min_hz: float = 15.0
    max_hz: float = 100.0
    min_samples: int = 32   # skip if the resampled window would be shorter than this


@dataclass
class ChannelDropoutCfg:
    """P4 — drop a whole sensor group (e.g. gyro) so the model is robust to
    deployments that expose only an accelerometer. Updates channel list + text.

    Note: reduces a sample's channel count, so within a ChannelBucketBatchSampler
    bucket batches become channel-heterogeneous. This is correct (collate pads to
    the batch max and the per-sample channel_mask isolates padding) but slightly
    reduces the sampler's padding-efficiency; keep `p` modest for that reason."""
    enabled: bool = False
    p: float = 0.3
    groups: tuple = ("gyro",)   # channel-name substrings eligible for dropping


@dataclass
class WindowCropCfg:
    """P5 — random temporal crop: keep a contiguous sub-window of a random duration, so the model
    sees variable OBSERVATION LENGTHS (session-length invariance). The encoder is a set over patches
    with physical-time positions + a patch-padding mask, so a shorter window is simply FEWER real
    patches — nothing structural forces the 6 s corpus window. Layout-breaking (changes the token
    count), so it is HALO-only like rate/channel_dropout: a fixed-window baseline cannot ingest it."""
    enabled: bool = False
    p: float = 0.5
    min_frac: float = 0.5    # keep at least this fraction of the window's timesteps
    min_samples: int = 32    # never crop below this many samples (one resolvable patch)


@dataclass
class ChannelTextPhraseCfg:
    """Paraphrase each channel description: swap ONLY sensor-family / axis surface forms and
    wrap in a template. Placement, units, and gravity state are left verbatim, so the
    load-bearing semantics are preserved."""
    enabled: bool = False
    p: float = 0.5   # fraction of samples whose channel descriptions get paraphrased


@dataclass
class ChannelTextDropoutCfg:
    """Neutralize a random subset of channel descriptions (KEEP the signal) so the model is
    robust to unknown/missing placement metadata. Never neutralizes more than `max_frac`."""
    enabled: bool = False
    p: float = 0.15          # fraction of samples that get any channel-text neutralized
    max_frac: float = 0.5    # never neutralize more than this fraction of a sample's channels
    neutral: str = "an inertial sensor channel"
    # Separate neutral for the AXIS-only role slot: the channel-level neutral above ("an inertial
    # sensor channel") is not an axis, so dropping a role to it stated the wrong KIND of fact.
    role_neutral: str = "an unspecified axis"


@dataclass
class SensorTextDropoutCfg:
    """Neutralize per-SENSOR identity text (device/placement/gravity) so the factored model learns
    to fall back gracefully when config metadata is missing (F7). Distinct from channel-text dropout,
    which only touches the per-channel ROLE text and never the sensor identity. Bounded: with >=2
    sensors it keeps >=1 described; a single-sensor stream may be fully neutralized (the
    fully-unconditioned case), but only within the low sample rate `p`."""
    enabled: bool = False
    p: float = 0.1           # fraction of samples that get any sensor-text neutralized
    max_frac: float = 0.5    # never neutralize more than this fraction of a sample's sensors (>=2 case)
    neutral: str = "an inertial sensor"
    # Share the decision (fire + which sensors) across a window's two VICReg views. MUST stay True
    # for the config-conditional thesis: with independent draws, ~2p(1-p) of positive pairs describe
    # the config in one view and neutralise it in the other, and the invariance term then trains
    # embed(config) == embed(no config) — pressure to IGNORE the config. Set False only to ablate
    # that effect deliberately.
    shared_across_views: bool = True


# Conservative, meaning-preserving substitutions for channel-description paraphrase. Only
# sensor-family + axis SURFACE FORMS are swapped; placement/units/gravity are never touched.
_CH_SYNONYMS = [
    (r"\baccelerometer\b", ["accelerometer", "accelerometer sensor"]),
    (r"\bacceleration\b", ["acceleration", "acceleration signal"]),
    (r"\bgyroscope\b", ["gyroscope", "gyro", "angular rate sensor"]),
    (r"\bangular velocity\b", ["angular velocity", "angular rate", "rotational velocity"]),
    (r"\bmagnetometer\b", ["magnetometer", "magnetic field sensor"]),
    (r"\bmagnetic field\b", ["magnetic field", "magnetic flux"]),
    (r"\bx-axis\b", ["x-axis", "x axis"]),
    (r"\by-axis\b", ["y-axis", "y axis"]),
    (r"\bz-axis\b", ["z-axis", "z axis"]),
    (r"\bmounted\b", ["mounted", "worn", "placed"]),
]
_CH_TEMPLATES = ["{}", "channel: {}", "sensor channel — {}", "signal from {}", "this channel measures {}"]


def _synonym_swap(desc: str) -> str:
    """Sensor/axis SYNONYM substitution only — no template wrapper.
    re.escape not needed — replacements are plain words; placement/units are never matched."""
    import re
    out = desc
    for pat, options in _CH_SYNONYMS:
        if re.search(pat, out, flags=re.I):
            out = re.sub(pat, _random.choice(options), out, flags=re.I)
    return out


def _paraphrase_channel(desc: str) -> str:
    """Surface-form paraphrase of one CHANNEL description (synonyms + channel template)."""
    return _random.choice(_CH_TEMPLATES).format(_synonym_swap(desc))


def _paraphrase_sensor(desc: str) -> str:
    """Surface-form paraphrase of one SENSOR-identity string (synonyms only, NO template).

    The `_CH_TEMPLATES` wrappers are CHANNEL-level ("this channel measures {}", "sensor channel — {}")
    and are false when applied to a sensor: a sensor is not a channel. Before this split they produced
    text like "this channel measures a watch accelerometer on the wrist". Sensor identity is exactly
    the config fact the model is supposed to read, so it gets synonym variation only.
    """
    return _synonym_swap(desc)


@dataclass
class AugmentationConfig:
    """Single source of truth for which augmentations run and how strong they are.

    ``phase_a()`` is the training recipe and ``none()`` disables every transform.
    Print `cfg.summary()` to see the ON/OFF table.
    """
    jitter: JitterCfg = field(default_factory=JitterCfg)
    scale: ScaleCfg = field(default_factory=ScaleCfg)
    gravity: GravityCfg = field(default_factory=GravityCfg)
    rotation_3d: Rotation3dCfg = field(default_factory=Rotation3dCfg)
    rate: RateCfg = field(default_factory=RateCfg)
    channel_dropout: ChannelDropoutCfg = field(default_factory=ChannelDropoutCfg)
    window_crop: WindowCropCfg = field(default_factory=WindowCropCfg)
    # Text augmentations (unified here so ALL augmentation lives in one config).
    channel_text_phrase: ChannelTextPhraseCfg = field(default_factory=ChannelTextPhraseCfg)
    channel_text_dropout: ChannelTextDropoutCfg = field(default_factory=ChannelTextDropoutCfg)
    sensor_text_dropout: SensorTextDropoutCfg = field(default_factory=SensorTextDropoutCfg)

    # Application order: metadata/physics-changing first, then value-space, then TEXT last
    # (so channel-text augs see the final, physics-mutated channel set/descriptions).
    # rotation_3d runs BEFORE gravity so a gravity-carrying accel is rotated with its gravity DC
    # intact (teaching gravity-direction augmentation) before gravity may be removed; rate runs
    # after gravity/rotation.
    # Configuration transforms run before independent nuisance transforms. This order is
    # load-bearing: if window_crop runs first, two independently cropped views can make `rate` pass
    # its minimum-length guard in one view and skip in the other despite sharing the same RNG seed.
    ORDER = ("channel_dropout", "rotation_3d", "gravity", "rate", "window_crop",
             "scale", "jitter", "channel_text_phrase", "channel_text_dropout",
             "sensor_text_dropout")

    # ------------------------------------------------------------------------------------------
    # NUISANCE vs CONFIG (docs/design/DESIGN_OF_RECORD.md)
    #
    # The grouping rule is "did the underlying ACQUISITION change", NOT "which field was touched".
    #
    #   * NUISANCE — same acquisition, different realization. Two views may draw INDEPENDENTLY and
    #     VICReg's invariance MSE applies: the representation genuinely should not move.
    #   * CONFIG   — the acquisition itself is different (sampling rate, units, gravity
    #     convention, which sensors are present). ONE draw per window per step, applied identically
    #     to EVERY view including the JEPA teacher's. Demanding invariance across these is demanding
    #     the model discard exactly the axes it conditions on.
    #
    # Measured 2026-08-11: with the old undifferentiated stack, config conditioning contributed
    # +0.0086 kNN-BA against a 0.0065 noise floor, with the sign flipping on 2 of 4 datasets — i.e.
    # nothing (training/tokenizer/outputs/parity_ablation/parity.json). This split is the structural
    # half of the fix; descriptor-mask JEPA is the other half.
    #
    # Text paraphrase is NUISANCE, deliberately: it rewords an IDENTICAL configuration. Keeping the
    # invariance pressure on it is what forces the model off memorising ~20 fixed strings and onto
    # text semantics — the only thing that can generalize to an unseen placement description.
    # ------------------------------------------------------------------------------------------
    # Nuisance transforms may differ across positive views. Rotation is intentionally excluded:
    # gravity-frame orientation can carry activity signal, so rotation invariance is an explicit
    # ablation rather than an assumption in the reference recipe.
    NUISANCE_GROUP = ("scale", "jitter", "channel_text_phrase")
    # sensor_text_dropout is listed here (rather than omitted) so the two groups partition ORDER: if
    # anyone re-enables it, it lands in the config group by default rather than silently reacquiring
    # VICReg invariance pressure — the failure this whole split exists to prevent.
    CONFIG_GROUP = ("channel_dropout", "rotation_3d", "gravity", "rate", "window_crop",
                    "channel_text_dropout", "sensor_text_dropout")

    @classmethod
    def phase_a(
        cls,
        *,
        rotation_p: float = 0.0,
        rate_p: float = 0.0,
        channel_dropout_p: float = 0.0,
        jitter_p: float = 0.0,
        scale_p: float = 0.0,
        gravity_p: float = 0.0,
        channel_text_phrase_p: float = 0.0,
        channel_text_dropout_p: float = 0.0,
    ) -> "AugmentationConfig":
        """Explicit controlled recipe: a transform is on exactly when its probability is nonzero.

        `jitter_p`/`scale_p` are the NUISANCE-suppressing transforms. They are separated from
        `rotation_p` deliberately. Measured 2026-08-18: with every augmentation off, the subject
        probe on the deployed retrieval row rises monotonically with training (0.342 random init ->
        0.525 at step 1k -> 0.584 at step 7.5k) while the fully-augmented 4k checkpoint held it at
        0.315. Nothing was suppressing subject and device idiosyncrasy, so the encoder spent
        capacity on who is wearing the device. Jitter and scale perturb amplitude and sensor noise
        WITHOUT touching gravity-frame orientation, so they suppress that nuisance without repeating
        the independent-SO(3) failure (which is a CONFIG transform, shared across views).
        """
        cfg = cls()
        probabilities = (
            (rotation_p, "rotation_p"), (rate_p, "rate_p"),
            (channel_dropout_p, "channel_dropout_p"),
            (jitter_p, "jitter_p"), (scale_p, "scale_p"), (gravity_p, "gravity_p"),
            (channel_text_phrase_p, "channel_text_phrase_p"),
            (channel_text_dropout_p, "channel_text_dropout_p"),
        )
        for value, name in probabilities:
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        cfg.rotation_3d.enabled = rotation_p > 0
        cfg.rotation_3d.p = float(rotation_p)
        cfg.rate.enabled = rate_p > 0
        cfg.rate.p = float(rate_p)
        cfg.channel_dropout.enabled = channel_dropout_p > 0
        cfg.channel_dropout.p = float(channel_dropout_p)
        cfg.jitter.enabled = jitter_p > 0
        cfg.jitter.p = float(jitter_p)
        cfg.scale.enabled = scale_p > 0
        cfg.scale.p = float(scale_p)
        # gravity/channel_text_dropout are CONFIG-group: one draw per window, shared across every
        # view, so they widen the ACQUISITION distribution the model sees rather than demanding
        # invariance to it. channel_text_phrase is NUISANCE-group by design -- paraphrase pressure
        # is what forces text semantics instead of memorising ~20 fixed description strings.
        cfg.gravity.enabled = gravity_p > 0
        cfg.gravity.p = float(gravity_p)
        cfg.channel_text_phrase.enabled = channel_text_phrase_p > 0
        cfg.channel_text_phrase.p = float(channel_text_phrase_p)
        cfg.channel_text_dropout.enabled = channel_text_dropout_p > 0
        cfg.channel_text_dropout.p = float(channel_text_dropout_p)
        return cfg

    def split_by_group(self) -> tuple["AugmentationConfig", "AugmentationConfig"]:
        """Return ``(nuisance_only, config_only)`` copies of this config.

        The caller draws the config view ONCE per window per step and reuses it for every view
        (VICReg A, VICReg B, JEPA teacher); the nuisance view is drawn per view. Splitting here
        rather than at the call site keeps the group membership in one place.
        """
        import copy
        nuisance, config = copy.deepcopy(self), copy.deepcopy(self)
        for name in self.CONFIG_GROUP:
            getattr(nuisance, name).enabled = False
        for name in self.NUISANCE_GROUP:
            getattr(config, name).enabled = False
        return nuisance, config

    @classmethod
    def phase_b_generic(cls) -> "AugmentationConfig":
        """Independent acquisition variation for Phase-B query/support executions.

        Phase B keeps duration, channel layout, gravity state, and text fixed so source patch
        ordinals remain valid. Subject character is applied separately before this transform.
        """
        cfg = cls()
        cfg.jitter.enabled = True
        cfg.jitter.p = 0.7
        cfg.jitter.sigma = 0.025
        cfg.scale.enabled = True
        cfg.scale.p = 0.7
        cfg.scale.low = 0.95
        cfg.scale.high = 1.05
        cfg.rotation_3d.enabled = True
        cfg.rotation_3d.p = 0.8
        return cfg

    @classmethod
    def none(cls) -> "AugmentationConfig":
        cfg = cls()
        for name in cls.ORDER:
            getattr(cfg, name).enabled = False
        return cfg

    def summary(self) -> str:
        lines = ["Augmentation config (ON/OFF + params):"]
        for name in self.ORDER:
            spec = getattr(self, name)
            params = ", ".join(
                f"{f.name}={getattr(spec, f.name)}"
                for f in _dc_fields(spec) if f.name not in ("enabled", "p")
            )
            flag = "ON " if spec.enabled else "off"
            lines.append(f"  [{flag}] {name:16s} p={spec.p:<4} {params}")
        return "\n".join(lines)


@dataclass
class IMUSample:
    """Per-sample carrier threaded through the augmenter. Physics augmentations mutate
    sampling_rate / channel metadata and the TEXT augmentations mutate both legacy per-channel
    descriptions and, when supplied, factored role/sensor descriptions."""
    data: "torch.Tensor"              # (T, C)
    channel_names: List[str]
    sampling_rate: float
    channel_descriptions: List[str]   # base per-channel text (no Hz/window suffix)
    channel_mask: Optional[List[bool]] = None   # True = REAL channel (False = zero-padded absent);
                                                # lets text dropout skip padded channels (F10b)
    role_descriptions: Optional[List[str]] = None
    sensor_descriptions: Optional[List[str]] = None
    sensor_id: Optional[List[int]] = None
    gravity_state: Optional[str] = None
    applied_augmentations: List[str] = field(default_factory=list)

def _gravity_present(triad: "np.ndarray", descs=None) -> bool:
    """True if an accelerometer triad still contains the gravity DC component.

    Prefers the DOCUMENTED gravity state (channel-description text is authoritative);
    only falls back to a hardened signal heuristic when the text is silent. The old
    pure DC/RMS ratio misfired on normalized data (e.g. recgym acc centered ~0.5, where
    dc/rms~1) and on low-motion gravity-removed data (e.g. kuhar static postures), so it
    now also requires the DC vector to be axis-concentrated (real gravity points ~down =
    one dominant axis; a uniform per-axis offset spreads across axes and is NOT gravity).
    """
    if descs:
        j = " ".join(str(d).lower() for d in descs)
        if "recgym" in j or "min-max normalized" in j or "dimensionless" in j:
            return False
        if any(k in j for k in ("gravity removed", "gravity-removed", "user acceleration",
                                "useracceleration", "linear acceleration")):
            return False
        if any(k in j for k in ("includes gravity", "including gravity", "with gravity",
                                "gravity included")):
            return True
    a = triad if isinstance(triad, np.ndarray) else triad.detach().cpu().numpy()
    a = a.astype(np.float64)
    m = a.mean(axis=0)
    dc = float(np.linalg.norm(m))
    rms = float(np.sqrt((a ** 2).sum(axis=1).mean())) + 1e-8
    axis_conc = float(np.max(np.abs(m))) / (dc + 1e-8)   # ->1 if one axis dominates
    return dc > 0.5 * rms and axis_conc > 0.6


def _mark_gravity_removed(desc: str) -> str:
    """Rewrite a channel description to state gravity was removed, stripping any
    contradictory 'includes gravity' clause first (avoids 'includes gravity
    (gravity removed)')."""
    import re
    d = re.sub(r"\([^)]*includes gravity[^)]*\)", "", desc, flags=re.I)
    # strip a "; includes gravity" / "includes gravity" clause with any leading separator, so the
    # sibling channel_description() format ("...; includes gravity") does not leave a dangling ";".
    d = re.sub(r"\s*;?\s*\bincludes gravity\b", "", d, flags=re.I)
    d = re.sub(r"\s{2,}", " ", d).strip().rstrip(",;").strip()
    if "gravity removed" not in d.lower():
        d = f"{d} (gravity removed)"
    return d


def _mark_partner_presence(desc: str, *, modality: str, partner_present: bool) -> str:
    """Keep factored sensor text truthful after modality-level channel dropout."""
    import re

    if modality == "accel":
        clause = (
            "recorded alongside a gyroscope" if partner_present
            else "recorded without a gyroscope"
        )
        partner = r"(?:a\s+)?gyroscope"
    elif modality == "gyro":
        clause = (
            "recorded alongside an accelerometer" if partner_present
            else "recorded without an accelerometer"
        )
        partner = r"(?:an?\s+)?accelerometer"
    else:
        return desc
    pattern = rf"recorded\s+(?:alongside|without)\s+{partner}"
    if re.search(pattern, desc, flags=re.I):
        return re.sub(pattern, clause, desc, flags=re.I)
    return f"{desc.rstrip().rstrip(';')}; {clause}"


@contextmanager
def _shared_draw(seed: int):
    """Seed BOTH module RNGs for the duration, then restore them exactly.

    The config transforms draw their parameters from the module-level ``random`` and ``np.random``
    generators, so making a transform reproducible across views means seeding those, not passing an
    rng object. State is saved and restored so the surrounding nuisance draws stay independent — a
    view whose nuisance stream got reseeded would silently become a duplicate of its partner.
    """
    py_state = _random.getstate()
    np_state = np.random.get_state()
    _random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    try:
        yield
    finally:
        _random.setstate(py_state)
        np.random.set_state(np_state)


def _random_so3() -> "torch.Tensor":
    """Uniform-random rotation matrix (3x3, float32) from SO(3) (Haar measure).

    Sampled via a random unit quaternion (Marsaglia): four i.i.d. N(0,1), normalized,
    mapped to a rotation matrix. Uniform over orientations and always a proper rotation
    (det=+1, no reflection)."""
    q = np.random.randn(4)
    q = q / (np.linalg.norm(q) + 1e-12)
    w, x, y, z = q
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)
    return torch.from_numpy(R)


class IMUAugmenter:
    """Applies the enabled augmentations (in AugmentationConfig.ORDER) to an
    IMUSample. Operates per sample on (T, C) tensors — padding is added later in
    collate, so no attention mask is needed here."""

    def __init__(self, config: "AugmentationConfig"):
        self.cfg = config

    def __call__(self, sample: "IMUSample", *, shared_config_seed: Optional[int] = None
                 ) -> "IMUSample":
        """Apply the configured augmentations in ORDER.

        ``shared_config_seed`` makes every CONFIG_GROUP augmentation deterministic for a given
        window, so all views of that window (VICReg A, VICReg B, and the JEPA teacher) see the SAME
        acquisition configuration. NUISANCE_GROUP augmentations still draw independently — that is
        what makes the pair a positive.

        WHY THE PARAMETERS, NOT JUST THE DECISION. An earlier revision shared only the
        sensor-text-dropout *decision*, with the right argument: if one view describes the config and
        the other neutralises it, the invariance term trains ``embed(config) == embed(no config)``,
        i.e. pressure to IGNORE acquisition config — the opposite of the thesis. But the physical
        config transforms draw their parameters INTERNALLY (``_random_so3`` from ``np.random.randn``,
        rate from ``np.random.uniform``, dropout from ``_random.sample``), so sharing only the
        decision leaves both views rotated by DIFFERENT rotations to different rates. VICReg then
        demands invariance to the difference, which is the same failure one level down.

        Measured 2026-08-11: under the undifferentiated stack, config conditioning was worth
        +0.0086 kNN-BA against a 0.0065 noise floor, sign-flipping on half the cohort — nothing.
        Seeding both module RNGs around the transform makes the drawn rotation, target rate, gravity
        decision and dropped channels identical across views, so the config becomes a property of
        the sample rather than a difference to be erased.
        """
        for index, name in enumerate(AugmentationConfig.ORDER):
            spec = getattr(self.cfg, name)
            if not spec.enabled:
                continue
            shared = (shared_config_seed is not None
                      and name in AugmentationConfig.CONFIG_GROUP
                      and getattr(spec, "shared_across_views", True))
            if shared:
                # Derive a per-(window, transform) seed deterministically. Not hash(): PYTHONHASHSEED
                # randomises it per process, so two dataloader workers would disagree about the same
                # window's configuration.
                with _shared_draw(shared_config_seed * 1_000_003 + index):
                    if _random.random() >= spec.p:
                        continue
                    sample = (self._sensor_text_dropout(sample, spec, _random)
                              if name == "sensor_text_dropout"
                              else getattr(self, "_" + name)(sample, spec))
                continue
            if _random.random() >= spec.p:
                continue
            sample = (self._sensor_text_dropout(sample, spec, _random)
                      if name == "sensor_text_dropout"
                      else getattr(self, "_" + name)(sample, spec))
        return sample

    # ---------- triad helper ----------
    @staticmethod
    def _triads(channel_names):
        """Return {location: [(indices3, group_name), ...]} for x/y/z triads only."""
        from data.scripts.curate.channels import (
            group_channels_by_sensor,
        )
        groups = group_channels_by_sensor(channel_names)
        ch_to_idx = {n: i for i, n in enumerate(channel_names)}
        def location(g):
            # Location = group name with its sensor-type token removed, robust to alternate
            # spellings ('watch_accel') and numeric range suffixes ('chest_acc16'). This keeps a
            # location's accel + gyro (+ mag) in ONE bucket so _rotation_3d rotates them by a
            # SHARED R (one rigid-body frame); mis-locating rotates accel apart from its gyro (#88).
            m = _SENSOR_TOKEN_RE.search(g)
            if not m:
                return g
            return (g[:m.start()] + g[m.end():]).strip("_")

        out = {}
        for g, chans in groups.items():
            if len(chans) != 3:
                continue
            out.setdefault(location(g), []).append(([ch_to_idx[c] for c in chans], g))
        return out

    # ---------- value-space (ported) ----------
    def _jitter(self, s, spec):
        scale = s.data.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        s.data = s.data + torch.randn_like(s.data) * (spec.sigma * scale)
        s.applied_augmentations.append("jitter")
        return s

    def _scale(self, s, spec):
        C = s.data.shape[1]
        factors = torch.empty(1, C, device=s.data.device).uniform_(spec.low, spec.high)
        s.data = s.data * factors
        s.applied_augmentations.append("scale")
        return s

    # ---------- P1: gravity add/remove ----------
    def _gravity(self, s, spec):
        sr = float(s.sampling_rate)
        wn = spec.cutoff_hz / (sr / 2.0)
        if not (0.0 < wn < 1.0):     # cutoff above Nyquist (very low rate) -> skip
            return s
        T = s.data.shape[0]
        if T <= 3 * (spec.order + 1):   # filtfilt needs enough samples
            return s
        b, a = _sps.butter(spec.order, wn, btype="low")
        x = s.data.detach().cpu().numpy().astype(np.float64)
        desc = list(s.channel_descriptions)
        changed = False
        affected_sensor_ids = set()
        for _loc, triads in self._triads(s.channel_names).items():
            for idxs, gname in triads:
                if "acc" not in gname:       # only accelerometer carries gravity
                    continue
                if s.channel_mask is not None and not all(s.channel_mask[j] for j in idxs):
                    continue                 # canonical zero-padding is not a physical accelerometer
                if not _gravity_present(x[:, idxs], [desc[j] for j in idxs]):  # already gravity-removed -> skip
                    continue
                for j in idxs:
                    grav = _sps.filtfilt(b, a, x[:, j])
                    x[:, j] = x[:, j] - grav
                    desc[j] = _mark_gravity_removed(desc[j])
                    if s.sensor_id is not None:
                        affected_sensor_ids.add(s.sensor_id[j])
                changed = True
        if changed:
            s.data = torch.from_numpy(x).float().to(s.data.device)
            s.channel_descriptions = desc
            s.gravity_state = "removed"
            if s.sensor_descriptions is not None:
                affected = (affected_sensor_ids if s.sensor_id is not None
                            else set(range(len(s.sensor_descriptions))))
                sensor_desc = list(s.sensor_descriptions)
                for sid in affected:
                    sensor_desc[sid] = _mark_gravity_removed(sensor_desc[sid])
                s.sensor_descriptions = sensor_desc
            s.applied_augmentations.append("gravity")
        return s

    # ---------- P2: full uniform-random SO(3) rotation ----------
    def _rotation_3d(self, s, spec):
        """Rotate every co-located sensor triad by one shared uniform-random SO(3) R
        per body location (acc/gyro/mag at the same place rotate together, preserving
        their physical relationship). The gravity DC rotates with the accel signal, so
        this teaches gravity-direction invariance rather than scrambling an unseen cue."""
        triloc = self._triads(s.channel_names)
        if not triloc:
            return s
        x = s.data

        for _loc, triads in triloc.items():
            R = _random_so3().to(x.dtype).to(x.device)
            for idxs, _gname in triads:
                x[:, idxs] = torch.einsum("ij,tj->ti", R, x[:, idxs])
        s.data = x
        s.applied_augmentations.append("rotation_3d")
        return s

    # ---------- P3: anti-aliased rate resample ----------
    def _rate(self, s, spec):
        old = float(s.sampling_rate)
        downsample_max = min(float(spec.max_hz), old)
        if downsample_max > float(spec.min_hz) and np.random.random() < 0.5:
            new = float(np.random.uniform(spec.min_hz, downsample_max))
        else:
            new = float(np.random.uniform(spec.min_hz, spec.max_hz))
        if old <= 0 or abs(new - old) < 1e-3:
            return s
        frac = Fraction(new / old).limit_denominator(50)
        up, down = frac.numerator, frac.denominator
        if up < 1 or down < 1:
            return s
        T = s.data.shape[0]
        if int(round(T * up / down)) < spec.min_samples:
            return s
        x = s.data.detach().cpu().numpy()
        y = _sps.resample_poly(x, up, down, axis=0, padtype="line")
        s.data = torch.from_numpy(np.ascontiguousarray(y)).float().to(s.data.device)
        s.sampling_rate = old * up / down               # actual achieved rate
        s.applied_augmentations.append("rate")
        return s

    # ---------- P5: random temporal crop (variable observation length) ----------
    def _window_crop(self, s, spec):
        T = s.data.shape[0]
        floor = min(T, spec.min_samples)
        lo = max(floor, int(round(spec.min_frac * T)))
        if lo >= T:                                        # nothing to crop (already at/below floor)
            return s
        length = int(np.random.randint(lo, T + 1))         # keep [lo, T] contiguous samples
        start = int(np.random.randint(0, T - length + 1))
        s.data = s.data[start:start + length].contiguous()
        s.applied_augmentations.append("window_crop")
        return s

    # ---------- P4: channel / sensor-group dropout ----------
    def _channel_dropout(self, s, spec):
        from data.scripts.curate.channels import (
            group_channels_by_sensor,
        )
        names = s.channel_names
        drop = {i for i, n in enumerate(names) if any(g in n for g in spec.groups)}
        keep = [i for i in range(len(names)) if i not in drop]
        if not drop or len(keep) < 3:
            return s
        kept_names = [names[i] for i in keep]
        # require at least one full x/y/z triad to survive the drop
        if not any(len(v) == 3 for v in group_channels_by_sensor(kept_names).values()):
            return s
        s.data = s.data[:, keep]
        s.channel_names = kept_names
        s.channel_descriptions = [s.channel_descriptions[i] for i in keep]
        if s.role_descriptions is not None:
            s.role_descriptions = [s.role_descriptions[i] for i in keep]
        if s.channel_mask is not None:
            s.channel_mask = [s.channel_mask[i] for i in keep]
        # Factored sensors: accel and gyro are separate modality-level SENSORS, so dropping a channel
        # group REMOVES that modality's sensor entirely (rather than rewriting a modality phrase in a
        # single shared description). Keep only sensors that still own a surviving channel and compact
        # sensor_id to a dense [0..n) index into the pruned sensor_descriptions.
        if s.sensor_id is not None:
            kept_ids = [s.sensor_id[i] for i in keep]
            if s.sensor_descriptions is not None:
                used = sorted(set(kept_ids))
                remap = {old: new for new, old in enumerate(used)}
                s.sensor_descriptions = [s.sensor_descriptions[sid] for sid in used]
                s.sensor_id = [remap[sid] for sid in kept_ids]
                # The descriptor records whether the encoder saw the partner modality. Once dropout
                # removes a modality, leaving "recorded alongside ..." would condition the signal on
                # a sensor that is no longer present. Recompute the clause from surviving channels.
                has_accel = any(name.startswith("acc") for name in kept_names)
                has_gyro = any(name.startswith("gyro") for name in kept_names)
                rewritten = []
                for sid, description in enumerate(s.sensor_descriptions):
                    owned = [name for name, owner in zip(kept_names, s.sensor_id) if owner == sid]
                    modality = "accel" if any(name.startswith("acc") for name in owned) else "gyro"
                    rewritten.append(_mark_partner_presence(
                        description,
                        modality=modality,
                        partner_present=has_gyro if modality == "accel" else has_accel,
                    ))
                s.sensor_descriptions = rewritten
            else:
                s.sensor_id = kept_ids
        s.applied_augmentations.append("channel_dropout")
        return s

    # ---------- text: channel-description phrase paraphrase ----------
    def _channel_text_phrase(self, s, spec):
        # Paraphrase each channel description independently (surface form only; placement /
        # units / gravity are preserved by construction — see _paraphrase_channel).
        s.channel_descriptions = [_paraphrase_channel(d) for d in s.channel_descriptions]
        # Role text is AXIS-ONLY ("x"/"y"/"z") since the factored rework — it is NOT paraphrased.
        # Channel templates turned bare axes into "signal from x" / "this channel measures z", which
        # is decoration with no semantic content and just adds SBERT noise to a 3-way distinction.
        if s.sensor_descriptions is not None:
            s.sensor_descriptions = [_paraphrase_sensor(d) for d in s.sensor_descriptions]
        s.applied_augmentations.append("channel_text_phrase")
        return s

    # ---------- text: channel-description dropout (neutralize, keep signal) ----------
    def _channel_text_dropout(self, s, spec):
        # Neutralize only REAL channels and never all of them (F10b): padded/absent channels are
        # masked out by the encoder, so neutralizing their text is a no-op — and counting them in
        # the budget let `max_frac` consume every real placement description on an acc-only stream
        # (3 real of 6 slots). Bound over the REAL-channel count and always keep >=1 real described.
        cm = s.channel_mask
        real = [i for i in range(len(s.channel_descriptions))
                if cm is None or (i < len(cm) and cm[i])]
        if len(real) > 1:
            max_drop = max(1, int(spec.max_frac * len(real)))
            k = min(_random.randint(1, max_drop), len(real) - 1)   # keep >=1 real channel described
            dropped = _random.sample(real, k)
            desc = list(s.channel_descriptions)
            for i in dropped:
                desc[i] = spec.neutral
            s.channel_descriptions = desc

            # The same selected CHANNEL roles are neutralized in the factored path. Sensor identity
            # is shared by every channel on that sensor, so dropping a one-sensor description would
            # erase 100% of placement/gravity metadata and violate max_frac.
            if s.role_descriptions is not None:
                role_desc = list(s.role_descriptions)
                for i in dropped:
                    role_desc[i] = getattr(spec, "role_neutral", spec.neutral)
                s.role_descriptions = role_desc
            s.applied_augmentations.append("channel_text_dropout")
        return s

    # ---------- text: per-SENSOR identity dropout (device/placement/gravity) ----------
    def _sensor_text_dropout(self, s, spec, rng=_random):
        # F7: the factored model always saw the full sensor identity, so it never learned to operate
        # when placement/device metadata is missing. Neutralize a bounded subset of the per-sensor
        # descriptions: with >=2 sensors keep >=1 described; a single-sensor stream may be fully
        # neutralized (the fully-unconditioned fallback), but that is rare (gated by the low spec.p).
        if s.sensor_descriptions is None:
            return s
        n = len(s.sensor_descriptions)
        if n == 0:
            return s
        if n == 1:
            s.sensor_descriptions = [spec.neutral]
            s.applied_augmentations.append("sensor_text_dropout")
            return s
        max_drop = max(1, int(spec.max_frac * n))
        k = min(rng.randint(1, max_drop), n - 1)          # keep >=1 sensor described
        dropped = rng.sample(range(n), k)                 # rng may be the shared per-window one
        desc = list(s.sensor_descriptions)
        for i in dropped:
            desc[i] = spec.neutral
        s.sensor_descriptions = desc
        s.applied_augmentations.append("sensor_text_dropout")
        return s
