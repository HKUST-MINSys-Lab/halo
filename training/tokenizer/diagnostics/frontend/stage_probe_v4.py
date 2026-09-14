"""Where does within-window time-invariance arise, and does the predictor beat persistence?
Stages: frontend tokens -> random transformer -> trained student -> teacher target, for the
fixed-multires and continuous-multispan arms. Also predictor vs 'copy the last context token'."""
import sys, json, dataclasses, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/home/alex/code/HALO/halo")
from data.scripts.eda.grid_io import discover_grids
from training.tokenizer.eval_transfer import build_encoder
from training.tokenizer.pretrain import PipelineAModel, PretrainConfig
from training.tokenizer.pretrain_data import (MultiResolutionCollate, MultiScaleCollate, modalities_present,
    stream_sensor_bias, stream_sensor_texts, STREAM_SOURCE_RATE_HZ, _stream_gravity_state)
from training.tokenizer.future_jepa import (FuturePredictor, make_future_target_plan, normalized_teacher_target, gather_token_rows)
dev = torch.device("cuda"); torch.manual_seed(0)
ARM = sys.argv[1]
CK = {"fixed": "training/tokenizer/outputs/jepa_fixed_multires_20260912_full/last.pt",
      "multispan": "training/tokenizer/outputs/jepa_multispan_20260912_full/last.pt"}[ARM]
blob = torch.load(CK, map_location="cpu", weights_only=False); cfg = blob["config"]
student = build_encoder(blob, dev).eval()
tb = dict(blob); tb["encoder"] = blob["jepa_teacher"]; teacher = build_encoder(tb, dev).eval()
# random-init encoder of the SAME architecture with the trained calibration buffers copied in
names = {f.name for f in dataclasses.fields(PretrainConfig)}
rcfg = PretrainConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg.items() if k in names})
random_enc = PipelineAModel(rcfg).encoder.to(dev).eval()
with torch.no_grad():
    for n, b in random_enc.named_buffers():
        if n in blob["encoder"] and blob["encoder"][n].shape == b.shape: b.copy_(blob["encoder"][n].to(b.dtype))
pred = FuturePredictor(cfg["d_model"], predictor_dim=cfg["future_predictor_dim"], num_layers=cfg["future_predictor_layers"],
                       num_heads=cfg["future_predictor_heads"], dropout=cfg["dropout"], max_resolutions=8).to(dev).eval()
pred.load_state_dict(blob["heads"]["future_predictor"])
durations = tuple(cfg["future_patch_durations"])
collate = (MultiResolutionCollate(fixed_patch_seconds=durations, min_resolution_ratio=cfg["min_resolution_ratio"], dft_size=cfg["dft_size"])
           if ARM == "fixed" else MultiScaleCollate(fixed_patch_seconds=cfg["patch_seconds"], dft_size=cfg["dft_size"]))
refs = {(r.dataset, r.stream): r for r in discover_grids("native")}
STREAMS = [("capture24_pretrain","watch_wrist"),("extrasensory_pretrain","watch_wrist"),("nymeria_xsens","xsens_pelvis")]
N = 128

def frontend_and_meta(enc, batch):
    """Sensor tokens straight out of the frontend plus the token grid metadata used by planner/encoder."""
    patches = batch["patches"].to(dev); sd, stid = enc.encode_sensor_descriptors_unique(batch["sensor_texts"], dev)
    common = dict(channel_mask=batch["channel_mask"].to(dev), source_rate_hz=batch["source_rates"].to(dev),
                  sensor_id=batch["sensor_id"].to(dev), n_sensors=stid.shape[1])
    if ARM == "fixed":
        toks = enc.tokenize(patches, batch["rates"].to(dev), batch["patch_len"].to(dev), **common)
        meta = dict(positions=batch["positions"], durations=batch["patch_durations"], resolution_ids=batch["resolution_ids"],
                    token_mask=batch["patch_padding_mask"], starts=batch["patch_starts"], ends=batch["patch_ends"])
    else:
        full = (batch["patch_len"] * batch["patch_padding_mask"]).sum(1).float() / batch["rates"].clamp_min(1e-6)
        g = enc.filterbank.token_grid(patches, batch["rates"].to(dev), batch["patch_len"].to(dev), patch_mask=batch["patch_padding_mask"].to(dev),
                                      grid_duration_seconds=full.to(dev), **common)
        toks = g["tokens"]; meta = dict(positions=g["positions"].cpu(), durations=g["durations"].cpu(), resolution_ids=g["resolution_ids"].cpu(), token_mask=g["token_mask"].cpu())
        meta["starts"] = meta["positions"] - 0.5 * meta["durations"]; meta["ends"] = meta["positions"] + 0.5 * meta["durations"]
    return toks, meta, sd, stid

def encode(enc, toks, meta, sd, stid, batch, context_mask=None, layer_states=False):
    pad = meta["token_mask"].to(dev); B, P = pad.shape
    if context_mask is not None:
        toks = toks * context_mask.view(B, P, 1, 1).to(toks.dtype); pad = context_mask
    return enc._encode_sensor(toks, None, None, meta["positions"].to(dev), patch_durations=meta["durations"].to(dev),
        resolution_ids=meta["resolution_ids"].to(dev), channel_mask=batch["channel_mask"].to(dev), patch_padding_mask=pad,
        sensor_descriptors=sd, sensor_id=batch["sensor_id"].to(dev), sensor_text_ids=stid, sensor_bias=batch["sensor_bias"].to(dev),
        return_retrieval_tokens=False, return_layer_states=layer_states)

def lnorm(t):  # per-token layernorm + L2, like the target
    t = t.float(); return F.normalize(F.layer_norm(t, (t.shape[-1],)), dim=-1)

def within_across(tokens, valid, res, bucket, channel_mask=None):
    """tokens (B,P,S,D) valid (B,P,S); same-window same-res cosine (excluding self) vs other-window, by motion tercile."""
    z = lnorm(tokens); B, P, S, D = z.shape
    if valid.shape[2] != S:
        valid = valid.any(2, keepdim=True).expand(-1, -1, S) & channel_mask.to(dev).view(B, 1, S)
    zf = z.flatten(1, 2); vf = valid.flatten(1); rf = res.unsqueeze(2).expand(-1, -1, S).flatten(1)
    sim = torch.einsum("bqd,bkd->bqk", zf, zf); n = zf.shape[1]
    eye = torch.eye(n, dtype=torch.bool, device=dev).unsqueeze(0)
    m = vf.unsqueeze(2) & vf.unsqueeze(1) & ~eye & (rf.unsqueeze(2) == rf.unsqueeze(1))
    cross = torch.einsum("bqd,bkd->bqk", zf, zf.roll(1, 0)); mc = vf.unsqueeze(2) & vf.roll(1, 0).unsqueeze(1)
    out = {"within_same_res": float(sim[m].mean()), "other_window": float(cross[mc].mean())}
    for bi, name in enumerate(("still", "mid", "moving")):
        rb = (bucket == bi).view(B, 1, 1)
        out[f"within_{name}"] = float(sim[m & rb].mean())
    return out

results = {}
for ds, st in STREAMS:
    ref = refs[(ds, st)]; rng = np.random.default_rng(0)
    lengths = np.asarray(ref.load_lengths()); full = np.flatnonzero(lengths == ref.shape[1]); idx = np.sort(rng.choice(full, N, replace=False))
    data = np.asarray(ref.load_data()[idx], dtype=np.float32)
    energy = np.linalg.norm(data[:, :, :3], axis=2).std(1); bucket = torch.as_tensor(np.digitize(energy, np.quantile(energy, [1/3, 2/3])), device=dev)
    cmask = torch.as_tensor(ref.mask, dtype=torch.bool); mods = modalities_present(cmask.tolist()); g = _stream_gravity_state(ds, st)
    role, stexts, sid = stream_sensor_texts(ds, st, gravity_removed=(g == "removed"), has_accel="accel" in mods, has_gyro="gyro" in mods, neutral=False)
    sbias = stream_sensor_bias(ds, st, mods); src_rate = min(STREAM_SOURCE_RATE_HZ.get(f"{ds}/{st}", ref.rate_hz), ref.rate_hz)
    items = [{"data": torch.tensor(w), "rate": ref.rate_hz, "source_rate": src_rate, "texts": [], "label_id": 0, "channel_mask": cmask,
              "gravity_state": g, "source": ds, "role_texts": role, "sensor_texts": stexts, "sensor_id": torch.tensor(sid), "sensor_bias": sbias} for w in data]
    batch = collate(items)
    r = {"motion_terciles_std_g": [float(energy[np.digitize(energy, np.quantile(energy, [1/3, 2/3])) == i].mean()) for i in range(3)]}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        # --- stage similarity
        stages = {}
        for name, enc in (("random_init", random_enc), ("trained_student", student), ("trained_teacher", teacher)):
            toks, meta, sd, stid = frontend_and_meta(enc, batch)
            pad = meta["token_mask"].to(dev); res = meta["resolution_ids"].to(dev)
            out = encode(enc, toks, meta, sd, stid, batch, layer_states=True)
            valid = pad.unsqueeze(2) & out["sensor_present"].unsqueeze(1)
            if name == "random_init":
                stages["A_frontend_tokens(random proj)"] = within_across(toks, valid, res, bucket, batch["channel_mask"])
                stages["B_random_transformer_tokens"] = within_across(out["tokens"], valid, res, bucket)
                stages["B_random_teacher_style_target"] = within_across(normalized_teacher_target(out["layer_states"], top_k=cfg["future_teacher_top_layers"]), valid, res, bucket)
            elif name == "trained_student":
                stages["A_frontend_tokens(trained proj)"] = within_across(toks, valid, res, bucket, batch["channel_mask"])
                stages["C_trained_student_tokens"] = within_across(out["tokens"], valid, res, bucket)
            else:
                tgt_grid = normalized_teacher_target(out["layer_states"], top_k=cfg["future_teacher_top_layers"])
                stages["D_trained_teacher_target"] = within_across(tgt_grid, valid, res, bucket)
                T_meta, T_valid, T_res = meta, valid, res
        r["stages"] = stages
        # --- predictor vs persistence baseline (trained arm)
        toks, meta, sd, stid = frontend_and_meta(student, batch)
        plan = make_future_target_plan(meta["starts"], meta["ends"], meta["token_mask"], meta["resolution_ids"],
                                       context_fraction=tuple(cfg["future_context_fraction"]), horizon_bins_seconds=tuple(map(tuple, cfg["future_horizon_bins_seconds"])))
        ctx, tmask, hor = plan.context_mask.to(dev), plan.target_mask.to(dev), plan.horizon_seconds.to(dev)
        S_out = encode(student, toks, meta, sd, stid, batch, context_mask=ctx)
        cv = ctx.unsqueeze(2) & S_out["sensor_present"].unsqueeze(1)
        p, ind, qv = pred(S_out["tokens"], cv, tmask, meta["positions"].to(dev), meta["durations"].to(dev), meta["resolution_ids"].to(dev), hor, S_out["descriptor"], S_out["sensor_present"])
    tg = lnorm(gather_token_rows(tgt_grid, ind)); pn = lnorm(p); B, Q, _ = pn.shape
    own = (pn * tg).sum(-1)
    # persistence: last context token with same resolution & sensor
    pos = meta["positions"].to(dev); res = meta["resolution_ids"].to(dev)
    pq = ind[..., 0]; sq = ind[..., 1]
    cand = ctx.unsqueeze(1) & (res.unsqueeze(1) == res.gather(1, pq).unsqueeze(2))            # (B,Q,P)
    score = torch.where(cand, pos.unsqueeze(1).expand(-1, Q, -1), torch.full_like(pos.unsqueeze(1).expand(-1, Q, -1), -1e9))
    p_last = score.argmax(2); has = cand.any(2)
    last_idx = torch.stack((p_last, sq), -1); last_tgt = lnorm(gather_token_rows(tgt_grid, last_idx))
    persist = (last_tgt * tg).sum(-1)
    # mean of all same-res same-sensor context targets
    ok = qv & has
    # specificity: prediction vs OTHER targets of the same window at the same resolution
    tg_all = lnorm(gather_token_rows(tgt_grid, ind)); sim = torch.einsum("bqd,bkd->bqk", pn, tg_all)
    rq = res.gather(1, pq); eye = torch.eye(Q, dtype=torch.bool, device=dev).unsqueeze(0)
    v2 = qv.unsqueeze(2) & qv.unsqueeze(1); same_res = rq.unsqueeze(2) == rq.unsqueeze(1)
    m_other = v2 & ~eye & same_res
    other_same_res = sim.masked_fill(~m_other, 0).sum(2) / m_other.sum(2).clamp_min(1); has_other = m_other.any(2)
    top1 = (sim.masked_fill(~(v2 & same_res), -9).argmax(2) == torch.arange(Q, device=dev)).float()
    chance = 1.0 / (v2 & same_res).sum(2).clamp_min(1).float()
    # controls: predictor fed the WRONG window's context (roll within batch, same stream), and NO context
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        p_wrong, _, _ = pred(S_out["tokens"].roll(1, 0), cv.roll(1, 0), tmask, meta["positions"].to(dev), meta["durations"].to(dev), meta["resolution_ids"].to(dev), hor, S_out["descriptor"], S_out["sensor_present"])
        p_none, _, _ = pred(S_out["tokens"], torch.zeros_like(cv), tmask, meta["positions"].to(dev), meta["durations"].to(dev), meta["resolution_ids"].to(dev), hor, S_out["descriptor"], S_out["sensor_present"])
    own_wrong = (lnorm(p_wrong) * tg).sum(-1); own_none = (lnorm(p_none) * tg).sum(-1)
    # more controls: (i) every context token replaced by the per-sensor MEAN context token (window summary, no temporal structure);
    # (ii) context tokens time-SHUFFLED among context positions (tests whether the predictor can even see context order)
    ctx_tokens = S_out["tokens"]; cvf = cv.to(ctx_tokens.dtype).unsqueeze(-1)
    mean_ctx = (ctx_tokens * cvf).sum(1, keepdim=True) / cvf.sum(1, keepdim=True).clamp_min(1)
    tok_mean = torch.where(cv.unsqueeze(-1), mean_ctx.expand_as(ctx_tokens), ctx_tokens)
    tok_shuf = ctx_tokens.clone(); gperm = torch.Generator(device="cpu").manual_seed(0)
    for b in range(B):
        ii = torch.nonzero(ctx[b]).flatten()
        if len(ii) > 1: tok_shuf[b, ii] = ctx_tokens[b, ii[torch.randperm(len(ii), generator=gperm).to(dev)]]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        p_mean, _, _ = pred(tok_mean, cv, tmask, meta["positions"].to(dev), meta["durations"].to(dev), meta["resolution_ids"].to(dev), hor, S_out["descriptor"], S_out["sensor_present"])
        p_shuf, _, _ = pred(tok_shuf, cv, tmask, meta["positions"].to(dev), meta["durations"].to(dev), meta["resolution_ids"].to(dev), hor, S_out["descriptor"], S_out["sensor_present"])
    own_mean = (lnorm(p_mean) * tg).sum(-1); own_shuf = (lnorm(p_shuf) * tg).sum(-1)
    shuf_delta = float((lnorm(p_shuf) - pn).abs().max())
    n_res = {f"tokens_per_window_res{d:g}s": float(((meta["resolution_ids"] == i) & meta["token_mask"]).sum(1).float().mean()) for i, d in enumerate(durations)}
    def top1_of(pp):
        s = torch.einsum("bqd,bkd->bqk", lnorm(pp), tg_all).masked_fill(~(v2 & same_res), -9)
        return float((s.argmax(2) == torch.arange(Q, device=dev)).float()[ok & has_other].mean())
    pr = {**n_res, "top1_same_res | NO context": top1_of(p_none), "top1_same_res | MEAN context": top1_of(p_mean), "top1_same_res | WRONG window": top1_of(p_wrong), "pred_vs_own | MEAN-context (summary only)": float(own_mean[ok].mean()),
          "pred_vs_own | time-SHUFFLED context": float(own_shuf[ok].mean()), "max|pred_shuffled - pred| (0 => predictor is order-blind)": shuf_delta,
          "pred_vs_OTHER_target_same_res_same_window": float(other_same_res[ok & has_other].mean()),
          "top1_among_same_res (chance)": f"{float(top1[ok & has_other].mean()):.3f} ({float(chance[ok & has_other].mean()):.3f})",
          "pred_vs_own_target | WRONG-window context": float(own_wrong[ok].mean()),
          "pred_vs_own_target | NO context (metadata only)": float(own_none[ok].mean()),"pred_vs_own_target": float(own[ok].mean()), "persistence(last ctx token, same res)_vs_target": float(persist[ok].mean()),
          "pred_minus_persistence": float((own - persist)[ok].mean()), "frac_queries_pred_beats_persistence": float(((own > persist)[ok]).float().mean())}
    for bi, name in enumerate(("still", "mid", "moving")):
        mb = ok & (bucket == bi).view(B, 1)
        pr[f"{name}: pred / mean-ctx / wrong-ctx / no-ctx / persistence / other-same-res"] = f"{float(own[mb].mean()):.3f} / {float(own_mean[mb].mean()):.3f} / {float(own_wrong[mb].mean()):.3f} / {float(own_none[mb].mean()):.3f} / {float(persist[mb].mean()):.3f} / {float(other_same_res[mb & has_other].mean()):.3f}"
    hq = hor.gather(1, pq)
    for lo, hi in map(tuple, cfg["future_horizon_bins_seconds"]):
        mb = ok & (hq >= lo) & (hq < hi); pr[f"h{lo:g}-{hi:g}s: pred / persistence"] = f"{float(own[mb].mean()):.3f} / {float(persist[mb].mean()):.3f}"
    r["predictor"] = pr
    results[f"{ds}/{st}"] = r
    print(f"\n## [{ARM}] {ds}/{st}  motion terciles std|acc| g = {np.round(r['motion_terciles_std_g'],3)}")
    print(f"   {'stage':38s} {'within-window same-res':>22s} {'other window':>13s} {'still':>7s} {'mid':>7s} {'moving':>7s}")
    for k, v in stages.items(): print(f"   {k:38s} {v['within_same_res']:22.3f} {v['other_window']:13.3f} {v['within_still']:7.3f} {v['within_mid']:7.3f} {v['within_moving']:7.3f}")
    for k, v in pr.items(): print(f"   {k:50s} {v if isinstance(v, str) else round(v, 4)}")
json.dump(results, open(f"training/tokenizer/diagnostics/frontend/out/stage_probe_{ARM}.json", "w"), indent=1)
