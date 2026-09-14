"""Does the future predictor predict the *specific* future token, or a recording-level vector?
For label-free 8 s windows: run teacher (full window) and student (context only) exactly as in
training, then compare each prediction to (a) its own target, (b) OTHER future targets of the
same window, (c) targets from other windows. Also report how similar teacher targets are to
each other within a window (the ceiling for a horizon-invariant predictor)."""
import sys, json, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/home/alex/code/HALO/halo")
from data.scripts.eda.grid_io import discover_grids
from training.tokenizer.eval_transfer import build_encoder
from training.tokenizer.pretrain_data import (MultiResolutionCollate, modalities_present, stream_sensor_bias,
    stream_sensor_texts, STREAM_SOURCE_RATE_HZ, _stream_gravity_state)
from training.tokenizer.future_jepa import (FuturePredictor, make_future_target_plan, normalized_teacher_target,
    gather_token_rows)
dev = torch.device("cuda"); torch.manual_seed(0)
CK = sys.argv[1] if len(sys.argv) > 1 else "training/tokenizer/outputs/jepa_fixed_multires_20260912_full/last.pt"
blob = torch.load(CK, map_location="cpu", weights_only=False); cfg = blob["config"]
student = build_encoder(blob, dev).eval()
tb = dict(blob); tb["encoder"] = blob["jepa_teacher"]; teacher = build_encoder(tb, dev).eval()
pred = FuturePredictor(cfg["d_model"], predictor_dim=cfg["future_predictor_dim"], num_layers=cfg["future_predictor_layers"],
                       num_heads=cfg["future_predictor_heads"], dropout=cfg["dropout"], max_resolutions=8).to(dev).eval()
pred.load_state_dict(blob["heads"]["future_predictor"])
durations = tuple(cfg["future_patch_durations"])
collate = MultiResolutionCollate(fixed_patch_seconds=durations, min_resolution_ratio=cfg["min_resolution_ratio"], dft_size=cfg["dft_size"])
refs = {(r.dataset, r.stream): r for r in discover_grids("native")}
STREAMS = [("capture24_pretrain","watch_wrist"),("extrasensory_pretrain","watch_wrist"),("nymeria_xsens","xsens_pelvis"),("nymeria_xsens","xsens_rforearm")]
N = 96

def encode(enc, batch, context_mask=None, layer_states=False):
    patches = batch["patches"].to(dev); B, P = patches.shape[:2]
    sd, stid = enc.encode_sensor_descriptors_unique(batch["sensor_texts"], dev)
    toks = enc.tokenize(patches, batch["rates"].to(dev), batch["patch_len"].to(dev), channel_mask=batch["channel_mask"].to(dev),
                        source_rate_hz=batch["source_rates"].to(dev), sensor_id=batch["sensor_id"].to(dev), n_sensors=stid.shape[1])
    pad = batch["patch_padding_mask"].to(dev)
    if context_mask is not None:
        toks = toks * context_mask.view(B, P, 1, 1).to(toks.dtype); pad = context_mask
    return enc._encode_sensor(toks, None, None, batch["positions"].to(dev), patch_durations=batch["patch_durations"].to(dev),
        resolution_ids=batch["resolution_ids"].to(dev), channel_mask=batch["channel_mask"].to(dev), patch_padding_mask=pad,
        sensor_descriptors=sd, sensor_id=batch["sensor_id"].to(dev), sensor_text_ids=stid, sensor_bias=batch["sensor_bias"].to(dev),
        return_retrieval_tokens=False, return_layer_states=layer_states)

out = {}
for ds, st in STREAMS:
    ref = refs[(ds, st)]; rng = np.random.default_rng(0)
    lengths = np.asarray(ref.load_lengths()); full = np.flatnonzero(lengths == ref.shape[1])
    idx = np.sort(rng.choice(full, N, replace=False))
    data = np.asarray(ref.load_data()[idx], dtype=np.float32)
    cmask = torch.as_tensor(ref.mask, dtype=torch.bool); mods = modalities_present(cmask.tolist())
    g = _stream_gravity_state(ds, st)
    role, stexts, sid = stream_sensor_texts(ds, st, gravity_removed=(g == "removed"), has_accel="accel" in mods, has_gyro="gyro" in mods, neutral=False)
    sbias = stream_sensor_bias(ds, st, mods); src_rate = min(STREAM_SOURCE_RATE_HZ.get(f"{ds}/{st}", ref.rate_hz), ref.rate_hz)
    items = [{"data": torch.tensor(w), "rate": ref.rate_hz, "source_rate": src_rate, "texts": [], "label_id": 0, "channel_mask": cmask,
              "gravity_state": g, "source": ds, "role_texts": role, "sensor_texts": stexts, "sensor_id": torch.tensor(sid), "sensor_bias": sbias} for w in data]
    batch = collate(items)
    plan = make_future_target_plan(batch["patch_starts"], batch["patch_ends"], batch["patch_padding_mask"], batch["resolution_ids"],
                                   context_fraction=tuple(cfg["future_context_fraction"]), horizon_bins_seconds=tuple(map(tuple, cfg["future_horizon_bins_seconds"])))
    ctx, tgt_mask, hor = plan.context_mask.to(dev), plan.target_mask.to(dev), plan.horizon_seconds.to(dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        T = encode(teacher, batch, layer_states=True)
        S = encode(student, batch, context_mask=ctx)
        target_grid = normalized_teacher_target(T["layer_states"], top_k=cfg["future_teacher_top_layers"])
        cv = ctx.unsqueeze(2) & S["sensor_present"].unsqueeze(1)
        p, ind, qv = pred(S["tokens"], cv, tgt_mask, batch["positions"].to(dev), batch["patch_durations"].to(dev),
                          batch["resolution_ids"].to(dev), hor, S["descriptor"], S["sensor_present"])
    tg = gather_token_rows(target_grid, ind).float()
    pn = F.normalize(F.layer_norm(p.float(), (p.shape[-1],)), dim=-1); tn = F.normalize(F.layer_norm(tg, (tg.shape[-1],)), dim=-1)
    B, Q, D = pn.shape
    sim = torch.einsum("bqd,bkd->bqk", pn, tn)                       # pred q vs targets k, same window
    eye = torch.eye(Q, dtype=torch.bool, device=dev).unsqueeze(0)
    valid2 = qv.unsqueeze(2) & qv.unsqueeze(1)
    own = sim[eye.expand(B, -1, -1) & valid2]
    other_same = sim[(~eye).expand(B, -1, -1) & valid2]
    # cross-window: pred of window b vs targets of window b+1 (roll)
    cross = torch.einsum("bqd,bkd->bqk", pn, tn.roll(1, 0))[valid2 & qv.roll(1, 0).unsqueeze(1)]
    # target-target structure (ceiling): within window vs across
    tt = torch.einsum("bqd,bkd->bqk", tn, tn); tt_in = tt[(~eye).expand(B, -1, -1) & valid2]
    tt_x = torch.einsum("bqd,bkd->bqk", tn, tn.roll(1, 0))[valid2 & qv.roll(1, 0).unsqueeze(1)]
    # is the prediction's nearest target within its own window the correct one?
    sim_m = sim.masked_fill(~valid2, -9); top1 = (sim_m.argmax(2) == torch.arange(Q, device=dev)).float()[qv].mean()
    # and horizon dependence: own-sim by horizon bin
    hq = hor.gather(1, ind[..., 0])
    hb = {f"own_sim_h{lo:g}-{hi:g}s": float(own_val) for (lo, hi), own_val in
          [((lo, hi), sim[eye.expand(B,-1,-1) & valid2 & ((hq >= lo) & (hq < hi)).unsqueeze(2)].mean()) for lo, hi in map(tuple, cfg["future_horizon_bins_seconds"])]}

    rq = batch["resolution_ids"].to(dev).gather(1, ind[..., 0])
    same_res = (rq.unsqueeze(2) == rq.unsqueeze(1))
    pq = batch["positions"].to(dev).gather(1, ind[..., 0])
    near = ((pq.unsqueeze(2) - pq.unsqueeze(1)).abs() < 1.0)
    m_other = (~eye).expand(B, -1, -1) & valid2
    r_extra = {"pred_vs_other_SAME_res_same_window": float(sim[m_other & same_res].mean()),
               "pred_vs_other_DIFF_res_same_window": float(sim[m_other & ~same_res].mean()),
               "pred_vs_other_same_res_within_1s": float(sim[m_other & same_res & near].mean()),
               "pred_vs_other_same_res_beyond_1s": float(sim[m_other & same_res & ~near].mean()),
               "top1_among_SAME_res_targets": float((sim.masked_fill(~(valid2 & same_res), -9).argmax(2) == torch.arange(Q, device=dev)).float()[qv].mean()),
               "chance_among_same_res": float((1.0 / (valid2 & same_res).sum(2).float()[qv]).mean()),
               "target_vs_target_same_res_same_window": float(tt[m_other & same_res].mean()),
               "target_vs_target_diff_res_same_window": float(tt[m_other & ~same_res].mean())}
    for rid, dur in enumerate(durations):
        mm = eye.expand(B,-1,-1) & valid2 & (rq == rid).unsqueeze(2)
        if mm.any(): r_extra[f"own_sim_res{dur:g}s"] = float(sim[mm].mean())
    r = {"n_valid_queries": int(qv.sum()), "pred_vs_OWN_target": float(own.mean()), "pred_vs_OTHER_target_same_window": float(other_same.mean()),
         "pred_vs_target_other_window": float(cross.mean()), "top1_within_window_acc": float(top1), "chance_within_window": float(1.0 / qv.sum(1).float().mean()),
         "target_vs_target_same_window": float(tt_in.mean()), "target_vs_target_other_window": float(tt_x.mean()), **hb, **r_extra}
    out[f"{ds}/{st}"] = r
    print(f"## {ds}/{st}"); [print(f"   {k:36s} {v:.4f}" if isinstance(v, float) else f"   {k:36s} {v}") for k, v in r.items()]
json.dump(out, open(f"training/tokenizer/diagnostics/frontend/out/jepa_spec_{CK.split('/')[-2]}.json", "w"), indent=1)
