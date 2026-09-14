"""Response scale, compression regime, standardisation, Nyquist fingerprint and gain redundancy."""
import sys, math, numpy as np, torch
sys.path.insert(0, "/home/alex/code/HALO/halo")
from data.scripts.eda.grid_io import discover_grids
from training.tokenizer.eval_transfer import build_encoder
dev = torch.device("cuda")
blob = torch.load("training/tokenizer/outputs/jepa_multispan_20260912_full/last.pt", map_location="cpu", weights_only=False)
enc = build_encoder(blob, dev).eval(); fb = enc.filterbank
print(
    f"K={fb.K} M={fb.M} spans={fb.span_list} "
    f"analysis_rate={fb.frame_rate_hz or ('legacy:' + str(fb.frames_per_span))} "
    f"token_rate={fb.token_rate_hz:g} stem={fb.stem_type} "
    f"sigma(frac)={fb._sigmas().min().item():.3f}-{fb._sigmas().max().item():.3f}"
)
print(f"norm_mu per kernel: {np.round(fb.norm_mu.cpu().numpy(),3).tolist()}")
print(f"norm_sd per kernel: {np.round(fb.norm_sd.cpu().numpy(),3).tolist()}")
print(f"amp_mu/sd per group: {fb.amp_mu.cpu().numpy().round(3).tolist()} / {fb.amp_sd.cpu().numpy().round(3).tolist()}   dc_mu/sd: {fb.dc_mu.cpu().numpy().round(3).tolist()} / {fb.dc_sd.cpu().numpy().round(3).tolist()}")
# Nyquist observability per source rate
print("\nlive kernels per source rate (nyq mask, k of 31) and distinct centres <2.5 Hz:")
for r in (20, 25, 50, 100, 240):
    nyq, res = fb.masks(torch.tensor([float(r)]), torch.tensor([8.0]), torch.tensor([float(r)]))
    live = nyq[0] > 0.5
    print(f"   {r:4d} Hz: {int(live.sum())}/31 live   groups: " + ", ".join(f"{fb.span_list[g]}s={int(live[fb.group_slice(g)].sum())}/{fb.group_sizes[g]}" for g in range(fb.G)))
# response regime on real data
refs = {(r.dataset, r.stream): r for r in discover_grids("native")}
for ds, st in (("capture24_pretrain","watch_wrist"),("hhar","phone_waist")):
    ref = refs[(ds, st)]; lengths = np.asarray(ref.load_lengths()); full = np.flatnonzero(lengths == ref.shape[1])
    idx = np.sort(np.random.default_rng(0).choice(full, 64, replace=False))
    x = torch.tensor(np.asarray(ref.load_data()[idx], dtype=np.float32), device=dev)   # (B,T,6)
    T = x.shape[1]; P = 8 if ds.endswith("pretrain") else 6; S = T // P
    patches = x[:, :P*S].reshape(64, P, S, 6); plen = torch.full((64, P), S, device=dev)
    with torch.no_grad():
        an = fb.analyze_grid(patches, float(ref.rate_hz), plen, source_rate_hz=float(ref.rate_hz))
    cm = torch.as_tensor(ref.mask, dtype=torch.bool)
    print(f"\n## {ds}/{st} rate={ref.rate_hz} channels_valid={cm.tolist()}")
    for g, grp in enumerate(an["groups"]):
        c = grp["compressed"][:, cm][..., grp["valid"][0]]           # (B,Cv,k,n) log1p|z|
        z = torch.expm1(c)
        sl = fb.group_slice(g); std = (c - fb.norm_mu[sl].view(1,1,-1,1)) / fb.norm_sd[sl].view(1,1,-1,1)
        q = torch.quantile(z.flatten()[::7], torch.tensor([.5,.9,.99], device=dev)).tolist()
        print(f"   span {fb.span_list[g]}s: |z| median/p90/p99 = {q[0]:.4f}/{q[1]:.4f}/{q[2]:.4f}  frac |z|>0.5 = {(z>0.5).float().mean():.4f}  (log1p linear regime if ~0)"
              f"   standardised feature mean={std.mean():.2f} sd={std.std():.2f}")
    # gain redundancy: how much does doubling one kernel's gain change the standardised feature vs what the linear proj could absorb?
    g0 = 0; sl = fb.group_slice(g0); c = an["groups"][g0]["compressed"][:, cm][..., an["groups"][g0]["valid"][0]]
    z = torch.expm1(c); c2 = torch.log1p(2*z)
    std1 = (c - fb.norm_mu[sl].view(1,1,-1,1))/fb.norm_sd[sl].view(1,1,-1,1); std2 = (c2 - fb.norm_mu[sl].view(1,1,-1,1))/fb.norm_sd[sl].view(1,1,-1,1)
    # fit per-kernel affine std2 ≈ a*std1+b (what the following Linear can absorb); report residual
    a = ((std1*std2).mean((0,1,3)) - std1.mean((0,1,3))*std2.mean((0,1,3))) / (std1.var((0,1,3)) + 1e-9)
    b = std2.mean((0,1,3)) - a*std1.mean((0,1,3)); resid = (std2 - (a.view(1,1,-1,1)*std1 + b.view(1,1,-1,1))).pow(2).mean() / std2.var()
    print(f"   gain x2 on span-{fb.span_list[g0]}s kernels: fraction of feature change NOT absorbable by a per-kernel affine map = {resid.item():.4f}")
