"""Does the multispan continuous-kernel frontend receive usable gradients?
Per parameter group: grad norm, exact-zero fraction, cross-batch sign consistency (SNR) and
pairwise gradient cosine, under a supervised contrastive loss on pooled recording vectors.
Compared for the trained multispan JEPA student and a random re-init of the same architecture."""
import sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/home/alex/code/HALO/halo")
from data.scripts.eda.grid_io import discover_grids
from training.tokenizer.eval_transfer import build_encoder, encode_dataset_detailed
from training.tokenizer.pretrain_data import stream_channel_descriptions, _stream_gravity_state
dev = torch.device("cuda"); torch.manual_seed(0)
CK = "training/tokenizer/outputs/jepa_multispan_20260912_full/last.pt"
blob = torch.load(CK, map_location="cpu", weights_only=False)
refs = {(r.dataset, r.stream): r for r in discover_grids("native")}
STREAMS = [("hhar", "phone_waist"), ("kuhar", "phone_waist"), ("wisdm", "watch_wrist")]
NB, BS = 10, 96

def groups_of(enc):
    fb = enc.filterbank
    g = {"ck/cos_coeff": [fb.cos_coeff], "ck/sin_coeff": [fb.sin_coeff], "ck/sigma_logit": [fb.sigma_logit],
         "ck/gain_logit": [fb.gain_logit], "ck/proj": list(fb.proj.parameters())}
    fb_ids = {id(p) for p in fb.parameters()}
    g["trunk(all other)"] = [p for p in enc.parameters() if id(p) not in fb_ids and p.requires_grad]
    return g

def supcon(z, y, tau=0.1):
    z = F.normalize(z, dim=-1); sim = z @ z.T / tau
    y = torch.as_tensor(y, device=z.device); pos = (y[:, None] == y[None, :]) & ~torch.eye(len(y), dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(torch.eye(len(y), dtype=torch.bool, device=z.device), -1e9)
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    n = pos.sum(1).clamp_min(1); return -(logp * pos).sum(1).div(n)[pos.any(1)].mean()

def probe(enc, label):
    enc.train(False)  # dropout off so batch gradients differ only by data
    for p in enc.parameters(): p.requires_grad_(True)
    G = groups_of(enc); grads = {k: [] for k in G}
    for ds, st in STREAMS:
        ref = refs[(ds, st)]; lengths = np.asarray(ref.load_lengths()); full = np.flatnonzero(lengths == ref.shape[1])
        rng = np.random.default_rng(1); labels = np.asarray(ref.labels)
        for b in range(NB):
            idx = np.sort(rng.choice(full, BS, replace=False))
            x = torch.tensor(np.asarray(ref.load_data()[idx], dtype=np.float32))
            out = encode_dataset_detailed(enc, x, stream_channel_descriptions(ds, st), dev, ref.rate_hz, _stream_gravity_state(ds, st),
                                          channel_mask=torch.as_tensor(ref.mask, dtype=torch.bool), dataset=ds, stream=st,
                                          lengths=torch.as_tensor(lengths[idx]), requires_grad=True, _require_patches=False, batch_size=BS)
            loss = supcon(out["pooled"], [hash(l) % 100003 for l in labels[idx]])
            enc.zero_grad(set_to_none=True); loss.backward()
            for k, ps in G.items():
                grads[k].append(torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten().float().cpu() for p in ps]))
    print(f"\n## {label}   ({len(STREAMS)} streams x {NB} batches x {BS} windows, SupCon on pooled)")
    print(f"   {'group':18s} {'n_params':>9s} {'grad L2 (mean/batch)':>21s} {'zero frac':>9s} {'per-elem SNR |mean|/std':>24s} {'frac SNR>1':>10s} {'pairwise cos':>12s}")
    for k, gs in grads.items():
        Gm = torch.stack(gs); n = Gm.shape[1]
        norm = Gm.norm(dim=1).mean().item(); zero = (Gm == 0).float().mean().item()
        snr = (Gm.mean(0).abs() / (Gm.std(0) + 1e-12)); Gn = F.normalize(Gm, dim=1); cos = (Gn @ Gn.T)
        off = cos[~torch.eye(len(gs), dtype=torch.bool)].mean().item()
        print(f"   {k:18s} {n:9,d} {norm:21.3e} {zero:9.3f} {snr.median().item():24.3f} {(snr > 1).float().mean().item():10.3f} {off:12.3f}")
    return grads

student = build_encoder(blob, dev)
probe(student, "TRAINED multispan JEPA student")
torch.manual_seed(3)
for m in student.modules():
    if hasattr(m, "reset_parameters"): m.reset_parameters()
fb = student.filterbank
with torch.no_grad():  # restore Gabor init for the kernels (reset_parameters does not touch them)
    fb.cos_coeff.zero_(); fb.sin_coeff.zero_(); fb.cos_coeff[torch.arange(fb.K), fb.carrier - 1] = 1.0
    fb.gain_logit.zero_(); fb.sigma_logit.fill_(float(blob["encoder"]["filterbank.sigma_logit"][0]))
probe(student, "RANDOM re-init trunk/proj, Gabor kernels, trained calibration buffers")
