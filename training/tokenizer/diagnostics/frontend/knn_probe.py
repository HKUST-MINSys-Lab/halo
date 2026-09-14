"""Subject-disjoint 5-NN balanced accuracy on TRAINING sources for several feature extractors.
Answers: how much of the random-init encoder's separability is (a) trivial data structure,
(b) the physical filterbank, (c) the random transformer, (d) JEPA, (e) supervised adaptation."""
import sys, json, time, numpy as np, torch
sys.path.insert(0, "/home/alex/code/HALO/halo")
from data.scripts.eda.grid_io import discover_grids
from training.tokenizer.eval_transfer import build_encoder, encode_dataset, knn_balanced_acc
from training.tokenizer.pretrain_data import stream_channel_descriptions, _stream_gravity_state
from training.tokenizer.eval_quality import subj_split

dev = torch.device("cuda")
STREAMS = [("hhar","phone_waist"),("kuhar","phone_waist"),("realdisp","rla"),("dsads","right_wrist"),("wisdm","watch_wrist")]
CK = {
 "random_arm(d128,initial.pt)": "training/support_classifier/outputs/jepa_representation_20260912/random_fixed_seed20260901/initial.pt",
 "jepa_fixed_frozen(last.pt)": "training/tokenizer/outputs/jepa_fixed_multires_20260912_full/last.pt",
 "jepa_fixed+5k_neighbors": "training/support_classifier/outputs/jepa_representation_20260912/fixed_jepa_neighbors_5k/last.pt",
}
MAXN = 6000
refs = {(r.dataset, r.stream): r for r in discover_grids("native")}

def trivial_feats(x, mask, rate):
    x = x[:, :, mask]; N, T, C = x.shape
    mu = x.mean(1); sd = x.std(1)
    acc = np.linalg.norm(x[:, :, :3], axis=2) if mask[:3].all() else np.zeros((N, T))
    f = [mu, sd, acc.mean(1, keepdims=True), acc.std(1, keepdims=True)]
    spec = np.abs(np.fft.rfft(x - mu[:, None, :], axis=1)) ** 2
    freqs = np.fft.rfftfreq(T, 1.0 / rate)
    edges = [0, 0.5, 1.5, 3, 5, 8, rate / 2 + 1e-6]
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = (freqs >= lo) & (freqs < hi)
        f.append(np.log1p(spec[:, b, :].sum(1)))
    return torch.tensor(np.concatenate(f, 1), dtype=torch.float32)

def load_enc(path, reinit=False):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    enc = build_encoder(blob, dev).eval()
    if reinit:
        torch.manual_seed(1)
        for m in enc.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
    return enc

encs = {k: load_enc(p) for k, p in CK.items()}
encs["jepa_arch_RANDOM_trunk(d256,reset)"] = load_enc(CK["jepa_fixed_frozen(last.pt)"], reinit=True)

def run(enc, ref, x, lengths):
    return encode_dataset(enc, x, stream_channel_descriptions(ref.dataset, ref.stream), dev, ref.rate_hz,
                          _stream_gravity_state(ref.dataset, ref.stream), channel_mask=torch.as_tensor(ref.mask, dtype=torch.bool),
                          dataset=ref.dataset, stream=ref.stream, lengths=lengths, amp_dtype=torch.bfloat16).float().cpu()

results = {}
for ds, st in STREAMS:
    ref = refs[(ds, st)]
    lengths = np.asarray(ref.load_lengths()); full = lengths == ref.shape[1]
    idx = np.flatnonzero(full)
    rng = np.random.default_rng(0)
    if len(idx) > MAXN: idx = np.sort(rng.choice(idx, MAXN, replace=False))
    x = np.asarray(ref.load_data()[idx], dtype=np.float32)
    y = np.asarray(ref.labels)[idx].tolist(); subj = np.asarray(ref.subjects)[idx]
    mask = np.asarray(ref.mask, dtype=bool)
    tr, te = subj_split(subj, np.random.default_rng(3))
    ytr = [y[i] for i in tr]; yte = [y[i] for i in te]
    print(f"\n### {ds}/{st}: n={len(idx)} full-windows, labels={len(set(y))}, subjects={len(set(subj))}, train/test subj-split {len(tr)}/{len(te)}", flush=True)
    chance = 1.0 / len(set(y))
    feats = {}
    feats["trivial_stats+bands(np)"] = trivial_feats(x, mask, ref.rate_hz)
    xt = torch.tensor(x)
    for name, enc in encs.items():
        t = time.time(); feats[name] = run(enc, ref, xt, torch.as_tensor(lengths[idx])); 
    row = {}
    for name, z in feats.items():
        z = torch.nan_to_num(z)
        ba = knn_balanced_acc(z[tr], ytr, z[te], yte)
        # within-subject leave-one-out style (random split ignoring subjects) for inflation reference
        perm = np.random.default_rng(5).permutation(len(y)); h = len(y) // 2
        ba_rand = knn_balanced_acc(z[perm[:h]], [y[i] for i in perm[:h]], z[perm[h:]], [y[i] for i in perm[h:]])
        row[name] = (ba, ba_rand)
        print(f"   {name:36s} dim={z.shape[1]:4d}  subj-disjoint 5NN BA={ba:.3f}   random-split BA={ba_rand:.3f}   (chance {chance:.3f})", flush=True)
    results[f"{ds}/{st}"] = row
json.dump(results, open("training/tokenizer/diagnostics/frontend/out/knn_probe.json", "w"), indent=1)
