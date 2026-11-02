import csv
import glob
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler, TensorDataset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "nnlayer_repo"))  # git clone of asher-m/nnlayer
from nnlayer import LocallyConnected2d
sys.path.insert(0, os.path.join(HERE, "wetness_persistence_dem"))
from split_groups import split_of  # pixel-disjoint groups, not anchors (anchors share scenes)


"""corridor stack (D:/pinn_haze_lyzmf/stacks/<cell>_corridor_stack.npz):
depth         (T,560,560)   float32  pond depth in metres, NaN off-pond / unsolved
cls           (T,560,560)   uint8    0 nodata 1 ice 2 water 3 pond 4 cloud/other
Spatial VAE (as variational_autoencoder_2.py) with every Conv2d replaced by a stride-1 3x3 nnlayer.LocallyConnected2d:
each layer learns separate weights at every pixel position instead of one shared kernel (encoder downsampling by
average pooling; the decoder's ConvTranspose2d upsampling is unchanged). Channels: depth, known, ice, water, pond,
cloud/nodata. Standalone: data / loss / plotting are copies of the v1 / v2 versions committed in 69675f13."""

STACKS = r"D:\pinn_haze_lyzmf\stacks"


"""
-> Accelerate this hopefully
-> train if fast
-> evaluate
-> look at vision transformer (ViT) for pond depth reconstruction
"""

def done_cells():
    rows = [r for p in glob.glob(os.path.join(STACKS, "_log_shard*of[0-9].csv")) for r in csv.DictReader(open(p))]
    return sorted({r["cell"] for r in rows if r["result"] == "built"})


def paths(split):
    cells = done_cells(); sp = split_of(cells)
    return [os.path.join(STACKS, c + "_corridor_stack.npz") for c in cells if sp[c] == split]


class PondFrames(Dataset):
    def __init__(self, paths, crop=None, seed=0):
        self.items, self.crop, self.rng = [], crop, np.random.default_rng(seed)
        for p in paths:
            with np.load(p) as d:  # never hold a stack open: the rebuild writes into this folder
                cls, dep = d["cls"], d["depth"]
            self.items += [(p, t) for t in range(len(cls)) if np.isfinite(dep[t][cls[t] == 3]).any()]
        self._p, self._d = None, None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, t = self.items[i]
        if p != self._p:
            with np.load(p) as d:
                self._d = (d["depth"], d["cls"])
            self._p = p
        depth, cls = self._d[0][t], self._d[1][t]
        if self.crop:
            ys, xs = np.nonzero((cls == 3) & np.isfinite(depth))
            k = self.rng.integers(len(ys)); h = self.crop // 2
            r0 = int(np.clip(ys[k] - h, 0, cls.shape[0] - self.crop)); c0 = int(np.clip(xs[k] - h, 0, cls.shape[1] - self.crop))
            depth, cls = depth[r0:r0 + self.crop, c0:c0 + self.crop], cls[r0:r0 + self.crop, c0:c0 + self.crop]
        known = (cls == 1) | ((cls == 3) & np.isfinite(depth))  # ice = 0 m
        depth = np.where(known, np.nan_to_num(depth), 0.0)
        x = np.stack([depth, known, cls == 1, cls == 2, cls == 3, (cls == 0) | (cls == 4)])
        return torch.tensor(x, dtype=torch.float32)


class ByFile(Sampler):
    # shuffle files, then frames within a file, so the one-file cache in PondFrames is hit
    def __init__(self, ds, seed=0):
        self.ds, self.rng = ds, np.random.default_rng(seed)
        self.groups = {}
        for i, (p, _) in enumerate(ds.items):
            self.groups.setdefault(p, []).append(i)

    def __iter__(self):
        for p in self.rng.permutation(list(self.groups)):
            yield from self.rng.permutation(self.groups[p]).tolist()

    def __len__(self):
        return len(self.ds)


def block(conv, c_in, c_out):
    return nn.Sequential(conv(c_in, c_out, 4, 2, 1), nn.GroupNorm(8, c_out), nn.SiLU())


def loss_fn(recon, x, mu, log_var, beta, pond_w):
    depth, known = x[:, :1], x[:, 1:2]
    pond, ice = known * x[:, 4:5], known * x[:, 2:3]
    e2 = (recon - depth) ** 2
    rec = pond_w * (e2 * pond).sum() / pond.sum().clamp(min=1) + (1 - pond_w) * (e2 * ice).sum() / ice.sum().clamp(min=1)
    kl = -0.5 * (1 + log_var - mu ** 2 - log_var.exp()).flatten(1).sum(1).mean()
    return rec + beta * kl, rec, kl


def evaluate(model, loader):
    model.eval(); se = sp = n = npd = 0.0
    with torch.no_grad():
        for (x,) in loader:
            recon, mu, log_var = model(x)
            e2 = (recon - x[:, :1]) ** 2; known = x[:, 1:2]; pond = known * x[:, 4:5]
            se += (e2 * known).sum().item(); n += known.sum().item(); sp += (e2 * pond).sum().item(); npd += pond.sum().item()
    model.train(); return (se / max(n, 1)) ** 0.5, (sp / max(npd, 1)) ** 0.5


def plot_round(model, xv, hist, base_rmse, blur_rmse, blur_grid, path):
    model.eval()
    with torch.no_grad():
        recon, mu, log_var = model(xv)
    model.train()
    d, rc, pond = xv[:, 0].numpy(), recon[:, 0].numpy(), (xv[:, 1] * xv[:, 4]).numpy() > 0
    bl = nn.functional.interpolate(nn.functional.adaptive_avg_pool2d(xv[:, :1], blur_grid), size=xv.shape[-1], mode="nearest")[:, 0].numpy()
    k = min(4, len(xv)); vmax = float(np.percentile(d[pond], 98)) if pond.any() else 1.0
    fig = plt.figure(figsize=(20, 4 * (k + 1) / 1.6))
    for i in range(k):
        for j, (img, title) in enumerate([(d[i], "true depth"), (rc[i], "VAE reconstruction"),
                                          (bl[i], f"blur {blur_grid}x{blur_grid} (same # values)"),
                                          (np.where(pond[i], rc[i] - d[i], np.nan), "VAE error on ponds (m)")]):
            ax = fig.add_subplot(k + 1, 5, 5 * i + j + 1)
            im = ax.imshow(img, cmap="RdBu" if j == 3 else "Blues", vmin=-0.5 if j == 3 else 0, vmax=0.5 if j == 3 else vmax)
            ax.set_axis_off(); ax.set_title(title, fontsize=8); plt.colorbar(im, ax=ax, fraction=0.046)
        ax = fig.add_subplot(k + 1, 5, 5 * i + 5)
        ax.imshow(mu[i, 0].numpy(), cmap="viridis"); ax.set_axis_off(); ax.set_title("latent mu, channel 0", fontsize=8)
    ax = fig.add_subplot(k + 1, 5, 5 * k + 1)
    bins = np.linspace(0, max(vmax, 0.1) * 1.3, 50)
    ax.hist(d[pond], bins, alpha=0.6, density=True, label="true"); ax.hist(rc[pond], bins, alpha=0.6, density=True, label="recon")
    ax.set_title("pond depth distribution (val)", fontsize=8); ax.legend(fontsize=7)
    ax = fig.add_subplot(k + 1, 5, 5 * k + 2)
    ax.hist(mu.flatten().numpy(), 80, density=True, alpha=0.7, label="mu"); ax.hist(torch.exp(0.5 * log_var).flatten().numpy(), 80, density=True, alpha=0.7, label="sigma")
    ax.set_title("latent posterior (val)", fontsize=8); ax.legend(fontsize=7)
    ax = fig.add_subplot(k + 1, 5, 5 * k + 3)
    e = [h["ep"] for h in hist]
    ax.plot(e, [h["val_pond"] for h in hist], label="val pond rmse"); ax.plot(e, [h["val_rmse"] for h in hist], label="val all rmse")
    ax.plot(e, [h["rmse"] for h in hist], label="train rmse"); ax.axhline(base_rmse, ls="--", c="k", lw=0.8, label="mean-depth baseline")
    ax.axhline(blur_rmse, ls=":", c="k", lw=0.8, label="blur, same # values")
    ax.set_xlabel("epoch"); ax.legend(fontsize=7); ax.set_title("rmse (m)", fontsize=8)
    ax = fig.add_subplot(k + 1, 5, 5 * k + 4)
    ax.plot(e, [h["kl"] for h in hist]); ax.set_xlabel("epoch"); ax.set_title("KL (nats per crop)", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=80); plt.close(fig)
    print(f"plots -> {path}", flush=True)


def save(model, opt, sched, hist, ep, path):
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), hist=hist, epoch=ep), path)
    print(f"saved -> {path}", flush=True)


def report(hist, last):
    rows = hist[-last:]
    print(f"\n{'epoch':>5} {'loss':>8} {'rmse':>7} {'kl':>8} {'val_rmse':>9} {'val_pond':>9} {'min':>5}")
    sel = rows[::max(1, len(rows) // 20)]
    for h in sel + ([rows[-1]] if sel[-1] is not rows[-1] else []):
        print(f"{h['ep']:>5} {h['loss']:>8.4f} {h['rmse']:>7.3f} {h['kl']:>8.0f} {h['val_rmse']:>9.3f} {h['val_pond']:>9.3f} {h['min']:>5.1f}")
    best = min(hist, key=lambda h: h["val_pond"])
    print(f"best val pond rmse {best['val_pond']:.3f} m at epoch {best['ep']}; this round {rows[0]['val_pond']:.3f} -> {rows[-1]['val_pond']:.3f}")


def lc(c_in, c_out, side, k=3):
    layer = LocallyConnected2d(c_in, c_out, (side, side), k, padding=k // 2, stride=1)
    # nnlayer inits every output channel to the same 1/block_size average and bias ~ N(0,1); use nn.Linear's init instead
    bound = layer.weight.shape[-1] ** -0.5
    nn.init.uniform_(layer.weight, -bound, bound); nn.init.zeros_(layer.bias)
    return layer


def lc_block(c_in, c_out, side):
    # stride-1 locally connected 3x3 at full resolution, then a 2x average pool does the downsampling the stride-2 conv did
    return nn.Sequential(lc(c_in, c_out, side), nn.GroupNorm(8, c_out), nn.SiLU(), nn.AvgPool2d(2))


class LocalVAE(nn.Module):
    def __init__(self, size, in_channels=6, latent_c=4):
        super().__init__()
        side = size // 8
        self.encoder = nn.Sequential(lc_block(in_channels, 32, size), lc_block(32, 64, size // 2), lc_block(64, 128, size // 4),
                                     lc(128, 2 * latent_c, side))
        self.decoder = nn.Sequential(lc(latent_c, 128, side), nn.SiLU(),
                                     block(nn.ConvTranspose2d, 128, 64), block(nn.ConvTranspose2d, 64, 32),
                                     block(nn.ConvTranspose2d, 32, 32), lc(32, 1, size), nn.Softplus())

    def forward(self, x):
        mu, log_var = self.encoder(x).chunk(2, dim=1)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)
        return self.decoder(z), mu, log_var


if __name__ == "__main__":
    torch.set_num_threads(4)
    CROP, BATCH, BETA, LATENT_C, ROUND, BATCHES_PER_EPOCH, POND_W = 128, 16, 1e-5, 4, 100, 100, 0.5
    CKPT, RUNS = f"vae6_lc_crop{CROP}_c{LATENT_C}_grp.pt", "vae6_runs_grp"
    os.makedirs(RUNS, exist_ok=True)
    tr, va = PondFrames(paths("train"), CROP), PondFrames(paths("val"), CROP, seed=1)
    print(f"train {len(tr)} frames / {len({p for p, _ in tr.items})} cells | val {len(va)} frames")
    dl = DataLoader(tr, batch_size=BATCH, sampler=ByFile(tr))
    xall = torch.stack([va[i] for i in range(len(va))])  # fixed val crops, scored identically every epoch
    dv = DataLoader(TensorDataset(xall), batch_size=BATCH)
    vp = xall[:, 0][(xall[:, 1] * xall[:, 4]) > 0]
    base = float(((vp - vp.mean()) ** 2).mean().sqrt())
    xv = xall[torch.argsort((xall[:, 1] * xall[:, 4]).sum((1, 2)), descending=True)[:4]]
    g = int(round((LATENT_C * (CROP // 8) ** 2) ** 0.5))
    blur = nn.functional.interpolate(nn.functional.adaptive_avg_pool2d(xall[:, :1], g), size=CROP, mode="nearest")
    pm = (xall[:, 1:2] * xall[:, 4:5]) > 0
    blur_rmse = float(((blur - xall[:, :1]) ** 2)[pm].mean().sqrt())
    print(f"val ponds: mean-depth baseline {base:.3f} m, blur-to-{g}x{g} baseline {blur_rmse:.3f} m (beat both)")
    model = LocalVAE(CROP, latent_c=LATENT_C)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.3, patience=10, min_lr=1e-5)
    hist, ep, t0, rnd = [], 0, time.time(), 0
    if os.path.exists(CKPT) and input(f"resume from {CKPT}? [y/n] ").strip().lower() == "y":
        ck = torch.load(CKPT); model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); hist, ep = ck["hist"], ck["epoch"]
    while True:
        for _ in range(ROUND):
            sl = sr = sk = 0.0; nb = 0
            for i, x in enumerate(dl):
                if i == BATCHES_PER_EPOCH:
                    break
                recon, mu, log_var = model(x)
                loss, rec, kl = loss_fn(recon, x, mu, log_var, BETA, POND_W)
                opt.zero_grad(); loss.backward(); opt.step()
                sl += loss.item(); sr += rec.item(); sk += kl.item(); nb += 1
            v_all, v_pond = evaluate(model, dv)
            sched.step(v_pond); lr = opt.param_groups[0]["lr"]
            hist.append(dict(ep=ep, loss=sl / nb, rmse=(sr / nb) ** 0.5, kl=sk / nb, val_rmse=v_all, val_pond=v_pond,
                             min=(time.time() - t0) / 60, lr=lr))
            print(f"ep {ep}: loss {sl / nb:.4f} rmse {(sr / nb) ** 0.5:.3f} kl {sk / nb:.0f} | val {v_all:.3f} pond {v_pond:.3f} | lr {lr:.0e}", flush=True)
            ep += 1
        report(hist, ROUND)
        plot_round(model, xv, hist, base, blur_rmse, g, os.path.join(RUNS, f"round{rnd:02d}_ep{ep}.png")); rnd += 1
        while True:
            a = input("\n[c]ontinue 100 epochs / [s]ave weights / [e]nd: ").strip().lower()
            if a == "s":
                save(model, opt, sched, hist, ep, CKPT)
            elif a in ("c", "e"):
                break
        if a == "e":
            if input("save before ending? [y/n] ").strip().lower() == "y":
                save(model, opt, sched, hist, ep, CKPT)
            break
