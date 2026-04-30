"""
Train the Periodic Autoencoder (PAE) with temporal consistency.

Loss = reconstruction(x_t, x_{t+1}) + λ · ||z_{t+1} - R(ω·dt)·z_t||²

The temporal term forces the latent z to rotate by the known physical angle
ω·dt per step — no phase labels needed.  This disambiguates phase completely
(up to a global offset, which doesn't affect sync).

Run from Realistic_Analytics/:
    .venv/bin/python -m src.training.train_pae
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

from src.config import get_config, get_phase_estimator_config
from src.models.phase_autoencoder import PeriodicAutoencoder
from src.data.phase_estimator_dataset import build_temporal_pae_dataset
from src.utils.dataset_io import compute_norm_stats


TEMPORAL_WEIGHT = 10.0   # λ: weight of temporal loss vs reconstruction


def materialise(concat_dataset):
    """Collect all (inp_t, inp_next) pairs from a ConcatDataset into tensors."""
    ts, nexts = [], []
    for inp_t, inp_next in concat_dataset:
        ts.append(inp_t)
        nexts.append(inp_next)
    return torch.stack(ts), torch.stack(nexts)


def make_dataloaders(cfg, train_cfg):
    print("Building temporal PAE dataset …")
    dataset = build_temporal_pae_dataset(cfg, num_ic=50, omegas_per_ic=4)

    print("Materialising tensors …")
    inp_t, inp_next = materialise(dataset)   # [N, 3] each

    n       = len(inp_t)
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)

    perm   = torch.randperm(n)
    tr     = perm[:n_train]
    va     = perm[n_train:n_train + n_val]
    te     = perm[n_train + n_val:]

    # Norm stats from training split of inp_t only
    # (inp_next has same distribution — no separate stats needed)
    norm_stats = compute_norm_stats(inp_t[tr], inp_next[tr])
    mean_X, std_X = norm_stats["mean_X"], norm_stats["std_X"]

    def norm(x):
        return (x - mean_X) / std_X

    it_norm   = norm(inp_t)
    inext_norm = norm(inp_next)

    train_ds = TensorDataset(it_norm[tr],   inext_norm[tr])
    val_ds   = TensorDataset(it_norm[va],   inext_norm[va])
    test_ds  = TensorDataset(it_norm[te],   inext_norm[te])

    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=train_cfg.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=train_cfg.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, norm_stats


def denorm_omega(xb_norm, norm_stats, device):
    """Recover physical ω from the normalised batch (column 2)."""
    mean_w = norm_stats["mean_X"][2].to(device)
    std_w  = norm_stats["std_X"][2].to(device)
    return xb_norm[:, 2] * std_w + mean_w


def evaluate(model, loader, device, dt, norm_stats):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for xb, xnb in loader:
            xb, xnb = xb.to(device), xnb.to(device)
            omega_raw = denorm_omega(xb, norm_stats, device)
            loss, _ = model.compute_temporal_loss(xb, xnb, omega_raw, dt, TEMPORAL_WEIGHT)
            total += loss.item() * xb.shape[0]
            count += xb.shape[0]
    return total / count


def phase_quality_from_loader(model, loader, device):
    """Cosine similarity between consecutive latent rotation and expected R(ω·dt)·z."""
    model.eval()
    sims = []
    with torch.no_grad():
        for xb, xnb in loader:
            xb, xnb = xb.to(device), xnb.to(device)
            z_t    = model.encode(xb)
            z_next = model.encode(xnb)
            # normalise to unit circle for direction comparison
            z_t_n    = F.normalize(z_t,    dim=-1)
            z_next_n = F.normalize(z_next, dim=-1)
            sim = (z_t_n * z_next_n).sum(-1)   # cos of rotation between steps
            sims.append(sim)
    all_sim = torch.cat(sims)
    # Consecutive frames should have high cosine similarity (small ω·dt ~ 0.03 rad)
    return all_sim.mean().item()


def train_pae(train_cfg, cfg, device, model_save_path="models/pae.pt"):
    dt = cfg.dt

    train_loader, val_loader, test_loader, norm_stats = \
        make_dataloaders(cfg, train_cfg)

    model = PeriodicAutoencoder(
        hidden_dim=train_cfg.hidden_dim,
        num_layers=train_cfg.num_layers,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )

    best_val, best_state = float("inf"), None
    pbar = tqdm(range(1, train_cfg.epochs + 1))

    for epoch in pbar:
        pbar.set_description(f"PAE epoch {epoch}/{train_cfg.epochs}")

        model.train()
        total, count = 0.0, 0
        for xb, xnb in train_loader:
            xb, xnb = xb.to(device), xnb.to(device)
            omega_raw = denorm_omega(xb, norm_stats, device)
            optimizer.zero_grad()
            loss, metrics = model.compute_temporal_loss(xb, xnb, omega_raw, dt, TEMPORAL_WEIGHT)
            loss.backward()
            optimizer.step()
            total += loss.item() * xb.shape[0]
            count += xb.shape[0]

        train_loss = total / count
        val_loss   = evaluate(model, val_loader, device, dt, norm_stats)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                "model_state_dict": model.state_dict(),
                "hidden_dim": train_cfg.hidden_dim,
                "num_layers": train_cfg.num_layers,
                "norm_stats": {k: v.cpu() for k, v in norm_stats.items()},
            }

        if epoch % 10 == 0 or epoch == 1:
            cos_sim = phase_quality_from_loader(model, val_loader, device)
            print(f"Epoch {epoch:03d} | train={train_loss:.6f} | "
                  f"val={val_loss:.6f} | latent cos-sim={cos_sim:.4f}")
            print(f"          recon={metrics['recon']:.6f} | "
                  f"temporal={metrics['temporal']:.6f}")

    save_path = Path(model_save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, save_path)
    print(f"\nSaved PAE → {save_path}  (best val: {best_val:.6f})")

    model.load_state_dict(best_state["model_state_dict"])
    test_loss = evaluate(model, test_loader, device, dt, norm_stats)
    print(f"Test loss: {test_loss:.6f}")

    return model, best_state


if __name__ == "__main__":
    cfg       = get_config()
    train_cfg = get_phase_estimator_config()
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_pae(train_cfg, cfg, device, model_save_path="models/pae.pt")
