"""
Quick diagnostic: verify diffusion model output quality before running simulation.

Run from Realistic_Analytics/:
    .venv/bin/python diagnose_diffusion.py
"""

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.diffusion import load_diffusion_model
from src.utils.dataset_io import make_dataloaders, denormalize_Y

DEVICE = torch.device("cpu")
MODEL_PATH = "models/diffusion_state_freq.pt"
DATASET_PATH = "data/dataset_state_freq.pt"
SAVE_DIR = "graphs/diffusion"
N_SAMPLES = 5      # how many stochastic samples to overlay per condition
N_STEPS_PLOT = 200 # how many dataset steps to show in trajectory


def main():
    # ── Load model ───────────────────────────────────────────────────────────
    model, ckpt = load_diffusion_model(MODEL_PATH, DEVICE)
    norm_stats = ckpt.get("norm_stats")
    horizon = ckpt["metadata"]["horizon"]
    predict_velocity = ckpt["metadata"]["predict_velocity"]
    dt = ckpt["metadata"]["dt"]
    print(f"Model loaded. horizon={horizon}, predict_velocity={predict_velocity}")
    if norm_stats:
        print("  norm_stats present — denormalizing outputs")

    # ── Load dataset ─────────────────────────────────────────────────────────
    _, X, Y, metadata, loaded_norm_stats, train_loader, val_loader, _ = make_dataloaders(
        dataset_path=DATASET_PATH,
        batch_size=256,
        return_metadata=True,
    )
    print(f"Dataset: X={X.shape}, Y={Y.shape}")

    # ── 1. Loss on validation set ─────────────────────────────────────────────
    val_loss = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss, _ = model.compute_loss(xb, yb)
            val_loss += loss.item() * xb.shape[0]
            count += xb.shape[0]
    val_loss /= count
    print(f"\nValidation noise-prediction MSE: {val_loss:.6f}")
    print("  (lower = model predicts noise better; ~1.0 = no learning)")

    # stride=2 skips noisy copies when noise_copies=1 (clean[t], noisy[t], clean[t+1], ...)
    # Use 1 if dataset was built with noise_copies=0
    STRIDE = 2

    # ── 2. Single-step: individual samples vs mean ────────────────────────────
    # Use a mid-rollout index (even → clean sample), skip transient
    idx = 200 * STRIDE
    cond = X[idx].unsqueeze(0).to(DEVICE)
    y_true = Y[idx].unsqueeze(0)

    if norm_stats:
        y_true_plot = denormalize_Y(y_true, norm_stats)
    else:
        y_true_plot = y_true

    x_true = y_true_plot.squeeze(0)[:horizon].numpy()
    t = torch.arange(horizon).float() * dt

    with torch.no_grad():
        cond_rep = cond.expand(N_SAMPLES, -1)
        ys = model.sample(cond_rep).cpu()            # [N_SAMPLES, output_dim]
        if norm_stats:
            ys = denormalize_Y(ys, norm_stats)
        y_mean = ys.mean(dim=0)

    plt.figure(figsize=(10, 4))
    plt.plot(t, x_true, "k-", lw=2, label="ground truth")
    for s in range(N_SAMPLES):
        plt.plot(t, ys[s, :horizon].numpy(), alpha=0.35, color="steelblue",
                 label="individual samples" if s == 0 else None)
    plt.plot(t, y_mean[:horizon].numpy(), "r--", lw=2, label=f"mean of {N_SAMPLES}")
    plt.xlabel("Time (s)")
    plt.ylabel("Position")
    plt.title(f"DDPM samples vs mean (idx={idx}, mid-rollout)")
    plt.legend()
    plt.tight_layout()
    path = f"{SAVE_DIR}/diagnostic_single_step.png"
    plt.savefig(path)
    plt.close()
    print(f"\nSaved single-step plot → {path}")

    # ── 3. Mean-sample trajectory vs ground truth ─────────────────────────────
    # Step by STRIDE to only hit clean samples (avoid interleaved noisy copies)
    preds_1, preds_5, trues = [], [], []
    model.eval()
    with torch.no_grad():
        for step in range(N_STEPS_PLOT):
            i = step * STRIDE  # clean samples only
            c = X[i].unsqueeze(0).to(DEVICE)
            y_t = Y[i].unsqueeze(0)

            y1 = model.sample(c).cpu()
            y5 = model.sample(c.expand(5, -1)).mean(0, keepdim=True).cpu()

            if norm_stats:
                y_t = denormalize_Y(y_t, norm_stats)
                y1  = denormalize_Y(y1,  norm_stats)
                y5  = denormalize_Y(y5,  norm_stats)

            trues.append(y_t.squeeze(0)[0].item())
            preds_1.append(y1.squeeze(0)[0].item())
            preds_5.append(y5.squeeze(0)[0].item())

    t_axis = torch.arange(N_STEPS_PLOT).float() * dt
    plt.figure(figsize=(12, 4))
    plt.plot(t_axis, trues,    "k-",  lw=2, label="ground truth")
    plt.plot(t_axis, preds_1,  alpha=0.5,   label="single sample")
    plt.plot(t_axis, preds_5,  alpha=0.8,   label="mean of 5 samples")
    plt.xlabel("Time (s)")
    plt.ylabel("Position")
    plt.title("Trajectory: ground truth vs single sample vs mean×5")
    plt.legend()
    plt.tight_layout()
    path = f"{SAVE_DIR}/diagnostic_trajectory.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved trajectory plot → {path}")

    # ── 4. Sweep K: how many samples needed for a stable mean? ──────────────
    trues_t   = torch.tensor(trues)
    preds_1_t = torch.tensor(preds_1)
    preds_5_t = torch.tensor(preds_5)
    rmse_1 = ((trues_t - preds_1_t)**2).mean().sqrt().item()
    rmse_5 = ((trues_t - preds_5_t)**2).mean().sqrt().item()
    print(f"\nTrajectory RMSE (single sample): {rmse_1:.4f}")
    print(f"Trajectory RMSE (mean of 5):     {rmse_5:.4f}")

    print("\nSweep: RMSE vs number of samples averaged (clean inputs only)")
    for K in [1, 3, 5, 10, 20]:
        preds_k = []
        with torch.no_grad():
            for step in range(N_STEPS_PLOT):
                i = step * STRIDE
                c = X[i].unsqueeze(0).to(DEVICE)
                yk = model.sample(c.expand(K, -1)).mean(0, keepdim=True).cpu()
                if norm_stats:
                    yk = denormalize_Y(yk, norm_stats)
                preds_k.append(yk.squeeze(0)[0].item())
        preds_k_t = torch.tensor(preds_k)
        rmse_k = ((trues_t - preds_k_t)**2).mean().sqrt().item()
        print(f"  K={K:2d}: RMSE={rmse_k:.4f}")

    # ── 5. Steady-state RMSE (skip first 40% = transient) ────────────────────
    skip = int(0.4 * N_STEPS_PLOT)
    trues_ss   = trues_t[skip:]
    preds_5_ss = preds_5_t[skip:]
    rmse_ss = ((trues_ss - preds_5_ss)**2).mean().sqrt().item()
    print(f"\nSteady-state RMSE (mean×5, last 60% of steps): {rmse_ss:.4f}")
    print("  If this is < 0.10, the model is good enough for the control loop.")


if __name__ == "__main__":
    main()
