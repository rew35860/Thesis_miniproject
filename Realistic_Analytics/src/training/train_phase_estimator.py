"""
Train the MLP phase estimator (supervised: (x, v, ω) → (sin φ, cos φ)).

Run from Realistic_Analytics/:
    .venv/bin/python -m src.training.train_phase_estimator
"""

from pathlib import Path
import torch
from tqdm import tqdm

from src.config import get_config, get_phase_estimator_config
from src.models.phase_estimator import PhaseEstimatorMLP
from src.utils.dataset_io import make_dataloaders


def _eval_loss(model, loader, device):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss, _ = model.compute_loss(xb, yb)
            total += loss.item() * xb.shape[0]
            count += xb.shape[0]
    return total / count


def train_phase_estimator(
    dataset_path,
    train_cfg,
    device,
    model_save_path="models/phase_estimator.pt",
):
    _, X, Y, metadata, norm_stats, train_loader, val_loader, test_loader = make_dataloaders(
        dataset_path=dataset_path,
        batch_size=train_cfg.batch_size,
        return_metadata=True,
        device=device,
    )

    print(f"Dataset:   {dataset_path}")
    print(f"Input dim: {X.shape[1]}  |  Output dim: {Y.shape[1]}  |  Samples: {len(X)}")

    model = PhaseEstimatorMLP(
        hidden_dim=train_cfg.hidden_dim,
        num_layers=train_cfg.num_layers,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )

    best_val_loss, best_state = float("inf"), None
    train_losses, val_losses  = [], []

    pbar = tqdm(range(1, train_cfg.epochs + 1))
    for epoch in pbar:
        pbar.set_description(f"PhaseEst epoch {epoch}/{train_cfg.epochs}")

        model.train()
        total_loss, total_count = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss, _ = model.compute_loss(xb, yb)
            loss.backward()
            optimizer.step()
            total_loss  += loss.item() * xb.shape[0]
            total_count += xb.shape[0]

        train_loss = total_loss / total_count
        val_loss   = _eval_loss(model, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                "model_state_dict": model.state_dict(),
                "hidden_dim":       train_cfg.hidden_dim,
                "num_layers":       train_cfg.num_layers,
                "input_dim":        X.shape[1],
                "metadata":         metadata,
                "norm_stats":       {k: v.cpu() for k, v in norm_stats.items()},
            }

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | train={train_loss:.6f} | val={val_loss:.6f}")

    save_path = Path(model_save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, save_path)
    print(f"\nSaved → {save_path}  (best val: {best_val_loss:.6f})")

    model.load_state_dict(best_state["model_state_dict"])
    print(f"Test loss: {_eval_loss(model, test_loader, device):.6f}")

    return model, norm_stats


if __name__ == "__main__":
    cfg       = get_config()
    train_cfg = get_phase_estimator_config()
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_phase_estimator(
        dataset_path="data/omega_random/dataset_phase_estimator.pt",
        train_cfg=train_cfg,
        device=device,
        model_save_path="models/phase_estimator.pt",
    )
