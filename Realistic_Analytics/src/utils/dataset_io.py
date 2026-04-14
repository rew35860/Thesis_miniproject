import torch
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader


def dataset_to_tensors(dataset):
    """Convert a dataset into two tensors: X and Y."""
    if hasattr(dataset, "inputs") and hasattr(dataset, "targets"):
        X = dataset.inputs
        Y = dataset.targets
    else:
        xs, ys = [], []
        for x, y in dataset:
            xs.append(x)
            ys.append(y)
        X = torch.stack(xs)
        Y = torch.stack(ys)

    return X.float(), Y.float()


def save_dataset_to_pt(dataset, path, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    X, Y = dataset_to_tensors(dataset)

    payload = {"inputs": X, "targets": Y}
    if metadata is not None:
        payload["metadata"] = metadata

    torch.save(payload, path)

    print(f"Saved dataset to {path}")
    print("inputs shape:", X.shape)
    print("targets shape:", Y.shape)
    if metadata is not None:
        print("metadata:", metadata)


def load_dataset_from_pt(path, return_metadata=False, device="cpu"):
    data = torch.load(path, map_location=device)

    X = data["inputs"].float()
    Y = data["targets"].float()
    metadata = data.get("metadata", None)

    if return_metadata and metadata is None:
        raise ValueError("Metadata not found in dataset file")

    if return_metadata:
        return X, Y, metadata
    return X, Y


def load_tensor_dataset(path, return_metadata=False, device="cpu"):
    if return_metadata:
        X, Y, metadata = load_dataset_from_pt(path, return_metadata=True, device=device)
        return TensorDataset(X, Y), X, Y, metadata

    X, Y = load_dataset_from_pt(path, return_metadata=False, device=device)
    return TensorDataset(X, Y), X, Y


# ── Normalization ────────────────────────────────────────────────────────────

def compute_norm_stats(X, Y):
    """
    Compute per-feature mean and std from tensors X and Y.
    Should be called on the training split only to avoid data leakage.
    """
    return {
        "mean_X": X.mean(0),
        "std_X":  X.std(0) + 1e-8,
        "mean_Y": Y.mean(0),
        "std_Y":  Y.std(0) + 1e-8,
    }


def apply_normalization(X, Y, stats):
    """Normalize X and Y using precomputed stats."""
    X_norm = (X - stats["mean_X"]) / stats["std_X"]
    Y_norm = (Y - stats["mean_Y"]) / stats["std_Y"]
    return X_norm, Y_norm


def denormalize_Y(Y_norm, stats):
    """Invert normalization on model outputs for evaluation / inference."""
    return Y_norm * stats["std_Y"] + stats["mean_Y"]


# ── Dataloaders ──────────────────────────────────────────────────────────────

def make_dataloaders(
    dataset_path,
    batch_size=128,
    train_ratio=0.8,
    val_ratio=0.1,
    seed=42,
    return_metadata=False,
    device="cpu",
):
    """
    Load a dataset, split into train/val/test, normalize using train-split
    statistics, and return DataLoaders together with the norm stats.

    Returns (return_metadata=True):
        X, Y, metadata, norm_stats, train_loader, val_loader, test_loader

    Returns (return_metadata=False):
        X, Y, norm_stats, train_loader, val_loader, test_loader

    X and Y are the full normalized tensors (useful for quick inspection).
    norm_stats contains mean_X, std_X, mean_Y, std_Y for denormalization.
    """
    if return_metadata:
        _, X, Y, metadata = load_tensor_dataset(dataset_path, return_metadata=True, device=device)
    else:
        _, X, Y = load_tensor_dataset(dataset_path, device=device)
        metadata = None

    # ── Split indices ────────────────────────────────────────────────────────
    n_total = len(X)
    n_train = int(train_ratio * n_total)
    n_val   = int(val_ratio   * n_total)

    generator = torch.Generator().manual_seed(seed)
    perm      = torch.randperm(n_total, generator=generator)
    train_idx = perm[:n_train]
    val_idx   = perm[n_train : n_train + n_val]
    test_idx  = perm[n_train + n_val :]

    # ── Normalization (stats from train split only) ──────────────────────────
    norm_stats    = compute_norm_stats(X[train_idx], Y[train_idx])
    X_norm, Y_norm = apply_normalization(X, Y, norm_stats)

    # ── Build split datasets ─────────────────────────────────────────────────
    train_ds = TensorDataset(X_norm[train_idx], Y_norm[train_idx])
    val_ds   = TensorDataset(X_norm[val_idx],   Y_norm[val_idx])
    test_ds  = TensorDataset(X_norm[test_idx],  Y_norm[test_idx])

    # ── DataLoaders ──────────────────────────────────────────────────────────
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    if return_metadata:
        return X_norm, Y_norm, metadata, norm_stats, train_loader, val_loader, test_loader

    return X_norm, Y_norm, norm_stats, train_loader, val_loader, test_loader
