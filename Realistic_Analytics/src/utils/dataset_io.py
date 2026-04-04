import torch
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader, random_split


def dataset_to_tensors(dataset):
    """
    Convert a dataset into two tensors: X and Y.
    """
    if hasattr(dataset, "inputs") and hasattr(dataset, "targets"):
        X = dataset.inputs
        Y = dataset.targets
    else:
        xs = []
        ys = []
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

    payload = {
        "inputs": X,
        "targets": Y,
    }

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
        dataset = TensorDataset(X, Y)
        return dataset, X, Y, metadata

    X, Y = load_dataset_from_pt(path, return_metadata=False, device=device)
    dataset = TensorDataset(X, Y)
    return dataset, X, Y


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
    Load dataset from disk, split into train/val/test,
    and return DataLoaders.

    Returns:
        dataset, X, Y, train_loader, val_loader, test_loader
    or
        dataset, X, Y, metadata, train_loader, val_loader, test_loader
    """
    if return_metadata:
        dataset, X, Y, metadata = load_tensor_dataset(
            dataset_path,
            return_metadata=True,
            device=device
        )
    else:
        dataset, X, Y = load_tensor_dataset(
            dataset_path,
            return_metadata=False,
            device=device

        )
        metadata = None

    n_total = len(dataset)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    n_test = n_total - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    if return_metadata:
        return dataset, X, Y, metadata, train_loader, val_loader, test_loader

    return dataset, X, Y, train_loader, val_loader, test_loader