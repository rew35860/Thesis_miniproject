import torch
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import ConcatDataset, DataLoader

from Realistic_Analytics.main import (
    get_config,
    initialize_states,
    initialize_modules,
    run_simulation,
)
from Realistic_Analytics.src.utils.dataset_io import save_dataset_to_pt
from Realistic_Analytics.src.data.dataset_builder import TrajectoryWindowDataset


def generate_rollout(cfg, seed):
    x, v, phi, omega = initialize_states(cfg["N"], cfg["device"], seed=seed)
    oscillators, reference_generator, controller, sync_controller, omega = initialize_modules(cfg, omega)

    results = run_simulation(
        cfg,
        oscillators,
        reference_generator,
        controller,
        sync_controller,
        x,
        v,
        phi,
        omega,
    )
    return results


def build_dataset_from_many_rollouts(
    num_rollouts=50,
    horizon=20,
    condition_mode="phase_freq",
    predict_velocity=False,
    flatten_target=True,
    use_sin_cos_phase=False,
):
    cfg = get_config()
    datasets = []

    for seed in tqdm(range(num_rollouts), desc="Generating rollouts"):
        results = generate_rollout(cfg, seed=seed)

        ds = TrajectoryWindowDataset(
            results=results,
            horizon=horizon,
            condition_mode=condition_mode,
            predict_velocity=predict_velocity,
            flatten_target=flatten_target,
            use_sin_cos_phase=use_sin_cos_phase,
        )
        datasets.append(ds)

    return ConcatDataset(datasets)


def dataset_to_tensors(dataset):
    """
    Convert either a TrajectoryWindowDataset or a ConcatDataset
    into two tensors: X and Y.
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

    return X, Y


def save_dataset_to_pt(dataset, path):
    X, Y = dataset_to_tensors(dataset)

    torch.save(
        {
            "inputs": X,
            "targets": Y,
        },
        path,
    )

    print(f"Saved dataset to {path}")
    print("inputs shape:", X.shape)
    print("targets shape:", Y.shape)


def load_dataset_from_pt(path):
    data = torch.load(path)

    X = data["inputs"].float()
    Y = data["targets"].float()

    print(f"Loaded dataset from {path}")
    print("inputs shape:", X.shape)
    print("targets shape:", Y.shape)

    return X, Y


if __name__ == "__main__":
    # 1) phase + frequency only
    ds_phase = build_dataset_from_many_rollouts(
        num_rollouts=50,
        horizon=20,
        condition_mode="phase_freq",
        predict_velocity=False,
        flatten_target=True,
        use_sin_cos_phase=True,   # better than raw wrapped phi
    )

    # 2) state + phase + frequency
    ds_state = build_dataset_from_many_rollouts(
        num_rollouts=50,
        horizon=20,
        condition_mode="state_phase_freq",
        predict_velocity=False,
        flatten_target=True,
        use_sin_cos_phase=True,
    )

    DATA_DIR = Path("Realistic_Analytics/data")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    save_dataset_to_pt(
        ds_phase,
        DATA_DIR / "dataset_phase_freq.pt",
        metadata={
            "horizon": 20,
            "condition_mode": "phase_freq",
            "predict_velocity": False,
            "flatten_target": True,
            "use_sin_cos_phase": True,
            "num_rollouts": 50,
        },
    )

    save_dataset_to_pt(
        ds_state,
        DATA_DIR / "dataset_state_phase_freq.pt",
        metadata={
            "horizon": 20,
            "condition_mode": "state_phase_freq",
            "predict_velocity": False,
            "flatten_target": True,
            "use_sin_cos_phase": True,
            "num_rollouts": 50,
        },
    )

    print("phase+freq dataset size:", len(ds_phase))
    print("state+phase+freq dataset size:", len(ds_state))

    # inspect one batch
    loader = DataLoader(ds_phase, batch_size=32, shuffle=True)
    xb, yb = next(iter(loader))
    print("phase+freq input shape:", xb.shape)
    print("phase+freq target shape:", yb.shape)

    loader2 = DataLoader(ds_state, batch_size=32, shuffle=True)
    xb2, yb2 = next(iter(loader2))
    print("state+phase+freq input shape:", xb2.shape)
    print("state+phase+freq target shape:", yb2.shape)