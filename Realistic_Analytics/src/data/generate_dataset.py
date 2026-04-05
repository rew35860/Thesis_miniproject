import torch
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import ConcatDataset

from src.config import get_config
from src.simulation.sim_helper import (
    initialize_states,
    initialize_modules,
    run_simulation,
)
from src.utils.dataset_io import save_dataset_to_pt
from src.data.dataset_builder import TrajectoryWindowDataset


def generate_rollout(cfg, seed):
    x, v, phi, omega = initialize_states(cfg, seed=seed)

    oscillators, ref_gen, controller, sync_controller, omega = \
        initialize_modules(cfg, "sinusoidal", None, omega)

    results = run_simulation(
        cfg,
        oscillators,
        ref_gen,
        controller,
        sync_controller,
        x,
        v,
        phi,
        omega,
    )
    return results


def build_dataset_from_many_rollouts(
    cfg,
    num_rollouts=50,
    horizon=20,
    condition_mode="phase_freq",
    predict_velocity=False,
    flatten_target=True,
    use_sin_cos_phase=False,
):
    datasets = []

    for seed in tqdm(range(num_rollouts), desc="Generating rollouts"):
        results = generate_rollout(cfg, seed)

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


if __name__ == "__main__":
    cfg = get_config()

    # 1) phase + frequency only
    ds_phase = build_dataset_from_many_rollouts(
        cfg,
        num_rollouts=50,
        horizon=20,
        condition_mode="phase_freq",
        predict_velocity=True,
        use_sin_cos_phase=True,
    )

    # 2) state + phase + frequency
    ds_state = build_dataset_from_many_rollouts(
        cfg,
        num_rollouts=50,
        horizon=20,
        condition_mode="state_phase_freq",
        predict_velocity=True,
        use_sin_cos_phase=True,
    )

    path_data_dir = "data"

    save_dataset_to_pt(
        ds_phase,
        f"{path_data_dir}/dataset_phase_trig_freq.pt",
        metadata={
            "T": cfg.T,
            "dt": cfg.dt,
            "N": cfg.N,
            "horizon": 20,
            "condition_mode": "phase_freq",
            "predict_velocity": True,
            "flatten_target": True,
            "use_sin_cos_phase": True,
            "num_rollouts": 50,
        },
    )

    save_dataset_to_pt(
        ds_state,
        f"{path_data_dir}/dataset_state_phase_trig_freq.pt",
        metadata={
            "T": cfg.T,
            "dt": cfg.dt,
            "N": cfg.N,
            "horizon": 20,
            "condition_mode": "state_phase_freq",
            "predict_velocity": True,
            "flatten_target": True,
            "use_sin_cos_phase": True,
            "num_rollouts": 50,
        },
    )