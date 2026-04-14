import torch
from tqdm import tqdm
from torch.utils.data import ConcatDataset

from src.config import get_config
from src.simulation.sim_helper import (
    initialize_modules,
    run_simulation,
)
from src.utils.dataset_io import save_dataset_to_pt
from src.data.dataset_builder import TrajectoryWindowDataset


# ── Rollout helpers ──────────────────────────────────────────────────────────

def _sample_ic(cfg, seed, x_range=3.0, v_range=30.0):
    """
    Sample initial (x, v, phi) broadly so trajectories cover states far from
    the reference A·sin(φ) ∈ [-A, A].
    """
    torch.manual_seed(seed)

    x   = (2 * torch.rand(cfg.N) - 1) * x_range   # [-x_range, x_range]
    v   = (2 * torch.rand(cfg.N) - 1) * v_range   # [-v_range, v_range]
    phi = torch.rand(cfg.N) * 2 * torch.pi         # [0, 2π]

    return x, v, phi


def generate_diverse_rollout(cfg, x, v, phi, omega_scalar):
    """
    Run one closed-loop rollout from (x, v, phi) at a fixed omega_scalar.

    Returning the full trajectory (transient + steady-state) so the dataset
    includes both off-reference and on-reference states.
    """
    omega = torch.ones(cfg.N) * omega_scalar

    oscillators, ref_gen, controller, sync_controller, omega = \
        initialize_modules(cfg, "sinusoidal", None, omega)

    return run_simulation(
        cfg, oscillators, ref_gen, controller, sync_controller,
        x.clone(), v.clone(), phi.clone(), omega,
    )


# ── Dataset builder ──────────────────────────────────────────────────────────

def build_diverse_dataset(
    cfg,
    num_ic=50,
    omegas_per_ic=4,
    horizon=20,
    predict_velocity=False,
    flatten_target=True,
    x_range=3.0,
    v_range=30.0,
    omega_range=(torch.pi, 3 * torch.pi),
    state_noise_fraction=0.05,
):
    """
    Build a training dataset with condition_mode='state_freq': input = [x, v, ω̃].

    Design choices
    --------------
    * No phase in the input — removes the direct shortcut to A·sin(φ) and
      forces the model to reason from state + frequency alone.

    * Wide initial conditions (x_range=3·A, v_range=30) — oscillators start
      far outside the reference so the data covers the full state space, not
      just the closed-loop attractor.

    * Multiple ω per initial condition — for each IC we run `omegas_per_ic`
      rollouts with different frequencies sampled from omega_range.  This
      produces explicit (same state, different ω) → (different y) pairs so
      the model learns that ω is causally relevant to the output.

    * Full trajectory per rollout — transient (off-reference) and
      steady-state (on-reference) are both included, preventing the dataset
      from collapsing onto the sinusoidal reference distribution.
    """
    datasets = []
    omega_lo, omega_hi = omega_range

    for ic_seed in tqdm(range(num_ic), desc="Generating diverse rollouts"):
        x, v, phi = _sample_ic(cfg, ic_seed, x_range=x_range, v_range=v_range)

        # Deterministically spread omegas for this IC so they don't overlap
        torch.manual_seed(ic_seed + 10_000)
        omega_values = torch.empty(omegas_per_ic).uniform_(omega_lo, omega_hi)

        for omega_scalar in omega_values.tolist():
            results = generate_diverse_rollout(cfg, x, v, phi, omega_scalar)

            ds = TrajectoryWindowDataset(
                results=results,
                horizon=horizon,
                condition_mode="state_freq",
                predict_velocity=predict_velocity,
                flatten_target=flatten_target,
                use_sin_cos_phase=False,
                state_noise_fraction=state_noise_fraction,
            )

            datasets.append(ds)

    return ConcatDataset(datasets)


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = get_config()

    num_ic           = 50
    omegas_per_ic    = 4
    horizon          = 20
    predict_velocity = True
    omega_range      = (torch.pi, 3 * torch.pi)
    state_noise_fraction  = 0.05

    ds = build_diverse_dataset(
        cfg,
        num_ic=num_ic,
        omegas_per_ic=omegas_per_ic,
        horizon=horizon,
        predict_velocity=predict_velocity,
        x_range=3.0,
        v_range=30.0,
        omega_range=omega_range,
        state_noise_fraction=state_noise_fraction,
    )

    save_dataset_to_pt(
        ds,
        "data/omega_random/dataset_state_freq.pt",
        metadata={
            "T": cfg.T,
            "dt": cfg.dt,
            "N": cfg.N,
            "A": cfg.A,
            "horizon": horizon,
            "condition_mode": "state_freq",
            "predict_velocity": predict_velocity,
            "flatten_target": True,
            "use_sin_cos_phase": False,
            "num_ic": num_ic,
            "omegas_per_ic": omegas_per_ic,
            "num_rollouts": num_ic * omegas_per_ic,
            "x_range": 3.0,
            "v_range": 30.0,
            "omega_range": list(omega_range),
            "state_noise_fraction": state_noise_fraction,
        },
    )
