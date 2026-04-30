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
    """Sample initial (x, v, phi) broadly so trajectories cover states far from
    the reference A·sin(φ) ∈ [-A, A]."""
    torch.manual_seed(seed)

    x   = (2 * torch.rand(cfg.N) - 1) * x_range
    v   = (2 * torch.rand(cfg.N) - 1) * v_range
    phi = torch.rand(cfg.N) * 2 * torch.pi

    return x, v, phi


def generate_diverse_rollout(cfg, x, v, phi, omega_scalar):
    """Run one closed-loop rollout from (x, v, phi) at a fixed omega_scalar.
    Returns the full trajectory (transient + steady-state)."""
    omega = torch.ones(cfg.N) * omega_scalar

    oscillators, ref_gen, controller, sync_controller, omega, phase_estimator = \
        initialize_modules(cfg, "sinusoidal", None, omega)

    return run_simulation(
        cfg, oscillators, ref_gen, controller, sync_controller,
        x.clone(), v.clone(), phi.clone(), omega, phase_estimator
    )


# ── Dataset builder ──────────────────────────────────────────────────────────

def build_diverse_dataset(
    cfg,
    num_ic=50,
    omegas_per_ic=4,
    horizon=20,
    condition_mode="state_freq",
    predict_velocity=False,
    flatten_target=True,
    x_range=3.0,
    v_range=30.0,
    omega_range=(torch.pi, 3 * torch.pi),
    state_noise_fraction=0.05,
    noise_copies=1,
):
    """
    Build a diverse training dataset with selectable condition mode.

    condition_mode options:
      "state_freq"            -> [x, v, ω̃]                     (no phase)
      "state_phase_trig_freq" -> [x, v, sin_phi, cos_phi, ω̃]   (trig phase)

    For the diffusion reference generator, use "state_phase_trig_freq" so
    the model knows the current phase and produces step-to-step coherent
    references. At inference, sin/cos come from the phase estimator.

    Wide ICs (x_range=3·A, v_range=30) ensure the dataset covers the full
    state space. Multiple omegas per IC produce explicit frequency-conditioned
    contrastive pairs.
    """
    use_sin_cos  = condition_mode == "state_phase_trig_freq"
    builder_mode = "state_phase_trig_freq" if use_sin_cos else "state_freq"

    datasets = []
    omega_lo, omega_hi = omega_range

    for ic_seed in tqdm(range(num_ic), desc="Generating diverse rollouts"):
        x, v, phi = _sample_ic(cfg, ic_seed, x_range=x_range, v_range=v_range)

        torch.manual_seed(ic_seed + 10_000)
        omega_values = torch.empty(omegas_per_ic).uniform_(omega_lo, omega_hi)

        for omega_scalar in omega_values.tolist():
            results = generate_diverse_rollout(cfg, x, v, phi, omega_scalar)

            ds = TrajectoryWindowDataset(
                results=results,
                horizon=horizon,
                condition_mode=builder_mode,
                predict_velocity=predict_velocity,
                flatten_target=flatten_target,
                use_sin_cos_phase=use_sin_cos,
                state_noise_fraction=state_noise_fraction,
                noise_copies=noise_copies,
            )

            datasets.append(ds)

    return ConcatDataset(datasets)


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = get_config()

    num_ic               = 50
    omegas_per_ic        = 4
    horizon              = 20
    condition_mode       = "state_phase_trig_freq"  # includes sin/cos phase for diffusion
    predict_velocity     = True
    omega_range          = (torch.pi, 3 * torch.pi)
    state_noise_fraction = 0.05
    noise_copies         = 1

    ds = build_diverse_dataset(
        cfg,
        num_ic=num_ic,
        omegas_per_ic=omegas_per_ic,
        horizon=horizon,
        condition_mode=condition_mode,
        predict_velocity=predict_velocity,
        x_range=3.0,
        v_range=30.0,
        omega_range=omega_range,
        state_noise_fraction=state_noise_fraction,
        noise_copies=noise_copies,
    )

    save_dataset_to_pt(
        ds,
        f"data/dataset_{condition_mode}.pt",
        metadata={
            "T": cfg.T,
            "dt": cfg.dt,
            "N": cfg.N,
            "A": cfg.A,
            "horizon": horizon,
            "condition_mode": condition_mode,
            "predict_velocity": predict_velocity,
            "flatten_target": True,
            "use_sin_cos_phase": condition_mode == "state_phase_trig_freq",
            "num_ic": num_ic,
            "omegas_per_ic": omegas_per_ic,
            "num_rollouts": num_ic * omegas_per_ic,
            "x_range": 3.0,
            "v_range": 30.0,
            "omega_range": list(omega_range),
            "state_noise_fraction": state_noise_fraction,
            "noise_copies": noise_copies,
        },
    )
