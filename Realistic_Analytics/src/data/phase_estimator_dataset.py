import torch
from torch.utils.data import Dataset, ConcatDataset
from tqdm import tqdm

from src.config import get_config
from src.simulation.sim_helper import initialize_modules, run_simulation
from src.utils.dataset_io import save_dataset_to_pt


class PhaseEstimatorDataset(Dataset):
    """
    Supervised dataset for the phase estimator.

    Input:  [x_t, v_t, omega_t]          shape [3]
    Target: [sin(phi_t), cos(phi_t)]     shape [2]

    sin/cos encoding avoids the phase-wrapping problem and is exactly what
    the SynchronizationController needs: sin(phi_i - phi_j) expands as
    sin_i*cos_j - cos_i*sin_j, so no atan2 is needed at inference time.
    """

    def __init__(self, results, omega_scalar):
        x           = results["x"]            # [T, N]
        v           = results["v"]            # [T, N]
        phi         = results["phi"]          # [T, N]
        T, N        = x.shape

        inputs  = []
        targets = []

        for i in range(N):
            for t in range(T):
                inp = torch.tensor(
                    [x[t, i], v[t, i], omega_scalar],
                    dtype=torch.float32,
                )
                tgt = torch.tensor(
                    [torch.sin(phi[t, i]), torch.cos(phi[t, i])],
                    dtype=torch.float32,
                )
                inputs.append(inp)
                targets.append(tgt)

        self.inputs  = torch.stack(inputs)   # [T*N, 3]
        self.targets = torch.stack(targets)  # [T*N, 2]

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class TemporalPAEDataset(Dataset):
    """
    Consecutive-pair dataset for temporal PAE training.

    Each sample is a pair of adjacent timesteps from the same oscillator:
        inp_t    : [x_t,   v_t,   ω]   shape [3]
        inp_next : [x_t+1, v_t+1, ω]   shape [3]

    The temporal consistency loss uses the fact that phase must advance by
    exactly ω·dt per step:  z_{t+1} = R(ω·dt) · z_t
    where R is a 2D rotation matrix.  No phase labels required.
    """

    def __init__(self, results, omega_scalar, dt):
        x    = results["x"]    # [T, N]
        v    = results["v"]    # [T, N]
        T, N = x.shape

        inp_t_list    = []
        inp_next_list = []

        for i in range(N):
            for t in range(T - 1):
                inp_t    = torch.tensor([x[t,   i], v[t,   i], omega_scalar], dtype=torch.float32)
                inp_next = torch.tensor([x[t+1, i], v[t+1, i], omega_scalar], dtype=torch.float32)
                inp_t_list.append(inp_t)
                inp_next_list.append(inp_next)

        self.inp_t    = torch.stack(inp_t_list)     # [M, 3]
        self.inp_next = torch.stack(inp_next_list)  # [M, 3]
        self.dt       = dt

    def __len__(self):
        return self.inp_t.shape[0]

    def __getitem__(self, idx):
        return self.inp_t[idx], self.inp_next[idx]


# ── Dataset generation ───────────────────────────────────────────────────────

def _sample_ic(cfg, seed, x_range=3.0, v_range=30.0):
    torch.manual_seed(seed)
    x   = (2 * torch.rand(cfg.N) - 1) * x_range
    v   = (2 * torch.rand(cfg.N) - 1) * v_range
    phi = torch.rand(cfg.N) * 2 * torch.pi
    return x, v, phi


def build_phase_estimator_dataset(
    cfg,
    num_ic=50,
    omegas_per_ic=4,
    x_range=3.0,
    v_range=30.0,
    omega_range=(torch.pi, 3 * torch.pi),
):
    """
    Build a dataset for training the phase estimator.

    Uses the same diverse IC + multi-omega strategy as the reference model
    dataset so the estimator is trained on the same state distribution it
    will encounter at inference time.
    """
    datasets = []
    omega_lo, omega_hi = omega_range

    for ic_seed in tqdm(range(num_ic), desc="Generating phase estimator rollouts"):
        x, v, phi = _sample_ic(cfg, ic_seed, x_range=x_range, v_range=v_range)

        torch.manual_seed(ic_seed + 10_000)
        omega_values = torch.empty(omegas_per_ic).uniform_(omega_lo, omega_hi)

        for omega_scalar in omega_values.tolist():
            omega = torch.ones(cfg.N) * omega_scalar
            oscillators, ref_gen, controller, sync_controller, omega, _ = \
                initialize_modules(cfg, "sinusoidal", None, omega)

            results = run_simulation(
                cfg, oscillators, ref_gen, controller, sync_controller,
                x.clone(), v.clone(), phi.clone(), omega,
            )

            ds = PhaseEstimatorDataset(results, omega_scalar)
            datasets.append(ds)

    return ConcatDataset(datasets)


def build_temporal_pae_dataset(
    cfg,
    num_ic=50,
    omegas_per_ic=4,
    x_range=3.0,
    v_range=30.0,
    omega_range=(torch.pi, 3 * torch.pi),
):
    """Same rollout strategy as build_phase_estimator_dataset but returns
    consecutive pairs for temporal PAE training."""
    datasets = []
    omega_lo, omega_hi = omega_range

    for ic_seed in tqdm(range(num_ic), desc="Generating temporal PAE rollouts"):
        x, v, phi = _sample_ic(cfg, ic_seed, x_range=x_range, v_range=v_range)

        torch.manual_seed(ic_seed + 10_000)
        omega_values = torch.empty(omegas_per_ic).uniform_(omega_lo, omega_hi)

        for omega_scalar in omega_values.tolist():
            omega = torch.ones(cfg.N) * omega_scalar
            oscillators, ref_gen, controller, sync_controller, omega, _ = \
                initialize_modules(cfg, "sinusoidal", None, omega)

            results = run_simulation(
                cfg, oscillators, ref_gen, controller, sync_controller,
                x.clone(), v.clone(), phi.clone(), omega,
            )

            ds = TemporalPAEDataset(results, omega_scalar, cfg.dt)
            datasets.append(ds)

    return ConcatDataset(datasets)


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = get_config()

    num_ic        = 50
    omegas_per_ic = 4
    omega_range   = (torch.pi, 3 * torch.pi)

    ds = build_phase_estimator_dataset(
        cfg,
        num_ic=num_ic,
        omegas_per_ic=omegas_per_ic,
        x_range=3.0,
        v_range=30.0,
        omega_range=omega_range,
    )

    save_dataset_to_pt(
        ds,
        "data/omega_random/dataset_phase_estimator.pt",
        metadata={
            "T": cfg.T,
            "dt": cfg.dt,
            "N": cfg.N,
            "A": cfg.A,
            "condition_mode": "phase_estimator",
            "input_dim": 3,
            "output_dim": 2,
            "num_ic": num_ic,
            "omegas_per_ic": omegas_per_ic,
            "num_rollouts": num_ic * omegas_per_ic,
            "x_range": 3.0,
            "v_range": 30.0,
            "omega_range": list(omega_range),
        },
    )
