from dataclasses import dataclass


@dataclass
class Config:
    # ── simulation ─────────────────────────────────────────────────────────────
    device: str  = "cpu"
    N: int       = 3        # number of oscillators
    T: int       = 2000     # simulation steps
    dt: float    = 0.005    # timestep (s)

    # ── oscillator physics ─────────────────────────────────────────────────────
    A: float     = 1.0      # reference amplitude
    m: float     = 1.0      # mass (kg)
    d: float     = 0.4      # damping coefficient
    k: float     = 4.0      # spring constant → natural freq ω_n = √(k/m) ≈ 2 rad/s

    # ── PD controller ──────────────────────────────────────────────────────────
    kp: float    = 40.0     # proportional gain
    kd: float    = 30.0     # derivative gain

    # ── synchronization controller ─────────────────────────────────────────────
    k_sync: float = 0.22    # Kuramoto coupling strength


@dataclass
class MLPConfig:
    batch_size:   int   = 128
    hidden_dim:   int   = 128
    num_layers:   int   = 3
    lr:           float = 1e-3
    epochs:       int   = 100
    weight_decay: float = 1e-6


@dataclass
class DiffusionConfig:
    # ── shared training params ─────────────────────────────────────────────────
    batch_size:   int   = 128
    hidden_dim:   int   = 256
    num_layers:   int   = 4
    lr:           float = 1e-3
    epochs:       int   = 100
    weight_decay: float = 1e-6

    # ── diffusion-specific ─────────────────────────────────────────────────────
    num_diffusion_steps: int = 100  # DDPM denoising steps T
    time_dim:            int = 64   # sinusoidal time-embedding dimension


@dataclass
class PhaseEstimatorConfig:
    batch_size:   int   = 256
    hidden_dim:   int   = 64
    num_layers:   int   = 3
    lr:           float = 1e-3
    epochs:       int   = 100
    weight_decay: float = 1e-6


def get_config():
    return Config()

def get_phase_estimator_config():
    return PhaseEstimatorConfig()

def get_mlp_config():
    return MLPConfig()

def get_diffusion_config():
    return DiffusionConfig()
