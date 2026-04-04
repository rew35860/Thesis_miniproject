# src/config.py
from dataclasses import dataclass

@dataclass
class Config:
    device: str = "cpu"
    N: int = 3
    T: int = 2000
    dt: float = 0.005

    A: float = 1.0
    m: float = 1.0
    d: float = 0.4
    k: float = 4.0

    kp: float = 40.0
    kd: float = 30.0
    k_sync: float = 0.15


@dataclass
class MLPConfig:
    batch_size: int = 128
    hidden_dim: int = 128
    num_layers: int = 3
    lr: float = 1e-3
    epochs: int = 100
    weight_decay: float = 1e-6


def get_config():
    return Config()

def get_mlp_config():
    return MLPConfig()