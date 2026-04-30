import torch
import torch.nn as nn
from src.utils.dataset_io import normalize_X, denormalize_Y


class PhaseEstimatorMLP(nn.Module):
    """
    Estimates (sin φ, cos φ) from (x, v, ω).

    The output is not constrained to the unit circle during training — the
    loss drives it there implicitly.  At inference, we optionally normalize
    the output so downstream code always gets a unit vector.
    """

    def __init__(self, hidden_dim=64, num_layers=3):
        super().__init__()
        dims = [3] + [hidden_dim] * (num_layers - 1) + [2]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def compute_loss(self, x, y):
        pred = self.forward(x)
        loss = nn.functional.mse_loss(pred, y)
        return loss, {"mse": loss.item()}


class PhaseEstimator:
    """
    Wraps PhaseEstimatorMLP for use inside the simulation loop.

    Given the current state (x_i, v_i) and natural frequency omega_i,
    returns estimated (sin_phi, cos_phi) for oscillator i.
    """

    def __init__(self, model, device, norm_stats=None):
        self.model      = model
        self.device     = device
        self.norm_stats = norm_stats
        self.model.eval()

    @torch.no_grad()
    def estimate(self, x, v, omega):
        """
        x, v, omega: scalars (torch tensors)
        Returns: (sin_phi_est, cos_phi_est) as scalar tensors
        """
        inp = torch.stack([x, v, omega]).float().to(self.device).unsqueeze(0)  # [1, 3]

        if self.norm_stats is not None:
            inp = normalize_X(inp, self.norm_stats)

        out = self.model(inp).squeeze(0)  # [2]

        if self.norm_stats is not None:
            out = denormalize_Y(out, self.norm_stats)

        # normalize to unit circle so downstream sin(phi_i - phi_j) is correct
        norm = out.norm().clamp(min=1e-6)
        out  = out / norm

        return out[0], out[1]   # sin_phi, cos_phi


def load_phase_estimator(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)

    model = PhaseEstimatorMLP(
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=checkpoint["num_layers"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if "norm_stats" in checkpoint:
        checkpoint["norm_stats"] = {
            k: v.to(device) for k, v in checkpoint["norm_stats"].items()
        }

    return model, checkpoint
