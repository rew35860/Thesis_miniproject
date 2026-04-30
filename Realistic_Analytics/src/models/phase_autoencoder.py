"""
Periodic Autoencoder (PAE) for phase estimation.

Inspired by: "DeepPhase: Periodic Autoencoders for Learning Motion Phase Manifolds"

Architecture
------------
Encoder : (x, v, ω) → z ∈ ℝ²   (unconstrained latent)
Latent  : A = ||z||₂,  (sin φ, cos φ) = z / A   ← circular projection
Decoder : (z, ω) → (x̂, v̂)                       ← reconstruction

The key insight from DeepPhase: by projecting the latent onto the unit circle
and using the unnormalized magnitude as amplitude, the model is *forced* to
represent its latent as a phase angle — no supervision needed. Periodicity is
baked into the architecture, not the loss.

For the oscillator problem this is especially natural because
(x, v/ω) already lives on a circle of radius A in state space.

Usage
-----
At inference:
    sin_phi, cos_phi = pae.get_phase(x, v, omega)

These drop directly into the sync controller's corrected_frequency_from_sin_cos.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.dataset_io import normalize_X, denormalize_Y


def _mlp(in_dim, out_dim, hidden_dim, num_layers):
    dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
    layers = []
    for i in range(len(dims) - 2):
        layers += [nn.Linear(dims[i], dims[i + 1]), nn.ELU()]
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


class PeriodicAutoencoder(nn.Module):
    """
    PAE with a 2-D circular latent space.

    The latent z ∈ ℝ² is interpreted as:
        amplitude A  = ||z||₂
        phase vector = z / A  =  (sin φ, cos φ)

    Encoder and decoder both receive ω so the model can handle
    variable-frequency oscillators without conflating amplitude with frequency.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 3,
    ):
        super().__init__()
        # input: [x, v, ω]  → latent z ∈ ℝ²
        self.encoder = _mlp(3, 2, hidden_dim, num_layers)
        # input: [z₁, z₂, ω] → reconstructed [x, v]
        self.decoder = _mlp(3, 2, hidden_dim, num_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 3] = [x, v, ω]  →  z: [B, 2]"""
        return self.encoder(x)

    def decode(self, z: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        """z: [B, 2], omega: [B, 1]  →  x_hat: [B, 2] = [x̂, v̂]"""
        inp = torch.cat([z, omega], dim=-1)
        return self.decoder(inp)

    def forward(self, inp: torch.Tensor):
        """
        inp: [B, 3] = [x, v, ω]

        Returns:
            x_hat  : [B, 2]  reconstructed [x, v]
            z      : [B, 2]  latent (amplitude × phase vector)
            sin_phi: [B]     unit-circle phase component
            cos_phi: [B]     unit-circle phase component
            amp    : [B]     learned amplitude ||z||
        """
        z = self.encode(inp)

        amp = z.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # [B, 1]
        phase_vec = z / amp                                   # [B, 2]  on unit circle
        sin_phi = phase_vec[:, 0]
        cos_phi = phase_vec[:, 1]

        omega = inp[:, 2:3]
        x_hat = self.decode(z, omega)

        return x_hat, z, sin_phi, cos_phi, amp.squeeze(-1)

    def compute_loss(self, inp: torch.Tensor):
        """
        Self-supervised reconstruction loss — no phase labels required.

        inp: [B, 3] = [x, v, ω]  (normalised)
        target for reconstruction: inp[:, :2] = [x, v]
        """
        x_hat, z, sin_phi, cos_phi, amp = self.forward(inp)

        target = inp[:, :2]
        recon_loss = F.mse_loss(x_hat, target)

        return recon_loss, {"recon": recon_loss.item()}

    def compute_temporal_loss(
        self,
        inp_t: torch.Tensor,
        inp_next: torch.Tensor,
        omega_raw: torch.Tensor,
        dt: float,
        temporal_weight: float = 1.0,
    ):
        """
        Combined reconstruction + temporal consistency loss.

        Physics constraint: phase advances by ω·dt per step, so the latent
        z must rotate by that angle:
            z_{t+1}  =  R(ω·dt) · z_t
        where R(θ) = [[cos θ, -sin θ], [sin θ, cos θ]].

        This pins the phase up to a global offset (irrelevant for sync) and
        forces the encoder to genuinely encode phase, not just amplitude.

        inp_t, inp_next: [B, 3] = [x, v, ω]  consecutive normalised frames
        """
        # Reconstruction losses for both frames
        x_hat_t,    _, _, _, _ = self.forward(inp_t)
        x_hat_next, _, _, _, _ = self.forward(inp_next)
        recon_loss = F.mse_loss(x_hat_t,    inp_t[:, :2]) \
                   + F.mse_loss(x_hat_next, inp_next[:, :2])

        # Temporal consistency: z_next should equal R(ω·dt) · z_t
        z_t    = self.encode(inp_t)       # [B, 2]
        z_next = self.encode(inp_next)    # [B, 2]

        # omega_raw: [B] physical ω values (not normalised) passed by the trainer
        # so that θ = ω·dt is the true rotation angle in radians.
        theta = omega_raw * dt            # [B]

        cos_t = torch.cos(theta)          # [B]
        sin_t = torch.sin(theta)          # [B]

        # R(θ) · z_t  = [cos·z0 - sin·z1,  sin·z0 + cos·z1]
        z_t_rotated = torch.stack([
            cos_t * z_t[:, 0] - sin_t * z_t[:, 1],
            sin_t * z_t[:, 0] + cos_t * z_t[:, 1],
        ], dim=-1)                        # [B, 2]

        temporal_loss = F.mse_loss(z_next, z_t_rotated)

        total = recon_loss + temporal_weight * temporal_loss
        return total, {
            "recon":    recon_loss.item(),
            "temporal": temporal_loss.item(),
            "total":    total.item(),
        }


class PAEPhaseEstimator:
    """
    Thin inference wrapper around PeriodicAutoencoder.

    Mirrors the PhaseEstimator interface so it can be swapped in anywhere
    a PhaseEstimator is used.
    """

    def __init__(self, model: PeriodicAutoencoder, device, norm_stats=None):
        self.model = model
        self.device = device
        self.norm_stats = norm_stats
        self.model.eval()

    @torch.no_grad()
    def estimate(self, x: torch.Tensor, v: torch.Tensor, omega: torch.Tensor):
        """
        Returns (sin_phi, cos_phi) as scalars on the unit circle.

        x, v, omega: scalar tensors (single oscillator, single timestep)
        """
        inp = torch.stack([x, v, omega]).float().to(self.device).unsqueeze(0)

        if self.norm_stats is not None:
            inp = normalize_X(inp, self.norm_stats)

        _, _, sin_phi, cos_phi, _ = self.model(inp)

        return sin_phi.squeeze(), cos_phi.squeeze()


def load_pae(model_path: str, device):
    checkpoint = torch.load(model_path, map_location=device)

    model = PeriodicAutoencoder(
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
