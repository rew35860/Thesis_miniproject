import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.model_io import build_condition, decode_prediction
from src.utils.dataset_io import normalize_X, denormalize_Y


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2

        if half_dim == 0:
            return torch.zeros(t.shape[0], self.dim, device=device)

        scale = math.log(10000) / max(half_dim - 1, 1)
        freqs = torch.exp(torch.arange(half_dim, device=device) * -scale)
        angles = t.float().unsqueeze(1) * freqs.unsqueeze(0)

        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)

        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return emb


class DiffusionDenoiser(nn.Module):
    def __init__(
        self,
        cond_dim: int,
        target_dim: int,
        hidden_dim: int = 256,
        time_dim: int = 64,
        num_layers: int = 4,
    ):
        super().__init__()

        self.time_embed = SinusoidalTimeEmbedding(time_dim)

        input_dim = cond_dim + target_dim + time_dim
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [target_dim]

        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        y_noisy: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        t_emb = self.time_embed(t)
        x = torch.cat([y_noisy, cond, t_emb], dim=-1)
        return self.net(x)


class ConditionalDDPM(nn.Module):
    def __init__(
        self,
        cond_dim: int,
        target_dim: int,
        num_diffusion_steps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        hidden_dim: int = 256,
        time_dim: int = 64,
        num_layers: int = 4,
    ):
        super().__init__()

        self.cond_dim = cond_dim
        self.target_dim = target_dim
        self.num_diffusion_steps = num_diffusion_steps

        self.denoiser = DiffusionDenoiser(
            cond_dim=cond_dim,
            target_dim=target_dim,
            hidden_dim=hidden_dim,
            time_dim=time_dim,
            num_layers=num_layers,
        )

        betas = torch.linspace(beta_start, beta_end, num_diffusion_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer(
            "sqrt_one_minus_alpha_bars",
            torch.sqrt(1.0 - alpha_bars),
        )

    def q_sample(self, y0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_ab = self.sqrt_alpha_bars[t].unsqueeze(1)
        sqrt_1mab = self.sqrt_one_minus_alpha_bars[t].unsqueeze(1)
        return sqrt_ab * y0 + sqrt_1mab * noise

    def compute_loss(self, cond: torch.Tensor, y0: torch.Tensor) -> torch.Tensor:
        batch_size = y0.shape[0]
        device = y0.device

        t = torch.randint(
            low=0,
            high=self.num_diffusion_steps,
            size=(batch_size,),
            device=device,
        )

        noise = torch.randn_like(y0)
        y_noisy = self.q_sample(y0, t, noise)
        noise_pred = self.denoiser(y_noisy, t, cond)

        loss = F.mse_loss(noise_pred, noise)
        return loss, {"mse": loss.item()}

    @torch.no_grad()
    def sample(self, cond: torch.Tensor) -> torch.Tensor:
        """Stochastic DDPM sampler (full T steps)."""
        device = cond.device
        batch_size = cond.shape[0]

        y = torch.randn(batch_size, self.target_dim, device=device)

        for step in reversed(range(self.num_diffusion_steps)):
            t = torch.full((batch_size,), step, device=device, dtype=torch.long)

            beta_t = self.betas[step]
            alpha_t = self.alphas[step]
            alpha_bar_t = self.alpha_bars[step]

            noise_pred = self.denoiser(y, t, cond)

            coef1 = 1.0 / torch.sqrt(alpha_t)
            coef2 = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)

            z = torch.randn_like(y) if step > 0 else torch.zeros_like(y)
            y = coef1 * (y - coef2 * noise_pred) + torch.sqrt(beta_t) * z

        return y

    @torch.no_grad()
    def ddim_sample(self, cond: torch.Tensor, num_steps: int = 20) -> torch.Tensor:
        """Deterministic DDIM sampler.

        Uses a sub-sequence of `num_steps` timesteps instead of the full T.
        σ=0 → completely deterministic: same cond always gives the same output.
        Much faster than DDPM and eliminates stochastic jitter in the reference.

        Args:
            cond:      condition tensor  [B, cond_dim]
            num_steps: number of denoising steps (10-20 is usually sufficient)
        """
        device = cond.device
        batch_size = cond.shape[0]

        # uniform sub-sequence of timesteps τ: T-1, ..., 0
        step_indices = torch.linspace(
            self.num_diffusion_steps - 1, 0, num_steps, dtype=torch.long
        )

        y = torch.randn(batch_size, self.target_dim, device=device)

        for i, step in enumerate(step_indices):
            step = step.item()
            t = torch.full((batch_size,), step, device=device, dtype=torch.long)

            alpha_bar_t = self.alpha_bars[step]
            noise_pred = self.denoiser(y, t, cond)

            # predict x_0 from x_t and ε_θ
            x0_pred = (y - torch.sqrt(1.0 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)

            if i < num_steps - 1:
                step_prev = step_indices[i + 1].item()
                alpha_bar_prev = self.alpha_bars[step_prev]
            else:
                alpha_bar_prev = torch.ones(1, device=device)  # ᾱ_0 ≈ 1 → y = x0

            # DDIM update (σ=0): interpolate toward x0 along the direction of ε_θ
            y = torch.sqrt(alpha_bar_prev) * x0_pred + torch.sqrt(1.0 - alpha_bar_prev) * noise_pred

        return y
    

class DiffusionReferenceGenerator:
    def __init__(self, model, condition_mode, predict_mode, horizon, dt, device,
                 norm_stats=None, num_samples=5):
        """Wraps ConditionalDDPM with the same interface as MLPReferenceGenerator.

        condition_mode: "state_freq", "phase_freq", etc.
        predict_mode:   True if model predicts both x and v, False for x only.
        norm_stats:     dict with mean_X, std_X, mean_Y, std_Y from training.
        num_samples:    number of DDPM samples to average per call (default 5).
                        Averaging approximates E[y|condition], which is stable
                        and close to ground truth even though each sample is noisy.
        """
        self.model = model
        self.condition_mode = condition_mode
        self.predict_mode = predict_mode
        self.horizon = horizon
        self.dt = dt
        self.device = device
        self.norm_stats = norm_stats
        self.num_samples = num_samples

        self.model.eval()

    def build_input(self, x, v, phi, phi_dot):
        inp = build_condition(
            x=x, v=v, phi=phi, phi_dot=phi_dot,
            mode=self.condition_mode,
            device=self.device,
        )
        if self.norm_stats is not None:
            inp = normalize_X(inp, self.norm_stats)
        return inp

    @torch.no_grad()
    def predict_future(self, x, v, phi, phi_dot):
        inp = self.build_input(x, v, phi, phi_dot)
        # repeat condition along batch dim, sample in one forward pass, then average
        inp_rep = inp.expand(self.num_samples, -1)
        pred = self.model.sample(inp_rep).mean(dim=0, keepdim=True)
        if self.norm_stats is not None:
            pred = denormalize_Y(pred, self.norm_stats)
        return pred.squeeze(0)

    @torch.no_grad()
    def get_reference(self, x, v, phi, phi_dot):
        pred = self.predict_future(x, v, phi, phi_dot)

        x_ref, v_ref, x_pred, v_pred = decode_prediction(
            pred=pred,
            predict_mode=self.predict_mode,
            horizon=self.horizon,
            dt=self.dt,
            device=self.device,
        )

        return x_ref, v_ref, x_pred, v_pred


def load_diffusion_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)

    model = ConditionalDDPM(
        cond_dim=checkpoint["input_dim"],
        target_dim=checkpoint["output_dim"],
        num_diffusion_steps=checkpoint["num_diffusion_steps"],
        hidden_dim=checkpoint["hidden_dim"],
        time_dim=checkpoint["time_dim"],
        num_layers=checkpoint["num_layers"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if "norm_stats" in checkpoint:
        checkpoint["norm_stats"] = {
            k: v.to(device) for k, v in checkpoint["norm_stats"].items()
        }

    return model, checkpoint