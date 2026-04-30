import torch

from src.models.oscillator import Oscillator
from src.models.reference_generator import SinusoidalReference
from src.controllers.pd_controller import PDController
from src.controllers.synchronization_controller import SynchronizationController
from src.models.mlp import MLPReferenceGenerator, load_model
from src.models.diffusion import DiffusionReferenceGenerator, load_diffusion_model
from src.models.phase_estimator import PhaseEstimator, load_phase_estimator
from src.models.phase_autoencoder import PAEPhaseEstimator, load_pae


def initialize_states(cfg, seed=0):
    torch.manual_seed(seed)

    x = 2 * torch.rand(cfg.N) - 1
    v = 20 * torch.rand(cfg.N) - 10
    phi = torch.rand(cfg.N) * 2 * torch.pi
    omega = torch.ones(cfg.N) * (2.0 * torch.pi)

    # training data generation: randomize frequencies a bit more to get more diverse data
    # omega_scalar = torch.empty(1).uniform_(1.5 * torch.pi, 2.5 * torch.pi).item()
    # omega = torch.ones(cfg.N) * omega_scalar

    return x, v, phi, omega


def initialize_modules(cfg, model, model_path, omega, phase_estimator_path=None):
    oscillators = [
        Oscillator(m=cfg.m, d=cfg.d, k=cfg.k)
        for _ in range(cfg.N)
    ]

    reference_generator = build_reference_generator(model, model_path, cfg)

    controller = PDController(
        kp=cfg.kp,
        kd=cfg.kd,
        d=cfg.d,
        k=cfg.k,
    )

    sync_controller = SynchronizationController(k_ps=cfg.k_sync)

    phase_estimator = None
    if phase_estimator_path is not None:
        phase_estimator = build_phase_estimator(phase_estimator_path, cfg)

    return oscillators, reference_generator, controller, sync_controller, omega, phase_estimator


def build_phase_estimator(model_path, cfg):
    """Load either an MLP PhaseEstimator or a PAE, detected from the checkpoint."""
    checkpoint = torch.load(model_path, map_location=cfg.device)

    if "hidden_dim" in checkpoint and "num_layers" in checkpoint \
            and "model_state_dict" in checkpoint:
        # Distinguish PAE (no input_dim key) from MLP (has input_dim key)
        if "input_dim" not in checkpoint:
            # PAE checkpoint
            model, checkpoint = load_pae(model_path, device=cfg.device)
            return PAEPhaseEstimator(
                model=model,
                device=cfg.device,
                norm_stats=checkpoint.get("norm_stats", None),
            )

    # MLP PhaseEstimator
    model, checkpoint = load_phase_estimator(model_path, device=cfg.device)
    return PhaseEstimator(
        model=model,
        device=cfg.device,
        norm_stats=checkpoint.get("norm_stats", None),
    )


def build_reference_generator(model, model_path, cfg=None):
    if model == "sinusoidal":
        return SinusoidalReference(A=cfg.A)

    elif model == "mlp":
        model, checkpoint = load_model(model_path, device=cfg.device)

        horizon = checkpoint["metadata"]["horizon"]

        return MLPReferenceGenerator(
            model=model,
            condition_mode=checkpoint["metadata"]["condition_mode"],
            predict_mode=checkpoint["metadata"]["predict_velocity"],
            horizon=horizon,
            dt=cfg.dt,
            device=cfg.device,
            norm_stats=checkpoint.get("norm_stats", None),
        )

    elif model == "diffusion":
        model_obj, checkpoint = load_diffusion_model(model_path, device=cfg.device)

        horizon = checkpoint["metadata"]["horizon"]

        return DiffusionReferenceGenerator(
            model=model_obj,
            condition_mode=checkpoint["metadata"]["condition_mode"],
            predict_mode=checkpoint["metadata"]["predict_velocity"],
            horizon=horizon,
            dt=cfg.dt,
            device=cfg.device,
            norm_stats=checkpoint.get("norm_stats", None),
            num_samples=1,  # single sample avoids phase cancellation from averaging
        )

    else:
        raise ValueError(f"Unknown reference_type: {cfg.reference_type}")
    
    
def run_simulation(cfg, oscillators, reference_generator,
                   controller, sync_controller,
                   x, v, phi, omega,
                   phase_estimator=None):
    """
    Run the closed-loop simulation.

    If phase_estimator is None, the true integrated phi is passed to the
    sync controller (original behavior, used for data generation).

    If phase_estimator is provided, phi is estimated from (x, v, omega) at
    every step and the sync controller uses estimated sin/cos phases — this
    is the full learned pipeline for inference.
    """
    x_hist, v_hist, phi_hist = [], [], []
    omega_tilde_hist, xref_hist = [], []
    err_hist, u_hist = [], []

    for _ in range(cfg.T):
        x_next   = torch.zeros_like(x)
        v_next   = torch.zeros_like(v)
        phi_next = torch.zeros_like(phi)

        x_ref_all       = torch.zeros_like(x)
        err_all         = torch.zeros_like(x)
        u_all           = torch.zeros_like(x)
        omega_tilde_all = torch.zeros_like(x)

        # --- estimate phases for all oscillators first (needed by sync controller) ---
        if phase_estimator is not None:
            sin_phi_est = torch.zeros(cfg.N)
            cos_phi_est = torch.zeros(cfg.N)
            for i in range(cfg.N):
                s, c = phase_estimator.estimate(x[i], v[i], omega[i])
                sin_phi_est[i] = s
                cos_phi_est[i] = c

        for i in range(cfg.N):
            if phase_estimator is not None:
                # closed-loop learned pipeline:
                # (x, v) -> phase_estimator -> (sin_phi, cos_phi)
                #        -> sync_controller -> omega_tilde
                #        -> reference_model(x, v, omega_tilde) -> x_ref
                phi_dot_i = sync_controller.corrected_frequency_from_sin_cos(
                    i=i,
                    sin_phi=sin_phi_est,
                    cos_phi=cos_phi_est,
                    omega_i=omega[i],
                )
            else:
                # original pipeline: uses ground-truth integrated phi
                phi_dot_i = sync_controller.corrected_frequency(
                    i=i, phi=phi, omega_i=omega[i]
                )

            x_ref_i, v_ref_i, _, _ = reference_generator.get_reference(
                x=x[i], v=v[i], phi=phi[i], phi_dot=phi_dot_i
            )

            u_i = controller.compute(
                x=x[i], v=v[i],
                x_ref=x_ref_i, v_ref=v_ref_i
            )

            x_i_next, v_i_next = oscillators[i].step(
                x=x[i], v=v[i], u=u_i, dt=cfg.dt
            )

            phi_i_next = torch.remainder(
                phi[i] + cfg.dt * phi_dot_i,
                2.0 * torch.pi
            )

            x_next[i]           = x_i_next
            v_next[i]           = v_i_next
            phi_next[i]         = phi_i_next
            x_ref_all[i]        = x_ref_i
            err_all[i]          = x_ref_i - x[i]
            u_all[i]            = u_i
            omega_tilde_all[i]  = phi_dot_i

        x_hist.append(x.clone())
        v_hist.append(v.clone())
        phi_hist.append(phi.clone())
        omega_tilde_hist.append(omega_tilde_all.clone())
        xref_hist.append(x_ref_all.clone())
        err_hist.append(err_all.clone())
        u_hist.append(u_all.clone())

        x, v, phi = x_next, v_next, phi_next

    return {
        "x":           torch.stack(x_hist),
        "v":           torch.stack(v_hist),
        "phi":         torch.stack(phi_hist),
        "omega_tilde": torch.stack(omega_tilde_hist),
        "x_ref":       torch.stack(xref_hist),
        "err":         torch.stack(err_hist),
        "u":           torch.stack(u_hist),
    }