import torch

from Realistic_Analytics.src.models.oscillator import Oscillator
from Realistic_Analytics.src.models.reference_generator import SinusoidalReference
from Realistic_Analytics.src.controllers.pd_controller import PDController
from Realistic_Analytics.src.controllers.synchronization_controller import SynchronizationController


def initialize_states(cfg, seed=0):
    torch.manual_seed(seed)

    x = 2 * torch.rand(cfg.N) - 1
    v = 20 * torch.rand(cfg.N) - 10
    phi = torch.rand(cfg.N) * 2 * torch.pi
    omega = torch.ones(cfg.N) * (2.0 * torch.pi)

    return x, v, phi, omega


def initialize_modules(cfg, omega):
    oscillators = [
        Oscillator(m=cfg.m, d=cfg.d, k=cfg.k)
        for _ in range(cfg.N)
    ]

    reference_generator = SinusoidalReference(A=cfg.A)

    controller = PDController(
        kp=cfg.kp,
        kd=cfg.kd,
        d=cfg.d,
        k=cfg.k,
    )

    sync_controller = SynchronizationController(k_ps=cfg.k_sync)

    return oscillators, reference_generator, controller, sync_controller, omega


def run_simulation(cfg, oscillators, reference_generator,
                   controller, sync_controller,
                   x, v, phi, omega):

    x_hist, v_hist, phi_hist = [], [], []
    omega_tilde_hist, xref_hist = [], []
    err_hist, u_hist = [], []

    for _ in range(cfg.T):
        x_next = torch.zeros_like(x)
        v_next = torch.zeros_like(v)
        phi_next = torch.zeros_like(phi)

        x_ref_all = torch.zeros_like(x)
        err_all = torch.zeros_like(x)
        u_all = torch.zeros_like(x)
        omega_tilde_all = torch.zeros_like(x)

        for i in range(cfg.N):
            phi_dot_i = sync_controller.corrected_frequency(
                i=i, phi=phi, omega_i=omega[i]
            )

            x_ref_i, v_ref_i = reference_generator.get_reference(
                phi=phi[i], phi_dot=phi_dot_i
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

            x_next[i] = x_i_next
            v_next[i] = v_i_next
            phi_next[i] = phi_i_next

            x_ref_all[i] = x_ref_i
            err_all[i] = x_ref_i - x[i]
            u_all[i] = u_i
            omega_tilde_all[i] = phi_dot_i

        x_hist.append(x.clone())
        v_hist.append(v.clone())
        phi_hist.append(phi.clone())
        omega_tilde_hist.append(omega_tilde_all.clone())
        xref_hist.append(x_ref_all.clone())
        err_hist.append(err_all.clone())
        u_hist.append(u_all.clone())

        x, v, phi = x_next, v_next, phi_next

    return {
        "x": torch.stack(x_hist),
        "v": torch.stack(v_hist),
        "phi": torch.stack(phi_hist),
        "omega_tilde": torch.stack(omega_tilde_hist),
        "x_ref": torch.stack(xref_hist),
        "err": torch.stack(err_hist),
        "u": torch.stack(u_hist),
    }