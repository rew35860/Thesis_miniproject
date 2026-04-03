import torch
import matplotlib
matplotlib.use("TkAgg")

from src.oscillator import Oscillator
from src.reference_generator import SinusoidalReference
from src.controllers.pd_controller import PDController
from src.controllers.synchronization_controller import SynchronizationController
from src.utils.plotting import (
    plot_reference_trajectories,
    plot_actual_trajectories,
    plot_tracking_error,
    plot_overlay_per_oscillator,
    plot_phase_evolution,
    plot_phase_error,
    plot_polar_phase,
)


def get_config():
    return {
        "device": "cpu",
        "N": 3,
        "T": 2000,
        "dt": 0.005,
        "A": 1.0,
        "m": 1.0,
        "d": 0.4,
        "k": 4.0,
        "kp": 40.0,
        "kd": 30.0,
        "k_sync": 0.15,
    }


def initialize_states(N, device, seed=0):
    torch.manual_seed(seed)

    # random initial positions, velocities (-1, 1)
    x = 2 * torch.rand(N, device=device) - 1
    v = 2 * torch.rand(N, device=device) - 1

    # random phases in [0, 2π)
    phi = torch.rand(N, device=device) * 2 * torch.pi

    # nominal frequencies (same for all)
    omega = torch.ones(N, device=device) * (2.0 * torch.pi)

    return x, v, phi, omega


def initialize_modules(cfg, omega):
    oscillators = [
        Oscillator(m=cfg["m"], d=cfg["d"], k=cfg["k"])
        for _ in range(cfg["N"])
    ]

    reference_generator = SinusoidalReference(A=cfg["A"])
    controller = PDController(
        kp=cfg["kp"],
        kd=cfg["kd"],
        d=cfg["d"],
        k=cfg["k"],
    )
    sync_controller = SynchronizationController(k_ps=cfg["k_sync"])

    return oscillators, reference_generator, controller, sync_controller, omega


def run_simulation(cfg, oscillators, reference_generator, controller, sync_controller, x, v, phi, omega):
    x_hist = []
    v_hist = []
    phi_hist = []
    xref_hist = []
    err_hist = []
    u_hist = []

    for _ in range(cfg["T"]):
        x_next = torch.zeros_like(x)
        v_next = torch.zeros_like(v)
        phi_next = torch.zeros_like(phi)

        x_ref_all = torch.zeros_like(x)
        err_all = torch.zeros_like(x)
        u_all = torch.zeros_like(x)

        for i in range(cfg["N"]):
            phi_dot_i = sync_controller.corrected_frequency(
                i=i,
                phi=phi,
                omega_i=omega[i]
            )

            x_ref_i, v_ref_i = reference_generator.get_reference(
                phi=phi[i],
                phi_dot=phi_dot_i
            )

            u_i = controller.compute(
                x=x[i],
                v=v[i],
                x_ref=x_ref_i,
                v_ref=v_ref_i
            )

            x_i_next, v_i_next = oscillators[i].step(
                x=x[i],
                v=v[i],
                u=u_i,
                dt=cfg["dt"]
            )

            phi_i_next = phi[i] + cfg["dt"] * phi_dot_i
            phi_i_next = torch.remainder(phi_i_next, 2.0 * torch.pi)

            x_next[i] = x_i_next
            v_next[i] = v_i_next
            phi_next[i] = phi_i_next

            x_ref_all[i] = x_ref_i
            err_all[i] = x_ref_i - x[i]
            u_all[i] = u_i

        x_hist.append(x.clone())
        v_hist.append(v.clone())
        phi_hist.append(phi.clone())
        xref_hist.append(x_ref_all.clone())
        err_hist.append(err_all.clone())
        u_hist.append(u_all.clone())

        x = x_next
        v = v_next
        phi = phi_next

    results = {
        "x": torch.stack(x_hist),
        "v": torch.stack(v_hist),
        "phi": torch.stack(phi_hist),
        "x_ref": torch.stack(xref_hist),
        "err": torch.stack(err_hist),
        "u": torch.stack(u_hist),
    }
    return results


def plot_results(results, dt):
    time = torch.arange(results["x"].shape[0]) * dt

    # plot_reference_trajectories(time, results["x_ref"])
    # plot_actual_trajectories(time, results["x"])
    # plot_tracking_error(time, results["err"])
    plot_overlay_per_oscillator(time, results["x"], results["x_ref"])
    # plot_phase_evolution(time, results["phi"])
    # plot_phase_error(time, results["phi"], i=0, j=1)
    plot_polar_phase(results["phi"], dt)


def main():
    cfg = get_config()

    x, v, phi, omega = initialize_states(cfg["N"], cfg["device"])

    oscillators, reference_generator, controller, sync_controller, omega = initialize_modules(cfg, omega)

    results = run_simulation(
        cfg,
        oscillators,
        reference_generator,
        controller,
        sync_controller,
        x,
        v,
        phi,
        omega,
    )

    plot_results(results, cfg["dt"])


if __name__ == "__main__":
    main()