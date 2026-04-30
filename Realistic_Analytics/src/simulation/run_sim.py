import torch
from src.config import get_config
from src.simulation.sim_helper import initialize_states, initialize_modules, run_simulation
from src.utils.plotting import (
    plot_reference_trajectories,
    plot_actual_trajectories,
    plot_tracking_error,
    plot_overlay_all_oscillator,
    plot_phase_evolution,
    plot_phase_error,
    plot_polar_phase,
)


def plot_results(results, dt, folder="graphs"):
    time = torch.arange(results["x"].shape[0]) * dt
    save_path = folder

    plot_reference_trajectories(time, results["x_ref"], save_path=f"{save_path}")
    plot_actual_trajectories(time, results["x"], save_path=f"{save_path}")
    plot_tracking_error(time, results["err"], save_path=f"{save_path}")
    # plot_overlay_per_oscillator(time, results["x"], results["x_ref"])
    plot_overlay_all_oscillator(time, results["x"], results["x_ref"], save_path=f"{save_path}")
    plot_phase_evolution(time, results["phi"], save_path=f"{save_path}")
    plot_phase_error(time, results["phi"], i=0, j=1, save_path=f"{save_path}")
    plot_polar_phase(results["phi"], dt, save_path=f"{save_path}")


def main():
    cfg = get_config()

    x, v, phi, omega = initialize_states(cfg)

    # mode: "sinusoidal", "mlp", or "diffusion"
    model_mode = "diffusion"
    model_path = f"models/{model_mode}_state_phase_trig_freq.pt"

    oscillators, ref_gen, ctrl, sync_ctrl, omega, phase_est = initialize_modules(
        cfg, model_mode, model_path, omega,
        phase_estimator_path="models/phase_estimator.pt",
    )
    results = run_simulation(cfg, oscillators, ref_gen, ctrl, sync_ctrl,
                             x, v, phi, omega, phase_estimator=phase_est)

    plot_results(results, cfg.dt, folder=f"graphs/{model_mode}")


if __name__ == "__main__":
    main()