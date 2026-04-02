import torch
import matplotlib.pyplot as plt


def plot_reference_trajectories(time, x_ref_hist):
    plt.figure(figsize=(10, 5))
    for i in range(x_ref_hist.shape[1]):
        plt.plot(time, x_ref_hist[:, i], label=f"ref {i}")
    plt.xlabel("Time [s]")
    plt.ylabel("Reference position")
    plt.title("Reference trajectories")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_actual_trajectories(time, x_hist):
    plt.figure(figsize=(10, 5))
    for i in range(x_hist.shape[1]):
        plt.plot(time, x_hist[:, i], label=f"x {i}")
    plt.xlabel("Time [s]")
    plt.ylabel("Actual position")
    plt.title("Actual trajectories")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_tracking_error(time, err_hist):
    plt.figure(figsize=(10, 5))
    for i in range(err_hist.shape[1]):
        plt.plot(time, err_hist[:, i], label=f"err {i}")
    plt.xlabel("Time [s]")
    plt.ylabel("Tracking error")
    plt.title("Tracking error for each oscillator")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_overlay_per_oscillator(time, x_hist, x_ref_hist):
    N = x_hist.shape[1]
    for i in range(N):
        plt.figure(figsize=(10, 4))
        plt.plot(time, x_hist[:, i], label=f"x_{i}")
        plt.plot(time, x_ref_hist[:, i], "--", label=f"xref_{i}")
        plt.xlabel("Time [s]")
        plt.ylabel("Position")
        plt.title(f"Oscillator {i}: actual vs reference")
        plt.legend()
        plt.tight_layout()
        plt.show()


def plot_phase_evolution(time, phi_hist):
    plt.figure(figsize=(10, 5))
    for i in range(phi_hist.shape[1]):
        plt.plot(time, phi_hist[:, i], label=f"phi {i}")
    plt.xlabel("Time [s]")
    plt.ylabel("Phase")
    plt.title("Phase evolution")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_phase_error(time, phi_hist, i=0, j=1):
    phase_error = torch.atan2(
        torch.sin(phi_hist[:, i] - phi_hist[:, j]),
        torch.cos(phi_hist[:, i] - phi_hist[:, j])
    )
    plt.figure(figsize=(10, 4))
    plt.plot(time, phase_error)
    plt.xlabel("Time [s]")
    plt.ylabel("Wrapped phase error")
    plt.title(f"Phase error between oscillator {i} and {j}")
    plt.tight_layout()
    plt.show()


def plot_polar_phase(phi_hist, dt):
    T = phi_hist.shape[0]
    time = torch.arange(T) * dt

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, projection="polar")

    for i in range(phi_hist.shape[1]):
        ax.plot(phi_hist[:, i].cpu(), time.cpu(), label=f"osc {i}")

    ax.set_title("Phase evolution (polar)")
    ax.legend(loc="upper right")
    plt.show()