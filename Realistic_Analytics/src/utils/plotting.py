import torch
import matplotlib.pyplot as plt


def plot_reference_trajectories(time, x_ref_hist, save_path="graphs"):
    plt.figure(figsize=(10, 5))
    for i in range(x_ref_hist.shape[1]):
        plt.plot(time, x_ref_hist[:, i], label=f"ref {i}")
    plt.xlabel("Time [s]")
    plt.ylabel("Reference position")
    plt.title("Reference trajectories")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_path}/reference_trajectories.png")


def plot_actual_trajectories(time, x_hist, save_path="graphs"):
    plt.figure(figsize=(10, 5))
    for i in range(x_hist.shape[1]):
        plt.plot(time, x_hist[:, i], label=f"x {i}")
    plt.xlabel("Time [s]")
    plt.ylabel("Actual position")
    plt.title("Actual trajectories")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_path}/actual_trajectories.png")


def plot_tracking_error(time, err_hist, save_path="graphs"):
    plt.figure(figsize=(10, 5))
    for i in range(err_hist.shape[1]):
        plt.plot(time, err_hist[:, i], label=f"err {i}")
    plt.xlabel("Time [s]")
    plt.ylabel("Tracking error")
    plt.title("Tracking error for each oscillator")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_path}/tracking_error.png")


def plot_overlay_per_oscillator(time, x_hist, x_ref_hist, save_path="graphs"):
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
        plt.savefig(f"{save_path}/overlay_oscillator_{i}.png")


def plot_overlay_all_oscillator(time, x_hist, x_ref_hist, save_path="graphs"):
    plt.figure(figsize=(10, 4))
    N = x_hist.shape[1]
    for i in range(N):
        plt.plot(time, x_hist[:, i], label=f"x_{i}")
        plt.plot(time, x_ref_hist[:, i], "--", label=f"xref_{i}")
    plt.xlabel("Time [s]")
    plt.ylabel("Position")
    plt.title(f"{N} Oscillators: actual vs reference")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_path}/overlay_all_oscillators.png")


def plot_phase_evolution(time, phi_hist, save_path="graphs"):
    plt.figure(figsize=(10, 5))
    for i in range(phi_hist.shape[1]):
        plt.plot(time, phi_hist[:, i], label=f"phi {i}")
    plt.xlabel("Time [s]")
    plt.ylabel("Phase")
    plt.title("Phase evolution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_path}/phase_evolution.png")


def plot_phase_error(time, phi_hist, i=0, j=1, save_path="graphs"):
    N = phi_hist.shape[1]
    phase_error = torch.zeros((phi_hist.shape[0], N-1))
    for j in range(1, N):
        phase_error[:, j-1] = torch.atan2(
            torch.sin(phi_hist[:, i] - phi_hist[:, j]),
            torch.cos(phi_hist[:, i] - phi_hist[:, j])
        )

    plt.figure(figsize=(10, 4))
    plt.plot(time, phase_error)
    plt.xlabel("Time [s]")
    plt.ylabel("Wrapped phase error")
    plt.title(f"Phase error between oscillator {i} and all others {N-1} oscillators")
    plt.tight_layout()
    plt.savefig(f"{save_path}/phase_error.png")


def plot_polar_phase(phi_hist, dt, save_path="graphs"):
    T = phi_hist.shape[0]
    time = torch.arange(T) * dt

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, projection="polar")

    for i in range(phi_hist.shape[1]):
        ax.plot(phi_hist[:, i].cpu(), time.cpu(), label=f"osc {i}")

    ax.set_title("Phase evolution (polar)")
    ax.legend(loc="upper right")
    plt.savefig(f"{save_path}/polar_phase.png")


def plot_control_input(time, u_hist, save_path="graphs"):
    plt.figure(figsize=(10, 5))
    for i in range(u_hist.shape[1]):
        plt.plot(time, u_hist[:, i], label=f"u {i}")
    plt.xlabel("Time [s]")
    plt.ylabel("Control input")
    plt.title("Control input for each oscillator")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_path}/control_input.png")


# Plotting For Training
def plot_losses(train_losses, val_losses, model_mode, save_path="graphs"):
    plt.figure(figsize=(7, 4))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title(f"{model_mode.capitalize()} training loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_path}_{model_mode}_training_loss.png")


def plot_full_trajectory(
    model,
    X,
    Y,
    metadata,
    device,
    model_mode="mlp",
    save_path="graphs",
    rollout_idx=0,
    osc_idx=0,
):
    model.eval()

    N = metadata["N"]
    T = metadata["T"]
    horizon = metadata["horizon"]
    dt = metadata["dt"]

    samples_per_osc = T - horizon - 1
    start_idx = rollout_idx * N * samples_per_osc + osc_idx * samples_per_osc

    preds = []
    trues = []

    with torch.no_grad():
        for local_t in range(samples_per_osc):
            idx = start_idx + local_t

            cond = X[idx].unsqueeze(0).to(device)
            y_true = Y[idx].cpu()

            if model_mode == "mlp":
                y_pred = model(cond).squeeze(0).cpu()
            elif model_mode == "diffusion":
                y_pred = model.sample(cond).squeeze(0).cpu()
            else:
                raise ValueError(f"Unknown model_mode: {model_mode}")

            # first predicted future position x_{t+1}
            trues.append(y_true[0].item())
            preds.append(y_pred[0].item())

    preds = torch.tensor(preds)
    trues = torch.tensor(trues)
    t_axis = torch.arange(samples_per_osc, dtype=torch.float32) * dt

    plt.figure(figsize=(10, 4))
    plt.plot(t_axis.numpy(), trues.numpy(), label="ground truth")
    plt.plot(t_axis.numpy(), preds.numpy(), label="prediction")
    plt.xlabel("Time (s)")
    plt.ylabel("Position")
    plt.title(f"{model_mode.upper()} full trajectory | rollout={rollout_idx}, osc={osc_idx}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_path}{model_mode}_full_trajectory_{rollout_idx}_{osc_idx}.png")
    plt.close()


def plot_dataset_samples(Y, metadata, start_idx=0, num_points=2000, save_path="graphs"):
    Y = Y.cpu()

    segment = Y[start_idx:start_idx + num_points]
    values = segment[:, 0] if segment.ndim > 1 else segment
    dt = metadata["dt"]

    t = torch.arange(len(values)) * dt

    plt.figure(figsize=(8, 4))
    plt.plot(t.numpy(), values.numpy())

    plt.title("Reconstructed trajectory (time-corrected)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("x")
    plt.tight_layout()
    plt.savefig(f"{save_path}_dataset_samples.png")


def plot_predictions(
    model,
    X,
    Y,
    device,
    model_mode="mlp",
    predict_velocity=False,
    horizon=20,
    dt=0.005,
    idx=0,
    save_path="graphs",
):
    model.eval()

    with torch.no_grad():
        cond = X[idx].unsqueeze(0).to(device)
        y_true = Y[idx].cpu().numpy()

        if model_mode == "mlp":
            y_pred = model(cond).squeeze(0).cpu().numpy()
        elif model_mode == "diffusion":
            y_pred = model.sample(cond).squeeze(0).cpu().numpy()
        else:
            raise ValueError(f"Unknown model_mode: {model_mode}")

    # === Always compute x ===
    x_true = y_true[:horizon]
    x_pred = y_pred[:horizon]

    if predict_velocity:
        v_true = y_true[horizon:]
        v_pred = y_pred[horizon:]

    t_future = torch.arange(len(x_true)) * dt

    # === Always plot x ===
    plt.figure(figsize=(8, 4))
    plt.plot(t_future.numpy(), x_true, label="x true")
    plt.plot(t_future.numpy(), x_pred, label="x pred")
    plt.xlabel("Time (s)")
    plt.ylabel("Position")
    plt.title(f"{model_mode.upper()} position prediction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_path}{model_mode}_x_prediction_window_{idx}.png")
    plt.close()

    # === Only plot v if available ===
    if predict_velocity:
        v_true = y_true[horizon:]
        v_pred = y_pred[horizon:]

        t_future = torch.arange(horizon) * dt

        plt.figure(figsize=(8, 4))
        plt.plot(t_future.numpy(), v_true, label="v true")
        plt.plot(t_future.numpy(), v_pred, label="v pred")
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity")
        plt.title(f"{model_mode.upper()} velocity prediction")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_path}{model_mode}_v_prediction_window_{idx}.png")
        plt.close()