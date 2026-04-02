from oscillator import Oscillator
import matplotlib.pyplot as plt
import numpy as np
import torch
from train_oscillator_mlp_condition import PhaseConditionedMLP


def initialize_oscillators(N, omega=2.0*np.pi):
    np.random.seed(42)  # for reproducibility

    return [
        Oscillator(theta=np.random.uniform(0, 2*np.pi), omega=omega)
        for _ in range(N)
    ]


def load_phase_conditioned_model(model_path, horizon=20, hidden_dim=64, device="cpu"):
    model = PhaseConditionedMLP(input_dim=4, hidden_dim=hidden_dim, horizon=horizon)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_future_motion(model, x_t, theta_t, omega, phase_error, horizon, device="cpu"):
    """
    Input: [x_t, theta_t, omega, phase_error]
    Output: predicted future sequence of shape (horizon, 2)
            each row = [x, theta]
    """
    inp = torch.tensor([[x_t, theta_t, omega, phase_error]], dtype=torch.float32).to(device)

    with torch.no_grad():
        pred = model(inp).cpu().numpy().reshape(horizon, 2)

    return pred


def run_simulation_with_mlp(oscillators, model, steps, dt, K, horizon=20, device="cpu"):
    phase_history = []
    analytical_position_history = []
    mlp_position_history = []
    mlp_theta_history = []
    phase_error_history = []

    for _ in range(steps):
        thetas = np.array([osc.theta for osc in oscillators])
        analytical_positions = np.array([osc.get_position() for osc in oscillators])

        phase_history.append(thetas.copy())
        analytical_position_history.append(analytical_positions.copy())

        # use mean phase as simple target
        target_phase = np.mean(thetas)

        mlp_positions = []
        mlp_thetas = []
        phase_errors = []

        # query MLP for each oscillator
        for osc in oscillators:
            x_t = osc.get_position()
            theta_t = osc.theta
            omega = osc.omega

            phase_error = target_phase - theta_t
            phase_errors.append(phase_error)

            pred_future = predict_future_motion(
                model=model,
                x_t=x_t,
                theta_t=theta_t,
                omega=omega,
                phase_error=phase_error,
                horizon=horizon,
                device=device,
            )

            # use first predicted future step
            pred_x_next = pred_future[0, 0]
            pred_theta_next = pred_future[0, 1]

            mlp_positions.append(pred_x_next)
            mlp_thetas.append(pred_theta_next)

        mlp_position_history.append(np.array(mlp_positions))
        mlp_theta_history.append(np.array(mlp_thetas))
        phase_error_history.append(np.array(phase_errors))

        # keep analytical synchronization update for actual oscillator state
        for osc in oscillators:
            osc.step(thetas, K, dt)

    return (
        np.array(phase_history),
        np.array(analytical_position_history),
        np.array(mlp_position_history),
        np.array(mlp_theta_history),
        np.array(phase_error_history),
    )


def plot_results_with_mlp(time, phase_history, analytical_position_history, mlp_position_history):
    # True phase history from analytical oscillator state
    plt.figure()
    for i in range(phase_history.shape[1]):
        plt.plot(time, phase_history[:, i])
    plt.xlabel("Time")
    plt.ylabel("Phase")
    plt.title("Analytical Phase vs Time")
    plt.show()

    # Analytical positions
    plt.figure()
    for i in range(analytical_position_history.shape[1]):
        plt.plot(time, analytical_position_history[:, i])
    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.title("Analytical Position vs Time")
    plt.show()

    # MLP-predicted first-step positions
    plt.figure()
    for i in range(mlp_position_history.shape[1]):
        plt.plot(time, mlp_position_history[:, i])
    plt.xlabel("Time")
    plt.ylabel("Predicted Position")
    plt.title("MLP-Predicted Position vs Time")
    plt.show()


def run_simulation(oscillators, steps, dt, K):
    phase_history = []
    position_history = []

    for t in range(steps):
        # collect phases
        thetas = np.array([osc.theta for osc in oscillators])

        # store phase history
        phase_history.append(thetas.copy())

        # store position history
        positions = np.array([osc.get_position() for osc in oscillators])
        position_history.append(positions.copy())

        # update oscillators
        for osc in oscillators:
            osc.step(thetas, K, dt)


    print(phase_history[0])
    return np.array(phase_history), np.array(position_history)


def plot_results(time, phase_history, position_history):
    # Plot phases
    plt.figure()
    for i in range(phase_history.shape[1]):
        plt.plot(time, phase_history[:, i])

    plt.xlabel("Time")
    plt.ylabel("Phase")
    plt.title("Phase vs Time")
    plt.show()

    # Plot positions
    plt.figure()
    for i in range(position_history.shape[1]):
        plt.plot(time, position_history[:, i])

    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.title("Position vs Time")
    plt.show()


def main():
    N = 5
    dt = 0.01
    K = 0.5
    steps = 1000

    oscillators = initialize_oscillators(N)

    phase_history, position_history = run_simulation(
        oscillators, steps, dt, K
    )

    time = np.arange(steps) * dt

    plot_results(time, phase_history, position_history)
    # N = 5
    # dt = 0.01
    # K = 0.5
    # steps = 1000
    # horizon = 20
    # device = "cpu"

    # oscillators = initialize_oscillators(N)

    # model = load_phase_conditioned_model(
    #     model_path="best_phase_conditioned_mlp.pth",
    #     horizon=horizon,
    #     hidden_dim=64,
    #     device=device,
    # )

    # (
    #     phase_history,
    #     analytical_position_history,
    #     mlp_position_history,
    #     mlp_theta_history,
    #     phase_error_history,
    # ) = run_simulation_with_mlp(
    #     oscillators=oscillators,
    #     model=model,
    #     steps=steps,
    #     dt=dt,
    #     K=K,
    #     horizon=horizon,
    #     device=device,
    # )

    # time = np.arange(steps) * dt

    # plot_results_with_mlp(
    #     time,
    #     phase_history,
    #     analytical_position_history,
    #     mlp_position_history,
    # )


if __name__ == "__main__":
    main()
