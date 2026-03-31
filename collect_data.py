import numpy as np
from oscillator import Oscillator


def simulate_single_oscillator(theta0, omega, steps, dt, amplitude=1.0):
    """
    Simulate one oscillator and return full phase and position histories.
    """
    osc = Oscillator(theta=theta0, omega=omega, amplitude=amplitude)

    theta_history = []
    position_history = []

    for _ in range(steps):
        theta_history.append(osc.theta)
        position_history.append(osc.get_position())

        # uncoupled oscillator for simple dataset generation
        thetas_all = np.array([osc.theta])
        osc.step(thetas_all=thetas_all, K=0.0, dt=dt)

    return np.array(theta_history), np.array(position_history)


def create_training_examples(theta_history, position_history, omega, horizon):
    """
    Create supervised examples.

    Input:
        [x_t, theta_t, omega]

    Target:
        [
            [x_{t+1}, theta_{t+1}],
            [x_{t+2}, theta_{t+2}],
            ...
            [x_{t+horizon}, theta_{t+horizon}]
        ]
    """
    X = []
    Y = []

    T = len(theta_history)

    for t in range(T - horizon):
        x_t = position_history[t]
        theta_t = theta_history[t]

        input_vec = np.array([x_t, theta_t, omega], dtype=np.float32)

        future_x = position_history[t + 1 : t + 1 + horizon]
        future_theta = theta_history[t + 1 : t + 1 + horizon]

        target_seq = np.stack([future_x, future_theta], axis=1).astype(np.float32)
        # shape: (horizon, 2)

        X.append(input_vec)
        Y.append(target_seq)

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def collect_dataset(
    num_trajectories=200,
    steps=300,
    dt=0.01,
    horizon=20,
    omega_range=(0.5, 2.0),
    amplitude=1.0,
    seed=42,
):
    """
    Generate a dataset from many simulated oscillator trajectories.

    Returns:
        X: shape (num_samples, 3)
           each row = [x_t, theta_t, omega]

        Y: shape (num_samples, horizon, 2)
           each row = future sequence of [x, theta]
    """
    rng = np.random.default_rng(seed)

    all_X = []
    all_Y = []

    for _ in range(num_trajectories):
        theta0 = rng.uniform(0.0, 2.0 * np.pi)
        omega = rng.uniform(omega_range[0], omega_range[1])

        theta_history, position_history = simulate_single_oscillator(
            theta0=theta0,
            omega=omega,
            steps=steps,
            dt=dt,
            amplitude=amplitude,
        )

        X, Y = create_training_examples(
            theta_history=theta_history,
            position_history=position_history,
            omega=omega,
            horizon=horizon,
        )

        all_X.append(X)
        all_Y.append(Y)

    all_X = np.concatenate(all_X, axis=0)
    all_Y = np.concatenate(all_Y, axis=0)

    return all_X, all_Y


def save_dataset(filename, X, Y):
    """
    Save dataset into a compressed .npz file.
    """
    np.savez_compressed(filename, X=X, Y=Y)
    print(f"Saved dataset to {filename}")
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")


def main():
    X, Y = collect_dataset(
        num_trajectories=200,
        steps=300,
        dt=0.01,
        horizon=20,
        omega_range=(0.5, 2.0),
        amplitude=1.0,
        seed=42,
    )

    save_dataset("oscillator_motion_dataset.npz", X, Y)


if __name__ == "__main__":
    main()