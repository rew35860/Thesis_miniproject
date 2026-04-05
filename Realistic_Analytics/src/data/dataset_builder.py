import torch
from torch.utils.data import Dataset


class TrajectoryWindowDataset(Dataset):
    """
    Builds supervised samples from simulation results.

    Two condition modes:
    1) "phase_freq"      -> input = [phi_t, omega_tilde_t]
    2) "state_phase_freq"-> input = [x_t, v_t, phi_t, omega_tilde_t]

    Target:
        future positions over horizon H:
            [x_{t+1}, x_{t+2}, ..., x_{t+H}]

    Optional:
        include future velocities too.
    """
    def __init__(
        self,
        results,
        horizon=20,
        condition_mode="phase_freq",
        predict_velocity=False,
        flatten_target=True,
        use_sin_cos_phase=False,
    ):
        super().__init__()

        self.horizon = horizon
        self.condition_mode = condition_mode
        self.predict_velocity = predict_velocity
        self.flatten_target = flatten_target
        self.use_sin_cos_phase = use_sin_cos_phase

        x = results["x"]                  # [T, N]
        v = results["v"]                  # [T, N]
        phi = results["phi"]              # [T, N]
        omega_tilde = results["omega_tilde"]  # [T, N]

        T, N = x.shape

        inputs = []
        targets = []

        # we need future steps t+1 ... t+H
        for i in range(N):
            for t in range(T - horizon - 1):
                x_t = x[t, i]
                v_t = v[t, i]
                phi_t = phi[t, i]
                omega_tilde_t = omega_tilde[t, i]

                if self.condition_mode == "phase_freq":
                    if use_sin_cos_phase:
                        inp = torch.tensor([
                            torch.sin(phi_t),
                            torch.cos(phi_t),
                            omega_tilde_t
                        ], dtype=torch.float32)
                    else:
                        inp = torch.tensor([
                            phi_t,
                            omega_tilde_t
                        ], dtype=torch.float32)

                elif self.condition_mode == "state_phase_freq":
                    if use_sin_cos_phase:
                        inp = torch.tensor([
                            x_t,
                            v_t,
                            torch.sin(phi_t),
                            torch.cos(phi_t),
                            omega_tilde_t
                        ], dtype=torch.float32)
                    else:
                        inp = torch.tensor([
                            x_t,
                            v_t,
                            phi_t,
                            omega_tilde_t
                        ], dtype=torch.float32)
                else:
                    raise ValueError(
                        f"Unknown condition_mode: {self.condition_mode}"
                    )

                x_future = x[t + 1:t + 1 + horizon, i]  # [H]

                if self.predict_velocity:
                    v_future = v[t + 1:t + 1 + horizon, i]  # [H]
                    y = torch.cat([x_future, v_future], dim=0)  # [2H]
                else:
                    y = x_future.unsqueeze(-1)  # [H, 1]

                if self.flatten_target:
                    y = y.reshape(-1)

                inputs.append(inp)
                targets.append(y)

        self.inputs = torch.stack(inputs)    # [num_samples, input_dim]
        self.targets = torch.stack(targets)  # [num_samples, target_dim]

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]