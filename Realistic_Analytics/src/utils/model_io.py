import torch


def get_input_dim(condition_mode):
    if condition_mode == "phase_freq":
        return 2
    elif condition_mode == "state_phase_freq":
        return 4
    elif condition_mode == "phase_trig_freq":
        return 3
    elif condition_mode == "state_phase_trig_freq":
        return 5
    else:
        raise ValueError(f"Unknown condition mode: {condition_mode}")


def get_output_dim(horizon, predict_mode):
    if predict_mode:
        return 2 * horizon
    else:
        return horizon


def build_condition(x, v, phi, phi_dot, mode, device):
    if mode == "phase_freq":
        parts = [phi, phi_dot]
    elif mode == "state_phase_freq":
        parts = [x, v, phi, phi_dot]
    elif mode == "phase_trig_freq":
        parts = [torch.sin(phi), torch.cos(phi), phi_dot]
    elif mode == "state_phase_trig_freq":
        parts = [x, v, torch.sin(phi), torch.cos(phi), phi_dot]
    else:
        raise ValueError(f"Unknown condition mode: {mode}")

    cond = torch.stack(parts).to(device).float()
    return cond.unsqueeze(0)


def decode_prediction(pred, predict_mode, horizon, dt, device):
    if not predict_mode:
        x_pred = pred[:horizon]
        x_ref = x_pred[0]

        if horizon >= 2:
            v_ref = (x_pred[1] - x_pred[0]) / dt
        else:
            v_ref = torch.tensor(0.0, device=device)

        return x_ref, v_ref, x_pred, None

    else:
        x_pred = pred[:horizon]
        v_pred = pred[horizon:]

        x_ref = x_pred[0]
        v_ref = v_pred[0]

        return x_ref, v_ref, x_pred, v_pred
