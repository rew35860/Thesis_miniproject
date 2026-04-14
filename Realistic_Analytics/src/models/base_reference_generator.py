from abc import ABC, abstractmethod
import torch


class BaseReferenceGenerator(ABC):
    def __init__(self, horizon, dt, device):
        self.horizon = horizon
        self.dt = dt
        self.device = device

    def build_condition(self, x, v, phi, phi_dot):
        """
        Generic condition used by any model:
        [x, v, sin(phi), cos(phi), phi_dot]
        """
        cond = torch.tensor(
            [x, v, torch.sin(phi), torch.cos(phi), phi_dot],
            dtype=torch.float32,
            device=self.device,
        )
        return cond.unsqueeze(0)  # [1, 5]

    @abstractmethod
    def predict_future(self, x, v, phi, phi_dot):
        """
        Must return predicted future trajectory of shape [H].
        """
        pass

    @torch.no_grad()
    def get_reference(self, x, v, phi, phi_dot):
        pred = self.predict_future(x, v, phi, phi_dot)  # [H]

        x_ref = pred[0]

        if pred.shape[0] >= 2:
            v_ref = (pred[1] - pred[0]) / self.dt
        else:
            v_ref = torch.tensor(0.0, device=self.device)

        return x_ref, v_ref, pred, None