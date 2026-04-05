import torch
import torch.nn as nn
from src.models.base_reference_generator import BaseReferenceGenerator
from src.utils.model_io import decode_prediction, build_condition


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=3):
        super().__init__()

        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]

        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def compute_loss(self, batch):
        x, y = batch
        pred = self.forward(x)
        loss = nn.functional.mse_loss(pred, y)
        return loss, {"mse": loss.item()}


class MLPReferenceGenerator:
    def __init__(self, model, condition_mode, predict_mode, horizon, dt, device):
        '''condition_mode: "phase_freq", "state_phase_freq", "phase_trig_freq", "state_phase_trig_freq"
           predict_mode:    True, model predicts both x and v
                            False, model predicts only x and we derive v from x'''
        
        self.model = model
        self.condition_mode = condition_mode
        self.predict_mode = predict_mode
        self.horizon = horizon
        self.dt = dt
        self.device = device

        self.model.eval()

    def build_input(self, x, v, phi, phi_dot):
        return build_condition(
            x=x,
            v=v,
            phi=phi,
            phi_dot=phi_dot,
            mode=self.condition_mode,
            device=self.device,
        )

    @torch.no_grad()
    def predict_future(self, x, v, phi, phi_dot):
        inp = self.build_input(x, v, phi, phi_dot)

        pred = self.model(inp).squeeze(0)
        return pred

    @torch.no_grad()
    def get_reference(self, x, v, phi, phi_dot):
        pred = self.predict_future(x, v, phi, phi_dot)

        x_ref, v_ref, x_pred, v_pred = decode_prediction(
            pred=pred,
            predict_mode=self.predict_mode,
            horizon=self.horizon,
            dt=self.dt,
            device=self.device,
        )

        return x_ref, v_ref, x_pred, v_pred
    

def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)

    model = MLP(
        input_dim=checkpoint["input_dim"],
        output_dim=checkpoint["output_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=checkpoint["num_layers"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint