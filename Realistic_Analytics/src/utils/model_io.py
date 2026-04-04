import torch
from src.models.mlp import MLP


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