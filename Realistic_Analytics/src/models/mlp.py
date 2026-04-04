import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt


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

