import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, random_split


# =========================
# 4. Model
# =========================
class MotionMLP(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, horizon=20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * 2)
        )

    def forward(self, x):
        return self.net(x)


# =========================
# 5. Prepare dataloaders
# =========================
def prepare_dataloaders(X, Y, batch_size=128, val_ratio=0.2, seed=42):
    Y = Y.reshape(Y.shape[0], -1)  # flatten from (N, H, 2) to (N, H*2)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, Y_tensor)

    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, Y.shape[1] // 2


# =========================
# 6. Validation function
# =========================
def compute_validation_loss(model, val_loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)

            batch_size = x_batch.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples


# =========================
# 7. Training with best-model saving
# =========================
def train_model(
    train_loader,
    val_loader,
    horizon,
    epochs=20,
    lr=1e-3,
    hidden_dim=64,
    model_path="best_motion_mlp.pth"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MotionMLP(hidden_dim=hidden_dim, horizon=horizon).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        total_train_samples = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = x_batch.size(0)
            total_train_loss += loss.item() * batch_size
            total_train_samples += batch_size

        avg_train_loss = total_train_loss / total_train_samples
        avg_val_loss = compute_validation_loss(model, val_loader, loss_fn, device)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            saved_msg = " <-- saved best model"
        else:
            saved_msg = ""

        print(
            f"Epoch {epoch+1:02d}/{epochs} | "
            f"train loss: {avg_train_loss:.6f} | "
            f"val loss: {avg_val_loss:.6f}"
            f"{saved_msg}"
        )

    print(f"\nBest validation loss: {best_val_loss:.6f}")
    print(f"Best model saved to: {model_path}")

    return model, train_losses, val_losses


# =========================
# 8. Plot loss curves
# =========================
def plot_losses(train_losses, val_losses):
    plt.figure()
    plt.plot(train_losses, label="train loss")
    plt.plot(val_losses, label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()


# =========================
# 9. Evaluate saved best model
# =========================
def evaluate_best_model(model_path, X, Y):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Y_flat = Y.reshape(Y.shape[0], -1)
    horizon = Y.shape[1]

    model = MotionMLP(horizon=horizon).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    i = np.random.randint(0, len(X))

    x_input = torch.tensor(X[i:i+1], dtype=torch.float32).to(device)
    y_true = Y[i]  # shape: (horizon, 2)

    with torch.no_grad():
        y_pred = model(x_input).cpu().numpy().reshape(horizon, 2)

    plt.figure()
    plt.plot(y_true[:, 0], label="true x")
    plt.plot(y_pred[:, 0], label="pred x")
    plt.legend()
    plt.title("Position Prediction")
    plt.show()

    plt.figure()
    plt.plot(y_true[:, 1], label="true theta")
    plt.plot(y_pred[:, 1], label="pred theta")
    plt.legend()
    plt.title("Phase Prediction")
    plt.show()


# =========================
# 10. Main
# =========================
def main():
    print("Generating dataset...")
    data = np.load("oscillator_motion_dataset.npz")
    X = data["X"]   # shape: (N, 3)
    Y = data["Y"]   # shape: (N, H, 2)

    print("\nPreparing train/validation split...")
    train_loader, val_loader, horizon = prepare_dataloaders(
        X, Y, batch_size=128, val_ratio=0.2, seed=42
    )

    print("\nTraining model...")
    _, train_losses, val_losses = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        horizon=horizon,
        epochs=30,
        lr=1e-3,
        hidden_dim=64,
        model_path="best_motion_mlp.pth"
    )

    print("\nPlotting losses...")
    plot_losses(train_losses, val_losses)

    print("\nEvaluating best saved model...")
    evaluate_best_model("best_motion_mlp.pth", X, Y)


if __name__ == "__main__":
    main()