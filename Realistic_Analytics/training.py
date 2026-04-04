from pathlib import Path
import torch
from tqdm import tqdm

from src.utils.plotting import plot_full_trajectory, plot_losses, plot_predictions, plot_dataset_samples
from src.utils.dataset_io import make_dataloaders
from src.models.mlp import MLP
from src.utils.model_io import load_model
from src.config import get_mlp_config


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            loss, metrics = model.compute_loss((xb, yb))

            batch_size = xb.shape[0]
            total_loss += loss.item() * batch_size
            total_count += batch_size

    return total_loss / total_count


def train_model(
    dataset_path,
    train_cfg,
    device,
    model_save_path="mlp_model.pt",
):
    
    dataset, X, Y, metadata, train_loader, val_loader, test_loader = make_dataloaders(
        dataset_path=dataset_path,
        batch_size=train_cfg.batch_size,
        return_metadata=True,
    )

    input_dim = X.shape[1]
    output_dim = Y.shape[1]

    print("Dataset:", dataset_path)
    print("Input dim:", input_dim)
    print("Output dim:", output_dim)
    print("Num samples:", len(dataset))
    print("Metadata:", metadata)

    model = MLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=train_cfg.hidden_dim,
        num_layers=train_cfg.num_layers,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_state = None
    pbar = tqdm(range(1, train_cfg.epochs + 1))

    for epoch in pbar:
        pbar.set_description(f"Epoch {epoch}/{train_cfg.epochs}")

        model.train()
        total_train_loss = 0.0
        total_count = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            loss, metrics = model.compute_loss((xb, yb))
            loss.backward()
            optimizer.step()

            batch_size_now = xb.shape[0]
            total_train_loss += loss.item() * batch_size_now
            total_count += batch_size_now

        train_loss = total_train_loss / total_count
        val_loss = evaluate(model, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "output_dim": output_dim,
                "hidden_dim": train_cfg.hidden_dim,
                "num_layers": train_cfg.num_layers,
                "dataset_path": str(dataset_path),
                "metadata": metadata,
            }

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | "
                f"train loss = {train_loss:.6f} | "
                f"val loss = {val_loss:.6f}"
            )

    model_save_path = Path(model_save_path)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(best_state, model_save_path)
    print(f"Saved best model to {model_save_path}")
    print(f"Best val loss: {best_val_loss:.6f}")

    model.load_state_dict(best_state["model_state_dict"])
    test_loss = evaluate(model, test_loader, device)
    print(f"Test loss: {test_loss:.6f}")

    return model, X, Y, metadata, train_losses, val_losses, device


if __name__ == "__main__":
    # Choose one:
    dataset_path = "./Realistic_Analytics/data/dataset_phase_freq.pt"
    model_save_path = "./Realistic_Analytics/models/mlp_phase_freq.pt"

    # dataset_path = "./Realistic_Analytics/data/dataset_state_phase_freq.pt"
    # model_save_path = "./Realistic_Analytics/models/mlp_state_phase_freq.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # model, X, Y, metadata, train_losses, val_losses, device = train_model(
    #     dataset_path=dataset_path,
    #     model_save_path=model_save_path,
    #     train_cfg=get_mlp_config(),
    #     device=device,
    # )

    # plot_losses(train_losses, val_losses)
    # plot_predictions(model, X, Y, device)
    # plot_full_trajectory(model, X, Y, metadata, device)
    # plot_dataset_samples(Y, metadata, num_points=2000)
