from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

# Set global default dtype for all tensors
torch.set_default_dtype(torch.float64)


class TorchMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 1,
        hidden_layers: Optional[List[int]] = None,
    ):
        super().__init__()
        hidden_layers = hidden_layers or []
        dims = [input_dim] + hidden_layers + [output_dim]

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.ndim == 1:
            X = X.unsqueeze(1)
        return self.net(X)

    def predict_numpy(self, X: np.ndarray) -> np.ndarray:
        """Run prediction on NumPy array input and return NumPy output."""
        self.eval()
        with torch.no_grad():
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            X_tensor = torch.from_numpy(X)
            Y = self.forward(X_tensor).cpu().numpy()
            return Y.squeeze(1) if Y.ndim == 2 and Y.shape[1] == 1 else Y


def init_all_weights(
    model: nn.Module,
    use_generator: bool = True,
    generator_seed: int = 9999,
) -> None:
    """Initialize all Linear layers with reproducibility."""
    if not isinstance(model, nn.Module):
        raise TypeError(f"Expected nn.Module, got {type(model)}")

    for i, layer in enumerate(model.modules()):
        if isinstance(layer, nn.Linear):
            gen = torch.Generator().manual_seed(generator_seed + i)
            if use_generator:
                nn.init.normal_(layer.weight, generator=gen)
            else:
                nn.init.kaiming_normal_(
                    layer.weight, nonlinearity="relu", generator=gen
                )
            nn.init.zeros_(layer.bias)


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    ## 1-dim tests
    Q, P = 1, 1
    N = 100
    # 1. Generate synthetic data from a ground-truth model
    generator = TorchMLP(input_dim=Q, output_dim=P, hidden_layers=[20, 20])
    init_all_weights(generator, use_generator=True, generator_seed=42)

    X_np = rng.standard_normal((N, Q))
    Y_np = generator.predict_numpy(X_np)
    assert Y_np.shape == (N,)

    # 2. Prepare training tensors
    X_tensor = torch.tensor(X_np)
    Y_tensor = torch.tensor(Y_np).unsqueeze(1)  # shape: (1000, 1)
    assert Y_tensor.shape == (N, P)

    # 3. Initialize training model
    model = TorchMLP(input_dim=Q, output_dim=P, hidden_layers=[20, 20])
    init_all_weights(model, use_generator=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # 4. Training loop
    model.train()
    for epoch in range(1000):
        optimizer.zero_grad()
        predictions = model(X_tensor)
        loss = loss_fn(predictions, Y_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    ## multi-dim tests
    Q, P = 20, 10
    N = 100
    # 1. Generate synthetic data from a ground-truth model
    generator = TorchMLP(input_dim=Q, output_dim=P, hidden_layers=[20, 20])
    init_all_weights(generator, use_generator=True, generator_seed=42)

    X_np = rng.standard_normal((N, Q))
    Y_np = generator.predict_numpy(X_np)
    assert Y_np.shape == (N, P)

    # 2. Prepare training tensors
    X_tensor = torch.tensor(X_np)
    Y_tensor = torch.tensor(Y_np)
    assert Y_tensor.shape == (N, P)

    # 3. Initialize training model
    model = TorchMLP(input_dim=Q, output_dim=P, hidden_layers=[20, 20])
    init_all_weights(model, use_generator=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # 4. Training loop
    model.train()
    for epoch in range(1000):
        optimizer.zero_grad()
        predictions = model(X_tensor)
        loss = loss_fn(predictions, Y_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
