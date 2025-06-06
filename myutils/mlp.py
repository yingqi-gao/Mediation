import torch
import torch.nn as nn
import numpy as np




torch.set_default_dtype(torch.float64)

class TorchMLP(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_layers=None):
        super().__init__()
        hidden_layers = hidden_layers or []

        dims = [input_dim] + hidden_layers + [output_dim]
        layers = []

        for i in range(len(dims) - 1):
            layer = nn.Linear(dims[i], dims[i + 1])
            layers.append(layer)
            if i < len(dims) - 2:
                layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, X):
        Y = self.net(X)
        # print(Y.shape)
        return Y  
    
    def predict_numpy(self, X: np.ndarray):
        self.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X)
            Y = self.forward(X_tensor).cpu().numpy()
            if Y.ndim == 2 and Y.shape[1] == 1:
                Y = Y.squeeze(1)
            return Y
    
    
    
def init_all_weights(model: nn.Module, generator=True, generator_seed=9999):
    if not isinstance(model, nn.Module):
        raise TypeError(f"Expected nn.Module, got {type(model)}")

    for i, m in enumerate(model.modules()):
        if isinstance(m, nn.Linear):
            gen = torch.Generator().manual_seed(generator_seed + i)
            if generator:
                nn.init.normal_(m.weight, generator=gen)
            else:
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu', generator=gen)
            nn.init.zeros_(m.bias)
    
    
    

if __name__ == "__main__":
    model = TorchMLP(input_dim=5, output_dim=1, hidden_layers=[20, 20])
    init_all_weights(model)
    x = torch.randn(10, 5)
    y = model(x)
    print(y.shape)
    
    x = np.random.randn(10, 5)
    y = model.predict_numpy(x)
    print(y.shape)