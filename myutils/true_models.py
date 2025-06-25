from typing import Callable, NamedTuple
import numpy as np

from mlp import TorchMLP, init_all_weights


class TrueModels(NamedTuple):
    r0: Callable
    g0: Callable
    f0: Callable


# === GLOBAL FUNCTIONS OR CLASSES ===


class NNBias1:
    def __init__(self, q, p):
        self.model = TorchMLP(input_dim=q, output_dim=p, hidden_layers=[16])
        init_all_weights(self.model, generator_seed=99999)

    def __call__(self, Z):
        return self.model.predict_numpy(Z)


class NNBias2:
    def __init__(self, q, p):
        self.model = TorchMLP(input_dim=q, output_dim=p, hidden_layers=[16, 16])
        init_all_weights(self.model, generator_seed=99999)

    def __call__(self, Z):
        return self.model.predict_numpy(Z)


# === MAIN INTERFACES ===


def generate_true_models(q, p):
    TRUE_r0_MLP = TorchMLP(input_dim=q, output_dim=p, hidden_layers=[32, 32])
    init_all_weights(TRUE_r0_MLP, generator_seed=9999)

    def r0(Z):
        return TRUE_r0_MLP.predict_numpy(Z)

    TRUE_g0_MLP = TorchMLP(input_dim=q, output_dim=1, hidden_layers=[64, 64, 64, 64])
    init_all_weights(TRUE_g0_MLP, generator_seed=9999)

    def g0(Z):
        return TRUE_g0_MLP.predict_numpy(Z).squeeze()

    def f0(X):
        if X.ndim == 1:
            return 2.0 * X
        elif X.ndim == 2:
            beta = np.full((X.shape[1], 1), 2.0)
            return (X @ beta).ravel()

    return TrueModels(r0, g0, f0)


def generate_bias_models(q, p):
    return NNBias1(q, p), NNBias2(q, p)


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    ## 1-dim tests
    Q = 1  # Z in R^Q
    P = 1  # X in R^P
    # Y in R

    N = 100
    Z = rng.standard_normal((N, Q))

    r0, g0, f0 = generate_true_models(Q, P)
    nn_bias_1, nn_bias_2 = generate_bias_models(Q, P)

    assert np.array_equal(r0(Z), r0(Z))
    assert r0(Z).shape == ((N, P) if P != 1 else (N,))
    assert g0(Z).shape == (N,)
    assert f0(r0(Z)).shape == (N,)
    assert nn_bias_1(Z).shape == ((N, P) if P != 1 else (N,))
    assert nn_bias_2(Z).shape == ((N, P) if P != 1 else (N,))

    ## multi-dim tests
    Q = 20  # Z in R^Q
    P = 10  # X in R^P
    # Y in R

    N = 100
    Z = rng.standard_normal((N, Q))

    r0, g0, f0 = generate_true_models(Q, P)
    nn_bias_1, nn_bias_2 = generate_bias_models(Q, P)

    assert np.array_equal(r0(Z), r0(Z))
    assert r0(Z).shape == ((N, P) if P != 1 else (N,))
    assert g0(Z).shape == (N,)
    assert f0(r0(Z)).shape == (N,)
    assert nn_bias_1(Z).shape == ((N, P) if P != 1 else (N,))
    assert nn_bias_2(Z).shape == ((N, P) if P != 1 else (N,))
