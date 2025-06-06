from typing import NamedTuple, Callable
from mlp import TorchMLP, init_all_weights
import numpy as np




class TrueModels(NamedTuple):
    r0: Callable
    g0: Callable
    f0: Callable
        

        
def generate_true_models(q, p):
    # Xi = r0(Zi) + Vi
    TRUE_r0_MLP = TorchMLP(
        input_dim=q,
        output_dim=p,
        hidden_layers=[32, 32],
    )
    init_all_weights(TRUE_r0_MLP, generator_seed=9999)
    def r0(Z):
        return TRUE_r0_MLP.predict_numpy(Z)
    
    TRUE_g0_MLP = TorchMLP(
        input_dim=p,
        output_dim=1,
        hidden_layers=[64, 64, 64, 64],
    )
    init_all_weights(TRUE_g0_MLP, generator_seed=9999)
    # Yi = g0(Z_i) + f0(Xi) + Ui
    def g0(Z):
        return TRUE_g0_MLP.predict_numpy(Z)


    BETA_ENTRY = 2.0  # Y = 2X + ...
    def f0(X):
        p = X.shape[1]
        beta = np.full((p, 1), BETA_ENTRY)
        return (X @ beta).ravel()
    
    
    return TrueModels(r0, g0, f0)


def generate_bias_models(q, p):
    TRUE_BIAS1 = TorchMLP(
        input_dim=q,
        output_dim=p,
        hidden_layers=[16],
    )
    init_all_weights(TRUE_BIAS1, generator_seed=99999)
    def nn_bias_1(Z):
        return TRUE_BIAS1.predict_numpy(Z)


    TRUE_BIAS2 = TorchMLP(
        input_dim=q,
        output_dim=p,
        hidden_layers=[16, 16],
    )
    init_all_weights(TRUE_BIAS2, generator_seed=99999)
    def nn_bias_2(Z):
        return TRUE_BIAS2.predict_numpy(Z)

    return nn_bias_1, nn_bias_2



if __name__ == '__main__':
    Q = 100   # Z in R^Q
    P = 100   # X in R^P
              # Y in R
    N = 100
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((N, Q))
    
    r0, g0, f0 = generate_true_models(Q, P)
    nn_bias_1, nn_bias_2 = generate_bias_models(Q, P)
    
    assert np.array_equal(r0(Z), r0(Z))
    assert r0(Z).shape == ((N, P) if P!=1 else (N, ))
    assert g0(Z).shape == (N, )
    assert f0(r0(Z)).shape == (N, )
    assert nn_bias_1(Z).shape == ((N, P) if P!=1 else (N, ))
    assert nn_bias_2(Z).shape == ((N, P) if P!=1 else (N, ))