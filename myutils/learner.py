import numpy as np
import torch

from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from skorch import NeuralNetRegressor
from xgboost import XGBRegressor

from mlp import TorchMLP, init_all_weights
from utils import set_seed


class MyNetRegressor(NeuralNetRegressor):
    """
    A wrapper around skorch.NeuralNetRegressor to ensure compatibility
    with scikit-learn-style regressors.

    - During training (`fit`), it reshapes 1D targets `y` to 2D if needed,
      which avoids errors from skorch expecting shape (n_samples, n_outputs).

    - During prediction, it squeezes the output back to 1D if the result has shape (n, 1),
      ensuring consistency with other regressors like LinearRegression or RandomForestRegressor.
    """

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params):
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return super().fit(X, y, **fit_params)

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred = super().predict(X)
        if y_pred.ndim == 2 and y_pred.shape[1] == 1:
            y_pred = y_pred.ravel()
        return y_pred


def wrap_with_pipeline(model, memory=None):
    return Pipeline([("scaler", MinMaxScaler()), ("estimator", model)], memory=memory)


def build_learner(model_type: str, memory=None, **kwargs) -> Pipeline:
    if model_type == "ols":
        return wrap_with_pipeline(LinearRegression(**kwargs), memory)

    elif model_type == "rf":
        output_dim = kwargs.pop("output_dim", 1)
        base_model = RandomForestRegressor(random_state=0, **kwargs)
        model = MultiOutputRegressor(base_model) if output_dim >= 2 else base_model
        return wrap_with_pipeline(model, memory)

    elif model_type == "krr":
        gamma = kwargs.pop("gamma", 0.1)
        alpha = kwargs.pop("alpha", 1.0)
        nystroem_kwargs = {
            k: kwargs.pop(k) for k in ("n_components", "degree", "coef0") if k in kwargs
        }
        nystroem_kwargs.update({"kernel": "rbf", "gamma": gamma, "random_state": 0})
        ridge_kwargs = {"alpha": alpha, "random_state": 0}
        return Pipeline(
            [
                ("scaler", MinMaxScaler()),
                ("feature_map", Nystroem(**nystroem_kwargs)),
                ("ridge", Ridge(**ridge_kwargs)),
            ],
            memory=memory,
        )

    elif model_type == "xgb":
        output_dim = kwargs.pop("output_dim", 1)
        base_model = XGBRegressor(**kwargs)
        model = MultiOutputRegressor(base_model) if output_dim >= 2 else base_model
        return wrap_with_pipeline(model, memory)

    elif model_type == "mlp":
        set_seed(0)
        mlp_params = {
            "input_dim": kwargs.pop("input_dim"),
            "output_dim": kwargs.pop("output_dim"),
            "hidden_layers": kwargs.pop("hidden_layers"),
        }
        model = MyNetRegressor(
            module=TorchMLP,
            module__input_dim=mlp_params["input_dim"],
            module__output_dim=mlp_params["output_dim"],
            module__hidden_layers=mlp_params["hidden_layers"],
            criterion=torch.nn.MSELoss,
            optimizer=torch.optim.Adam,
            verbose=0,
            max_epochs=kwargs.pop("epochs", 200),
            batch_size=kwargs.pop("batch_size", 32),
            train_split=None,
            lr=kwargs.pop("lr", 1e-3),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        model.initialize()
        init_all_weights(model.module_, use_generator=False, generator_seed=0)
        return wrap_with_pipeline(model, memory)

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


def evaluate_model(name: str, learner, X_train, Y_train):
    Y_pred = learner.fit(X_train, Y_train).predict(X_train)
    print(f"{name}: {mean_squared_error(Y_pred, Y_train):.4f}")


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    ## 1-dim tests
    Q, P = 1, 1
    N = 100
    Z = rng.standard_normal((N, Q))

    mlp_generator_ZtoX = TorchMLP(input_dim=Q, output_dim=P, hidden_layers=[32])
    init_all_weights(mlp_generator_ZtoX)
    X = mlp_generator_ZtoX.predict_numpy(Z)

    mlp_generator_XtoY = TorchMLP(input_dim=P, output_dim=1, hidden_layers=[32])
    init_all_weights(mlp_generator_XtoY)
    Y = mlp_generator_XtoY.predict_numpy(X)

    assert X.shape == ((N, P) if P != 1 else (N,))
    assert Y.shape == (N,)
    print(f"Z.mean()={Z.mean():.3f}, X.mean()={X.mean():.3f}, Y.mean()={Y.mean():.3f}")

    for name, model_args in [
        ("ols_X", dict(model_type="ols")),
        ("ols_Y", dict(model_type="ols")),
        (
            "rf_X",
            dict(
                model_type="rf",
                output_dim=P,
                n_estimators=100,
                max_depth=10,
                max_features="sqrt",
                n_jobs=-1,
            ),
        ),
        ("rf_Y", dict(model_type="rf", output_dim=1)),
        ("krr_X", dict(model_type="krr")),
        ("krr_Y", dict(model_type="krr")),
        ("xgb_X", dict(model_type="xgb", output_dim=P)),
        ("xgb_Y", dict(model_type="xgb", output_dim=1)),
        (
            "mlp_X",
            dict(
                model_type="mlp",
                input_dim=Q,
                output_dim=P,
                hidden_layers=[64],
                epochs=200,
                batch_size=32,
            ),
        ),
        (
            "mlp_Y",
            dict(
                model_type="mlp",
                input_dim=P,
                output_dim=1,
                hidden_layers=[64],
                epochs=200,
                batch_size=32,
            ),
        ),
    ]:
        learner = build_learner(**model_args)
        if name.endswith("_X"):
            evaluate_model(name, learner, Z, X)
        else:
            evaluate_model(name, learner, X.reshape(-1, 1), Y)

    ## multi-dim tests
    Q, P = 20, 10
    N = 100
    Z = rng.standard_normal((N, Q))

    mlp_generator_ZtoX = TorchMLP(input_dim=Q, output_dim=P, hidden_layers=[32])
    init_all_weights(mlp_generator_ZtoX)
    X = mlp_generator_ZtoX.predict_numpy(Z)

    mlp_generator_XtoY = TorchMLP(input_dim=P, output_dim=1, hidden_layers=[32])
    init_all_weights(mlp_generator_XtoY)
    Y = mlp_generator_XtoY.predict_numpy(X)

    assert X.shape == ((N, P) if P != 1 else (N,))
    assert Y.shape == (N,)
    print(f"Z.mean()={Z.mean():.3f}, X.mean()={X.mean():.3f}, Y.mean()={Y.mean():.3f}")

    for name, model_args in [
        ("ols_X", dict(model_type="ols")),
        ("ols_Y", dict(model_type="ols")),
        (
            "rf_X",
            dict(
                model_type="rf",
                output_dim=P,
                n_estimators=100,
                max_depth=10,
                max_features="sqrt",
                n_jobs=-1,
            ),
        ),
        ("rf_Y", dict(model_type="rf", output_dim=1)),
        ("krr_X", dict(model_type="krr")),
        ("krr_Y", dict(model_type="krr")),
        ("xgb_X", dict(model_type="xgb", output_dim=P)),
        ("xgb_Y", dict(model_type="xgb", output_dim=1)),
        (
            "mlp_X",
            dict(
                model_type="mlp",
                input_dim=Q,
                output_dim=P,
                hidden_layers=[64],
                epochs=200,
                batch_size=32,
            ),
        ),
        (
            "mlp_Y",
            dict(
                model_type="mlp",
                input_dim=P,
                output_dim=1,
                hidden_layers=[64],
                epochs=200,
                batch_size=32,
            ),
        ),
    ]:
        learner = build_learner(**model_args)
        if name.endswith("_X"):
            evaluate_model(name, learner, Z, X)
        else:
            evaluate_model(name, learner, X, Y)
