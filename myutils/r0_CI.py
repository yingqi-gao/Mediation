import os

import numpy as np
import scipy
from ppi_py import ppi_mean_ci

from data_generator import (
    GeneratedData,
    DataGeneratorParam,
    TrainDataParam,
    RealDataParam,
    ExpandedDataParam,
    DataGenerator,
)
from train_rhat import train_rhat
from utils import timeit, read_pickle, write_pickle, get_r0_CI_directory_uri


def _construct_r0_CI(
    *,
    real_data: GeneratedData,
    Z0: np.ndarray,
    expanded_data: np.ndarray,
    rhat,
    r0,
    alpha=0.05,
) -> dict:

    Z, X, Y = real_data
    N = X.shape[0]
    p = X.shape[1] if X.ndim > 1 else 1
    Zp = expanded_data
    Np = Zp.shape[0]

    # Retrieve r_hat using the first cross-fitting result & make predictions
    X_hat = rhat.predict(Z)
    assert X_hat.shape == X.shape
    X_hatp = rhat.predict(Zp)
    assert X_hatp.shape == (Zp.shape[0], p) if p > 1 else (Zp.shape[0], )

    lower, upper = ppi_mean_ci(X, X_hat, X_hatp, alpha=alpha)

    # # Find the midpoint
    # mean_X_hatp = X_hatp.mean(axis=0)
    # if p > 1:
    #     assert mean_X_hatp.shape == (p,)
    # else:
    #     assert np.isscalar(mean_X_hatp) or mean_X_hatp.shape == ()
    # delta = (X_hat - X).mean(axis=0)
    # if p > 1:
    #     assert delta.shape == (p,)
    # else:
    #     assert np.isscalar(delta) or delta.shape == ()
    # midpoint = mean_X_hatp - delta
    # #     print(f"{midpoint=}")
    # #     print(f"discrepancy from the midpoint: {np.mean(r0(Z0) - midpoint)}")

    # # Find the width
    # sigma2_1 = np.var(X_hat - X, axis=0, ddof=0)
    # if p > 1:
    #     assert sigma2_1.shape == (p,)
    # else:
    #     assert np.isscalar(sigma2_1) or sigma2_1.shape == ()
    # sigma2_2 = np.var(X_hatp, axis=0, ddof=0)
    # if p > 1:
    #     assert sigma2_2.shape == (p,)
    # else:
    #     assert np.isscalar(sigma2_2) or sigma2_2.shape == ()
    # z_crit = scipy.stats.norm.ppf(1 - alpha / (2 * p))
    # w_theta = z_crit * np.sqrt(sigma2_1 + sigma2_2)
    # if p > 1:
    #     assert w_theta.shape == (p,)
    # else:
    #     assert np.isscalar(w_theta) or w_theta.shape == ()
    # me = np.mean(w_theta)

    # # Find the endpoints
    # lower = midpoint - w_theta
    # upper = midpoint + w_theta

    # Construct the CI
    r0_CI = {
        "lower": lower,
        "upper": upper,
        "covers?": (lower < r0(Z0)) & (r0(Z0) < upper),
        "me": np.mean((upper - lower) / 2),
        # "mean_X_hatp": mean_X_hatp,
        # "delta": delta,
        # "z_crit": z_crit,
        # "prediction error": np.mean(sigma2_1),
        # "expanded error": np.mean(sigma2_2),
    }

    return r0_CI


@timeit
def construct_r0_CIs(
    *,
    data_generator_param: DataGeneratorParam,
    real_data_param: RealDataParam,
    expanded_data_param: ExpandedDataParam,
    model_directory_uri: str,
    rhat, 
    r0,
    alpha = 0.05,
    repetitions = 500,
    fresh = False
) -> dict:

    results_directory_uri = get_r0_CI_directory_uri(
        real_data_param=real_data_param,
        expanded_data_param=expanded_data_param,
        model_directory_uri=model_directory_uri,
    )
    results_uri = os.path.join(results_directory_uri, "r0_CIs.pkl")

    if os.path.exists(results_uri) and not fresh:
        r0_CIs = read_pickle(results_uri)

    data_generator = DataGenerator(data_generator_param)
    Z0 = data_generator.generate_target_point(real_data_param, seed=0).Z0
    r0_CIs = []

    for i in range(repetitions):
        real_data = data_generator.generate_real_data(real_data_param, seed=i)
        expanded_data = data_generator.generate_expanded_data(
            expanded_data_param, Z0, seed=i
        )
        r0_CIs.append(
            _construct_r0_CI(
                real_data = real_data,
                Z0 = Z0,
                expanded_data = expanded_data,
                rhat = rhat, 
                r0 = r0, 
                alpha = alpha,
            )
        )
    write_pickle(r0_CIs, results_uri)

    coverage = np.mean([ci["covers?"] for ci in r0_CIs])
    avg_me = np.mean([ci["me"] for ci in r0_CIs])
    print(f"Coverage: {coverage:.3f}\nAverage ME: {avg_me:.3f}")
    # print(f"z_crit: {np.mean([ci['z_crit'] for ci in r0_CIs])}\n")
    # print(f"average prediction error: {np.mean([ci['prediction error'] for ci in r0_CIs])}\n")
    # print(f"average expanded error: {np.mean([ci['expanded error'] for ci in r0_CIs])}\n")
    return r0_CIs, coverage, avg_me


if __name__ == '__main__':
    import numpy as np

    from true_models import generate_true_models, generate_bias_models
    from learner import build_learner
    from utils import get_model_directory_uri

    ## ## 1-dim tests
    Q, P = 1, 1
    N_TRAIN = 1000
    N_REAL = 100
    N_EXPANDED = 1000
    SEED = 999

    r0, g0, f0 = generate_true_models(Q, P)
    nn_bias_1, nn_bias_2 = generate_bias_models(Q, P)

    data_generator_param = DataGeneratorParam(p=P, q=Q, r0=r0, g0=g0, f0=f0)
    train_data_param = TrainDataParam(n_train=N_TRAIN)
    real_data_param = RealDataParam(bias_func=nn_bias_1, bias_scale=0, n_real=N_REAL)
    expanded_data_param = ExpandedDataParam(N_EXPANDED, 0.1)

    OUTPUT_DIRECTORY_URI = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "results"
    )

    learners = {
        "linear": build_learner(model_type="ols"),
        "random_forest": build_learner(
            model_type="rf",
            output_dim=P,
            n_estimators=100,
            max_features="sqrt",
            n_jobs=-1,
        ),
        "kernel": build_learner(model_type="krr"),
        "xgboost": build_learner(model_type="xgb", output_dim=P),
        "neural_net_128x128_1000_64": build_learner(
            model_type="mlp",
            input_dim=Q,
            output_dim=P,
            hidden_layers=[128, 128],
            epochs=1000,
            batch_size=64,
        ),
    }

    for name, learner in learners.items():
        model_directory_uri = get_model_directory_uri(
            data_generator_param=data_generator_param,
            train_data_param=train_data_param,
            r0_learner_name=name,
            output_directory_uri=OUTPUT_DIRECTORY_URI,
        )
        rhat = train_rhat(
            data_generator_param=data_generator_param,
            train_data_param=train_data_param,
            model_directory_uri=model_directory_uri,
            learner_name=name,
            learner=learner,
            seed=SEED,
            fresh=True,
        )
        construct_r0_CIs(
            data_generator_param=data_generator_param,
            real_data_param=real_data_param,
            expanded_data_param=expanded_data_param,
            model_directory_uri=model_directory_uri,
            rhat=rhat, 
            r0=r0,
            fresh=True,
        )

    ## ## multi-dim tests
    Q, P = 20, 10
    N_TRAIN = 10000
    N_REAL = 100
    N_EXPANDED = 1000
    SEED = 999

    r0, g0, f0 = generate_true_models(Q, P)
    nn_bias_1, nn_bias_2 = generate_bias_models(Q, P)

    data_generator_param = DataGeneratorParam(p=P, q=Q, r0=r0, g0=g0, f0=f0)
    train_data_param = TrainDataParam(n_train=N_TRAIN)
    real_data_param = RealDataParam(bias_func=nn_bias_1, bias_scale=0, n_real=N_REAL)
    expanded_data_param = ExpandedDataParam(N_EXPANDED, 0.1)

    OUTPUT_DIRECTORY_URI = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "results"
    )

    learners = {
        "linear": build_learner(model_type="ols"),
        "random_forest": build_learner(
            model_type="rf",
            output_dim=P,
            n_estimators=100,
            max_features="sqrt",
            n_jobs=-1,
        ),
        "kernel": build_learner(model_type="krr"),
        "xgboost": build_learner(model_type="xgb", output_dim=P),
        "neural_net_128x128_1000_64": build_learner(
            model_type="mlp",
            input_dim=Q,
            output_dim=P,
            hidden_layers=[128, 128],
            epochs=1000,
            batch_size=64,
        ),
    }

    for name, learner in learners.items():
        model_directory_uri = get_model_directory_uri(
            data_generator_param=data_generator_param,
            train_data_param=train_data_param,
            r0_learner_name=name,
            output_directory_uri=OUTPUT_DIRECTORY_URI,
        )
        rhat = train_rhat(
            data_generator_param=data_generator_param,
            train_data_param=train_data_param,
            model_directory_uri=model_directory_uri,
            learner_name=name,
            learner=learner,
            seed=SEED,
            fresh=True,
        )
        construct_r0_CIs(
            data_generator_param=data_generator_param,
            real_data_param=real_data_param,
            expanded_data_param=expanded_data_param,
            model_directory_uri=model_directory_uri,
            rhat=rhat,
            r0=r0,
            fresh=True,
        )
