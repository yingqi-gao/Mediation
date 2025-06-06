from data_generator import GeneratedData, DataGeneratorParam, TrainDataParam, RealDataParam, ExpandedDataParam, DataGenerator
from train_model import get_model_directory_uri, train_model
import numpy as np
import scipy
from sklearn.metrics import mean_squared_error
import torch
import os
from utils import get_dict_hash, timeit, read_pickle, write_pickle



def get_results_directory_uri(
    *,
    real_data_param: RealDataParam, 
    expanded_data_param: ExpandedDataParam,
    model_directory_uri: str,
) -> str:
    real_data_param_dict = real_data_param.to_dict()
    expanded_data_param_dict = expanded_data_param.to_dict()
    
    return os.path.join(
        model_directory_uri,
        "r0_CIs",
        f"real_data_param={get_dict_hash(real_data_param_dict)}",
        f"expanded_data_param={get_dict_hash(expanded_data_param_dict)}",
    )



def _construct_r0_CI(
    *,
    real_data: GeneratedData,
    target_point: GeneratedData,
    expanded_data: np.ndarray,
    r_hat, 
    alpha = 0.05,
) -> dict:
        
    Z, X, Y = real_data
    p = X.shape[1]
    Z0, X0, Y0, wZ0 = target_point
    Zp = expanded_data

    # Retrieve r_hat using the first cross-fitting result & make predictions
    X_hat = r_hat.predict(Z)
    assert X_hat.shape == X.shape
    X_hatp = r_hat.predict(Zp)
    assert X_hatp.shape == (Zp.shape[0], X.shape[1])
    
    # Find the midpoint
    mean_X_hatp = X_hatp.mean(axis=0)
    assert mean_X_hatp.shape == (p, )
    delta = (X_hat - X).mean(axis=0)
    assert delta.shape == (p, )
    midpoint = mean_X_hatp - delta   
#     print(f"{mean_X_hatp=}")
#     print(f"{delta=}")
#     print(f"{X0=}")
#     print(f"{midpoint=}")
#     print(f"discrepancy from the midpoint: {np.mean(X0 - midpoint)}")
    
    # Find the width
    sigma2_1 = np.var(X_hat - X, axis=0, ddof=0)
    assert sigma2_1.shape == (p, )
    sigma2_2 = np.var(X_hatp, axis=0, ddof=0)
    assert sigma2_2.shape == (p, )
    z_crit = scipy.stats.norm.ppf(1 - alpha / (2 * X.shape[1] if X.shape[1]>1 else 1))
    w_theta = z_crit * np.sqrt(sigma2_1 + sigma2_2)
    assert w_theta.shape == (p, )
#     print(f"MSE: {mean_squared_error(X, X_hat)}")
#     print(f"prediction error: {np.mean(sigma2_1)}")
#     print(f"expanded error: {np.mean(sigma2_2)}")
    me = np.mean(w_theta)
#     print(f"ME: {me}")
    
    # Find the endpoints
    lower = midpoint - w_theta
    upper = midpoint + w_theta
    
    # Construct the CI
    r0_CI = {
        "lower": lower,
        "upper": upper,
        "covers?": (np.all(lower < X0)) and (np.all(X0 < upper)),
        "me": me,
    }
    
    return r0_CI



@timeit
def construct_r0_CIs(
    *,
    data_generator_param: DataGeneratorParam,
    train_data_param: TrainDataParam,
    real_data_param: RealDataParam,
    expanded_data_param: ExpandedDataParam,
    output_directory_uri: str,
    r0_learner_name: str, 
    r0_learner,
    alpha = 0.05,
    repetitions: int = 500,
    fresh = False
) -> dict:
    
    r_hat, model_directory_uri = train_model(
        data_generator_param = data_generator_param, 
        train_data_param = train_data_param, 
        which_model = "rhat",
        output_directory_uri = output_directory_uri,
        r0_learner_name = r0_learner_name,
        r0_learner = r0_learner, 
    )
    
    results_directory_uri = get_results_directory_uri(
        real_data_param = real_data_param,
        expanded_data_param = expanded_data_param,
        model_directory_uri = model_directory_uri,
    )
    results_uri = os.path.join(results_directory_uri, "r0_CIs.pkl")
    if os.path.exists(results_uri) and not fresh:
        r0_CIs = read_pickle(results_uri)
    else:
        data_generator = DataGenerator(data_generator_param)
        target_point = data_generator.generate_target_point(real_data_param, seed=0)
        r0_CIs = []
        for i in range(repetitions):
            real_data = data_generator.generate_real_data(real_data_param, seed=i)
            expanded_data = data_generator.generate_expanded_data(
                expanded_data_param, 
                target_point.Z0, 
                seed=i
            )
            r0_CIs.append(
                _construct_r0_CI(
                    real_data = real_data,
                    target_point = target_point,
                    expanded_data = expanded_data,
                    r_hat = r_hat, 
                    alpha = alpha,
                )
            )
        write_pickle(r0_CIs, results_uri)
    
    coverage, avg_me = [], []
    for ci in r0_CIs:
        coverage.append(ci["covers?"])
        avg_me.append(ci["me"])
    print(f"Coverage: {np.mean(coverage)}\n Average ME: {np.mean(avg_me)}\n")
    
    return r0_CIs


if __name__ == '__main__':
    from true_models import generate_true_models, generate_bias_models
    from learner import build_learner
    import numpy as np

    
    
    Q = 100   # Z in R^Q
    P = 100   # X in R^P
              # Y in R

    r0, g0, f0 = generate_true_models(Q, P)
    nn_bias_1, nn_bias_2 = generate_bias_models(Q, P)
    
    data_generator_param = DataGeneratorParam(p=P, q=Q, r0=r0, g0=g0, f0=f0)
    train_data_param = TrainDataParam(n_train=50000)
    real_data_param = RealDataParam(bias_func=nn_bias_1, bias_scale=0, n_real=100)
    expanded_data_param = ExpandedDataParam(100, 0.1)
    
    
    
    OUTPUT_DIRECTORY_URI = "/u/scratch/y/yqg36/Mediation/results"

    
    
    ols_r0_CI = construct_r0_CIs(
        data_generator_param = data_generator_param,
        train_data_param = train_data_param,
        real_data_param = real_data_param,
        expanded_data_param = expanded_data_param,
        output_directory_uri = OUTPUT_DIRECTORY_URI,
        r0_learner_name = "linear", 
        r0_learner = build_learner(model_type='ols'),
    )
    
    
    rf_r0_CI = construct_r0_CIs(
        data_generator_param = data_generator_param,
        train_data_param = train_data_param,
        real_data_param = real_data_param,
        expanded_data_param = expanded_data_param,
        output_directory_uri = OUTPUT_DIRECTORY_URI,
        r0_learner_name = "random_forest", 
        r0_learner = build_learner(
            model_type='rf', 
            output_dim=P, 
            n_estimators=100, 
            # max_depth=10,
            max_features='sqrt',
            n_jobs=-1,
        ),
    )
    
    
    krr_r0_CI = construct_r0_CIs(
        data_generator_param = data_generator_param,
        train_data_param = train_data_param,
        real_data_param = real_data_param,
        expanded_data_param = expanded_data_param,
        output_directory_uri = OUTPUT_DIRECTORY_URI,
        r0_learner_name = "kernel", 
        r0_learner = build_learner(model_type='krr'), 
    )
    
    
    xgb_r0_CI = construct_r0_CIs(
        data_generator_param = data_generator_param,
        train_data_param = train_data_param,
        real_data_param = real_data_param,
        expanded_data_param = expanded_data_param,
        output_directory_uri = OUTPUT_DIRECTORY_URI,
        r0_learner_name = "xgboost", 
        r0_learner = build_learner(model_type='xgb', output_dim=P),  
    )
    
    
    mlp_r0_CI = construct_r0_CIs(
        data_generator_param = data_generator_param,
        train_data_param = train_data_param,
        real_data_param = real_data_param,
        expanded_data_param = expanded_data_param,
        output_directory_uri = OUTPUT_DIRECTORY_URI,
        r0_learner_name = "neural_net_128x128_1000_64", 
        r0_learner = build_learner(
            model_type = 'mlp', 
            input_dim = P,  
            output_dim = Q,
            hidden_layers = [128, 128],
            epochs = 1000,
            batch_size = 64
        ),   
    )
    