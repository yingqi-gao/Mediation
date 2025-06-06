from data_generator import DataGeneratorParam, TrainDataParam
from typing import Literal
from utils import get_dict_hash, timeit, read_pickle, write_pickle
import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.base import clone




def get_model_directory_uri(
    *,
    data_generator_param: DataGeneratorParam, 
    train_data_param: TrainDataParam,
    which_model: Literal['rhat', 'dml'],
    r0_learner_name: str,
    g0_learner_name: str = None,
    output_directory_uri: str,
) -> str:
    data_generator_param_dict = data_generator_param.to_dict()
    train_data_param_dict = train_data_param.to_dict()
    
    return os.path.join(
        output_directory_uri,
        f'data_generator_param={get_dict_hash(data_generator_param_dict)}',
        f'train_data_param={get_dict_hash(train_data_param_dict)}',
        f'model={which_model}',
        f'r0_learner={r0_learner_name}',
        f'g0_learner={g0_learner_name}'
    )



def train_dml(
    *,
    Z: np.ndarray, 
    X: np.ndarray, 
    Y: np.ndarray,
    r0_learner,
    g0_learner,
    kfolds = 2,
) -> dict:
    # ======= Cross-fitting: Learn r(Z), g(Z) =======
    kf = KFold(n_splits=kfolds, shuffle=True, random_state=0)
    r_hats, g_hats = [], []
    X_resid = np.zeros_like(X)
    Y_resid = np.zeros_like(Y)

    for train_idx, test_idx in kf.split(Z):
        Z_train, Z_test = Z[train_idx, :], Z[test_idx, :]
        X_train, X_test = X[train_idx, :], X[test_idx, :]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        # Train r0(Z) to predict X
        r = clone(r0_learner)
        r.fit(Z_train, X_train)
        r_hats.append(r)
        X_resid[test_idx, :] = X_test - r.predict(Z_test)

        # Train g0(Z) to predict Y
        g = clone(g0_learner)
        g.fit(Z_train, Y_train)
        g_hats.append(g)
        Y_resid[test_idx] = Y_test - g.predict(Z_test)

    # ======= Final Stage: Estimate beta in f(X) = X @ beta =======
    f = LinearRegression(fit_intercept=False)
    f.fit(X_resid, Y_resid)
    beta_hat = f.coef_

    # ======= Variance-Covariance Matrix =======
    N, P = X.shape
    XtX_inv = np.linalg.inv(X_resid.T @ X_resid / N)
    residuals = Y_resid - f.predict(X_resid)
    meat = (X_resid.T * residuals) @ X_resid / N
    vcov = XtX_inv @ meat @ XtX_inv / N
    assert vcov.shape == (P, P)
    assert beta_hat.shape == (P, )
    
    return {
        "r_hats": r_hats,
        "g_hats": g_hats,
        "f_hat": f,
        "beta_hat": beta_hat,
        "vcov": vcov
    }
    
    

@timeit
def train_model(
    *, 
    data_generator_param: DataGeneratorParam, 
    train_data_param: TrainDataParam,
    which_model: Literal['rhat', 'dml'],
    output_directory_uri: str,
    r0_learner_name: str,
    g0_learner_name: str = None,
    r0_learner,
    g0_learner = None,
    seed: int=999, # seed for train data generation
    fresh: bool=False
):
    model_directory_uri = get_model_directory_uri(
        data_generator_param = data_generator_param,
        train_data_param = train_data_param,
        which_model = which_model,
        r0_learner_name = r0_learner_name,
        g0_learner_name = g0_learner_name,
        output_directory_uri = output_directory_uri
    )
    model_uri = os.path.join(model_directory_uri, "trained_model.pkl")
    model_metadata_uri = os.path.join(model_directory_uri, "metadata.pkl")
    
    if os.path.exists(model_uri) and os.path.exists(model_metadata_uri) and not fresh:
        return read_pickle(model_uri), model_directory_uri
    
    data_generator = DataGenerator(data_generator_param=data_generator_param)
    Z, X, Y = data_generator.generate_train_data(train_data_param, seed)
    
    if which_model == 'rhat':
        rhat = clone(r0_learner)
        trained_model = rhat.fit(Z, X)
    elif which_model == 'dml':
        trained_model = train_dml(
            Z = Z, 
            X = X, 
            Y = Y, 
            r0_learner = r0_learner,
            g0_learner = g0_learner,
            kfolds = 2,
        )
    else:
        raise ValueError(f"Invalid model: {which_model}")
    write_pickle(trained_model, model_uri)

     
    model_metadata = {
        "DataGeneratorParam": data_generator_param.to_dict(),
        "TrainDataParam": train_data_param.to_dict(),
        "model": which_model,
        "r0_learner": r0_learner_name,
        "g0_learner": g0_learner_name
    }
    write_pickle(model_metadata, model_metadata_uri)
    
    return trained_model, model_directory_uri
    
    

    
if __name__ == '__main__':
    from true_models import generate_true_models, generate_bias_models
    from data_generator import DataGenerator, RealDataParam
    from learner import build_learner
    import numpy as np
    from sklearn.metrics import mean_squared_error
    
    
    Q = 100   # Z in R^Q
    P = 100   # X in R^P
              # Y in R

    r0, g0, f0 = generate_true_models(Q, P)
    nn_bias_1, nn_bias_2 = generate_bias_models(Q, P)
    
    data_generator_param = DataGeneratorParam(p=P, q=Q, r0=r0, g0=g0, f0=f0)
    data_generator = DataGenerator(data_generator_param)
    
    train_data_param = TrainDataParam(n_train=50000)
    train_data = data_generator.generate_train_data(train_data_param, seed=999)
    
    real_data_param = RealDataParam(bias_func=nn_bias_1, bias_scale=0, n_real=100)
    real_data = data_generator.generate_real_data(real_data_param, seed=1)
    
    assert not np.array_equal(train_data.Z[:100], real_data.Z)


#     OUTPUT_DIRECTORY_URI = "/u/home/y/yqg36/Mediation/results"
    OUTPUT_DIRECTORY_URI = "/u/scratch/y/yqg36/Mediation/results"
    
    
    ols_rhat, _ = train_model(
        data_generator_param = data_generator_param, 
        train_data_param = train_data_param, 
        which_model = "rhat",
        output_directory_uri = OUTPUT_DIRECTORY_URI,
        r0_learner_name = "linear",
        r0_learner = build_learner(model_type='ols'), 
        seed = 999,
#         fresh = True
    )
    print(f'ols train MSE: {mean_squared_error(ols_rhat.predict(train_data.Z), train_data.X)}')
    print(f'ols test MSE: {mean_squared_error(ols_rhat.predict(real_data.Z), real_data.X)}')
    
    rf_rhat, _ = train_model(
        data_generator_param = data_generator_param, 
        train_data_param = train_data_param, 
        which_model = "rhat",
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
        seed = 999,
#         fresh = True
    )
    print(f'rf train MSE: {mean_squared_error(rf_rhat.predict(train_data.Z), train_data.X)}')
    print(f'rf test MSE: {mean_squared_error(rf_rhat.predict(real_data.Z), real_data.X)}')
    
    krr_rhat, _ = train_model(
        data_generator_param = data_generator_param, 
        train_data_param = train_data_param, 
        which_model = "rhat",
        output_directory_uri = OUTPUT_DIRECTORY_URI,
        r0_learner_name = "kernel",
        r0_learner = build_learner(model_type='krr'), 
        seed = 999,
#         fresh = True
    )
    print(f'krr train MSE: {mean_squared_error(krr_rhat.predict(train_data.Z), train_data.X)}')
    print(f'krr test MSE: {mean_squared_error(krr_rhat.predict(real_data.Z), real_data.X)}')
    
    xgb_rhat, _ = train_model(
        data_generator_param = data_generator_param, 
        train_data_param = train_data_param, 
        which_model = "rhat",
        output_directory_uri = OUTPUT_DIRECTORY_URI,
        r0_learner_name = "xgboost",
        r0_learner = build_learner(model_type='xgb', output_dim=P), 
        seed = 999,
#         fresh = True
    )
    print(f'xgb train MSE: {mean_squared_error(xgb_rhat.predict(train_data.Z), train_data.X)}')
    print(f'xgb test MSE: {mean_squared_error(xgb_rhat.predict(real_data.Z), real_data.X)}')
    
    mlp_rhat, _ = train_model(
        data_generator_param = data_generator_param, 
        train_data_param = train_data_param, 
        which_model = "rhat",
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
        seed = 999,
#         fresh = True
    )
    print(f'mlp train MSE: {mean_squared_error(mlp_rhat.predict(train_data.Z), train_data.X)}')
    print(f'mlp test MSE: {mean_squared_error(mlp_rhat.predict(real_data.Z), real_data.X)}')



    ols_dml, _ = train_model(
        data_generator_param = data_generator_param, 
        train_data_param = train_data_param,
        which_model = 'dml',
        output_directory_uri = OUTPUT_DIRECTORY_URI,
        r0_learner_name = 'linear',
        g0_learner_name = 'linear',
        r0_learner = build_learner(model_type='ols'),
        g0_learner = build_learner(model_type='ols'),
        seed = 999,
#         fresh = True
    )
    print(f"ols mean beta_hat: {np.mean(ols_dml['beta_hat'])}")
    
    
    rf_dml, _ = train_model(
        data_generator_param = data_generator_param, 
        train_data_param = train_data_param,
        which_model = 'dml',
        output_directory_uri = OUTPUT_DIRECTORY_URI,
        r0_learner_name = 'random_forest',
        g0_learner_name = 'random_forest',
        r0_learner = build_learner(
            model_type='rf', 
            output_dim=P, 
            n_estimators=100, 
            # max_depth=10,
            max_features='sqrt',
            n_jobs=-1,
        ),
        g0_learner = build_learner(
            model_type='rf', 
            output_dim=1, 
            n_estimators=100, 
            # max_depth=10,
            max_features='sqrt',
            n_jobs=-1,
        ),
        seed = 999,
#         fresh = True
    )
    print(f"rf mean beta_hat: {np.mean(rf_dml['beta_hat'])}")
    
    
    krr_dml, _ = train_model(
        data_generator_param = data_generator_param, 
        train_data_param = train_data_param,
        which_model = 'dml',
        output_directory_uri = OUTPUT_DIRECTORY_URI,
        r0_learner_name = 'kernel',
        g0_learner_name = 'kernel',
        r0_learner = build_learner(model_type='krr'),
        g0_learner = build_learner(model_type='krr'),
        seed = 999,
#         fresh = True
    )
    print(f"krr mean beta_hat: {np.mean(krr_dml['beta_hat'])}")
    
    
    xgb_dml, _ = train_model(
        data_generator_param = data_generator_param, 
        train_data_param = train_data_param,
        which_model = 'dml',
        output_directory_uri = OUTPUT_DIRECTORY_URI,
        r0_learner_name = 'xgboost',
        g0_learner_name = 'xgboost',
        r0_learner = build_learner(model_type='xgb', output_dim=P),
        g0_learner = build_learner(model_type='xgb', output_dim=1),
        seed = 999,
#         fresh = True
    )
    print(f"xgb mean beta_hat: {np.mean(xgb_dml['beta_hat'])}")
    
    
    mlp_dml, _ = train_model(
        data_generator_param = data_generator_param, 
        train_data_param = train_data_param,
        which_model = 'dml',
        output_directory_uri = OUTPUT_DIRECTORY_URI,
        r0_learner_name = 'neural_net_128x128_1000_64',
        g0_learner_name = 'neural_net_256x256x256x256_1000_64',
        r0_learner = build_learner(
            model_type = 'mlp', 
            input_dim = P,  
            output_dim = Q,
            hidden_layers = [128, 128],
            epochs = 1000,
            batch_size = 64
        ), 
        g0_learner = build_learner(
            model_type = 'mlp', 
            input_dim = Q,  
            output_dim = 1,
            hidden_layers = [256, 256, 256, 256],
            epochs = 1000,
            batch_size = 64
        ), 
        seed = 999,
#         fresh = True
    )
    print(f"mlp mean beta_hat: {np.mean(mlp_dml['beta_hat'])}")