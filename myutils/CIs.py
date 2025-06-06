from data_generator import GeneratedData
import numpy as np
import scipy
from sklearn.metrics import mean_squared_error

def construct_r0_CI(
    *,
    real_data: GeneratedData,
    target_point: GeneratedData,
    expanded_data: np.ndarray,
    r_hat, 
    alpha = 0.05,
) -> dict:
        
    Z, X, Y = real_data
    p = X.shape[1]
    Z0, X0, Y0 = target_point
    Zp = expanded_data

    # Retrieve r_hat using the first cross-fitting result & make predictions
    X_hat = r_hat.predict(Z)
    assert X_hat.shape == X.shape
    X_hatp = r_hat.predict(Zp)
    assert X_hatp.shape == X.shape
    
    # Find the midpoint
    mean_X_hatp = X_hatp.mean(axis=0)
    assert mean_X_hatp.shape == (p, )
    delta = (X_hat - X).mean(axis=0)
    assert delta.shape == (p, )
    midpoint = mean_X_hatp - delta   
#     print(f"{mean_X_hatp=}")
#     print(f"{delta=}")
    print(f"{X0=}")
    print(f"{midpoint=}")
    
    # Find the width
    sigma2_1 = np.var(X_hat - X, axis=0, ddof=0)
    assert sigma2_1.shape == (p, )
    sigma2_2 = np.var(X_hatp, axis=0, ddof=0)
    assert sigma2_2.shape == (p, )
    z_crit = scipy.stats.norm.ppf(1 - alpha / (2 * X.shape[1] if X.shape[1]>1 else 1))
    w_theta = z_crit * np.sqrt(sigma2_1 + sigma2_2)
    assert w_theta.shape == (p, )
    print(f"MSE: {mean_squared_error(X, X_hat)}")
    print(f"{sigma2_1=}")
    print(f"{sigma2_2=}")
    print(f"{w_theta=}")
    
    # Find the endpoints
    lower = midpoint - w_theta
    upper = midpoint + w_theta
    
    # Construct the CI
    r0_CI = {
        "lower": lower,
        "upper": upper,
        "covers?": (np.all(lower < X0)) and (np.all(X0 < upper))
    }
    
    return r0_CI



def construct_w_CI(
    *,
    trained_models: dict,
    real_data: GeneratedData,
    target_point: GeneratedData,
    expanded_data: np.ndarray,
    beta_0: np.ndarray,
    r0,
    trained_models,
    alpha = 0.05, 
) -> dict:
    
    Z0, X0, Y0 = target_point
    q = Z0.shape[1]
    truth = r0(Z0) @ beta_0
    
    r0_CI_lowers = []
    r0_CI_uppers = []
    for rhat in trained_models.models['ml_m']:
        r0_CI = construct_r0_CI(
            real_data=real_data,
            target_point=target_point,
            expanded_data=expanded_data,
            r_hat=r_hat, 
        )
        r0_CI_lowers.append(r0_CI['lower'])
        r0_CI_uppers.append(r0_CI['upper'])
    tau_lower = np.minimum(r0_CI_lowers)
    assert tau_lower.shape == (q, )
    tau_upper = np.maximum(r0_CI_uppers)
    assert tau_upper.shape == (q, )
    
    beta_hat = trained_models.coef
    w_taus = [
        np.dot(tau_lower, beta_hat) - z_crit * sqrt(tau_lower.reshape(1, -1)@)
    
    z_crit = scipy.stats.norm.ppf(1 - alpha / 4)
    lower = np.dot(tau_lower, beta_hat) - z_crit * np.sqrt(tau_lower.reshape(1, -1) @ Sigma_hat @ tau_lower.reshape(-1, 1))
    assert lower.shape == Y0.shape
    upper = np.dot(tau_upper, beta_hat) + z_crit * np.sqrt(tau_upper.reshape(1, -1) @ Sigma_hat @ tau_upper.reshape(-1, 1))
    assert upper.shape == Y0.shape
    
    # Construct the CI
    w_CI = {
        "lower": lower,
        "upper": upper,
        "covers?": (lower < truth) and (truth < upper)
    }
    
    return f0_CI


if __name__ == '__main__':
    from true_models import generate_true_models, generate_bias_models
    from data_generator import DataGeneratorParam, TrainDataParam, RealDataParam, ExpandedDataParam, DataGenerator
    from learner import build_learner
    import numpy as np
    from train_model import train_model
    
    
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
    
    target_point = data_generator.generate_target_point(real_data_param, seed=0)
    
    expanded_data_param = ExpandedDataParam(100, 0.1)
    expanded_data = data_generator.generate_expanded_data(
        expanded_data_param, 
        target_point.Z, 
        seed=0
    )
    
    OUTPUT_DIRECTORY_URI = "/u/home/y/yqg36/Mediation/results"
    krr_trained_models = train_models(
        data_generator_param = data_generator_param, 
        train_data_param = train_data_param, 
        r0_learner_name = "kernel",
        g0_learner_name = "kernel",
        output_directory_uri = OUTPUT_DIRECTORY_URI,
        r0_learner = build_learner(model_type='krr'), 
        g0_learner = build_learner(model_type='krr'),
        seed = 999
    )
    krr_r0_CI = construct_r0_CI(
        real_data=real_data,
        target_point=target_point,
        expanded_data=expanded_data,
        r_hat=krr_trained_models["rhat_list"][0]
    )
    krr_f0_CI = construct_f0_CI(
        trained_models = krr_trained_models,
        r0_CI = krr_r0_CI, 
        target_point = target_point
    )
    print(krr_f0_CI["covers?"])
    
    nn_trained_models = train_models(
        data_generator_param = data_generator_param, 
        train_data_param = train_data_param, 
        r0_learner_name = "neural net 256x256",
        g0_learner_name = "neural net 128",
        output_directory_uri = OUTPUT_DIRECTORY_URI,
        r0_learner = build_learner(
            model_type = 'mlp', 
            input_dim = Q,  
            output_dim = P,
            hidden_layers = [256, 256],
            epochs = 200,
            batch_size = 32
        ),
        g0_learner = build_learner(
            model_type = 'mlp', 
            input_dim = Q,  
            output_dim = 1,
            hidden_layers = [128],
            epochs = 200,
            batch_size = 32
        ),
        seed = 999
    )
    nn_r0_CI = construct_r0_CI(
        real_data=real_data,
        target_point=target_point,
        expanded_data=expanded_data,
        r_hat=nn_trained_models["rhat_list"][0]
    )
    nn_f0_CI = construct_f0_CI(
        trained_models = nn_trained_models,
        r0_CI = nn_r0_CI, 
        target_point = target_point
    )
    print(nn_f0_CI["covers?"])
    