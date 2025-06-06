from mlp import TorchMLP, init_all_weights
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from xgboost import XGBRegressor
from skorch import NeuralNetRegressor
import torch
import numpy as np
from utils import timeit, set_seed
from functools import partial

    
    
    
class MyNetRegressor(NeuralNetRegressor):
    def fit(self, X, y=None, **fit_params):
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return super().fit(X, y, **fit_params)
    
    def predict(self, X):
        y_pred = super().predict(X)
        if y_pred.ndim == 2 and y_pred.shape[1] == 1:
            y_pred = y_pred.ravel()
        return y_pred
    
    
def build_learner(model_type: str, memory=None, **kwargs):

    if model_type == 'ols':
        return Pipeline([
            ('scaler', MinMaxScaler()),
            ('estimator', LinearRegression(**kwargs))
        ], memory=memory)

    elif model_type == 'rf':
        output_dim = kwargs.pop('output_dim', 1)
        base_model = RandomForestRegressor(random_state=0, **kwargs)
        model = MultiOutputRegressor(base_model) if output_dim >= 2 else base_model
        return Pipeline([
            ('scaler', MinMaxScaler()),
            ('estimator', model)
        ], memory=memory)

    elif model_type == 'krr':
        gamma = kwargs.pop('gamma', 0.1)
        alpha = kwargs.pop('alpha', 1.0)
        nystroem_kwargs = {
            'kernel': 'rbf', 'gamma': gamma, 'random_state': 0
        }
        for key in {'n_components', 'degree', 'coef0'}:
            if key in kwargs:
                nystroem_kwargs[key] = kwargs.pop(key)
        ridge_kwargs = {'alpha': alpha, 'random_state': 0}
        return Pipeline([
            ('scaler', MinMaxScaler()),
            ('feature_map', Nystroem(**nystroem_kwargs)),
            ('ridge', Ridge(**ridge_kwargs))
        ], memory=memory)

    elif model_type == 'xgb':
        output_dim = kwargs.pop('output_dim', 1)
        base_model = XGBRegressor(**kwargs)
        model = MultiOutputRegressor(base_model) if output_dim >= 2 else base_model
        return Pipeline([
            ('scaler', MinMaxScaler()),
            ('estimator', model)
        ], memory=memory)

    elif model_type == 'mlp':
        input_dim = kwargs.pop('input_dim')
        output_dim = kwargs.pop('output_dim')
        hidden_layers = kwargs.pop('hidden_layers')
        epochs = kwargs.pop('epochs', 200)
        batch_size = kwargs.pop('batch_size', 32)
        lr = kwargs.pop('lr', 1e-3)
        
        y_transform = FunctionTransformer(
            func=lambda y: y.reshape(-1, 1) if y.ndim == 1 else y,
            validate=False
        )
        model = MyNetRegressor(
            module=TorchMLP,
            module__input_dim=input_dim,
            module__output_dim=output_dim,
            module__hidden_layers=hidden_layers,
            criterion=torch.nn.MSELoss,
            optimizer=torch.optim.Adam,
            verbose=0,
            max_epochs=epochs,
            batch_size=batch_size,
            train_split=None,
            lr=lr,
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )     
        set_seed(0)
        model.initialize()
        init_all_weights(model.module_, generator=False, generator_seed=0)
        return Pipeline([
            ('scaler', MinMaxScaler()),
            ('estimator', model)
        ], memory=memory)
        
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
            
          

        
if __name__ == '__main__':
    import numpy as np
    from sklearn.metrics import mean_squared_error

    Q = 100   # Z in R^Q
    P = 100   # X in R^P
              # Y in R
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((100, Q))
    mlp_generator_ZtoX = TorchMLP(
        input_dim=P,
        output_dim=Q,
        hidden_layers=[32], 
    )
    init_all_weights(mlp_generator_ZtoX)
    X = mlp_generator_ZtoX.predict_numpy(Z)
    mlp_generator_XtoY = TorchMLP(
        input_dim=Q,
        output_dim=1,
        hidden_layers=[32], 
    )
    init_all_weights(mlp_generator_XtoY)
    Y = mlp_generator_XtoY.predict_numpy(X)

    assert X.shape == ((100, P) if P != 1 else (100,))
    assert Y.shape == (100,)
    print(f'{Z.mean()=}')
    print(f'{X.mean()=}')
    print(f'{Y.mean()=}')
    
    ols_learner = build_learner(model_type='ols')
    X_ols = ols_learner.fit(Z, X).predict(Z)
    print(f'ols: {mean_squared_error(X_ols, X)}')
    Y_ols = ols_learner.fit(X, Y).predict(X)
#     print(Y_ols.shape)
    print(f'ols: {mean_squared_error(Y_ols, Y)}')

    rf_learner1 = build_learner(
        model_type='rf', 
        output_dim=P, 
        n_estimators=100, 
        max_depth=10,
        max_features='sqrt',
        n_jobs=-1,
    )
    rf_learner2 = build_learner(model_type='rf', output_dim=1)
    X_rf = rf_learner1.fit(Z, X).predict(Z)
    print(f'rf: {mean_squared_error(X_rf, X)}')
    Y_rf = rf_learner2.fit(X, Y).predict(X)
    print(f'rf: {mean_squared_error(Y_rf, Y)}')

    krr_learner = build_learner(model_type='krr')
    X_krr = krr_learner.fit(Z, X).predict(Z)
    print(f'krr: {mean_squared_error(X_krr, X)}')
    Y_krr = krr_learner.fit(X, Y).predict(X)
    print(f'krr: {mean_squared_error(Y_krr, Y)}')

    xgb_learner1 = build_learner(model_type='xgb', output_dim=P)
    xgb_learner2 = build_learner(model_type='xgb', output_dim=1)
    X_xgb = xgb_learner1.fit(Z, X).predict(Z)
    print(f'xgb: {mean_squared_error(X_xgb, X)}')
    Y_xgb = xgb_learner2.fit(X, Y).predict(X)
    print(f'xgb: {mean_squared_error(Y_xgb, Y)}')

    mlp_learner_ZtoX = build_learner(
        model_type = 'mlp', 
        input_dim = P, 
        output_dim = Q, 
        hidden_layers = [64],
        epochs = 200,
        batch_size=32,
    )
    mlp_learner_XtoY = build_learner(
        model_type = 'mlp', 
        input_dim = Q, 
        output_dim = 1, 
        hidden_layers = [64],
        epochs = 200,
        batch_size=32,
    )
    X_mlp = mlp_learner_ZtoX.fit(Z, X).predict(Z)
#     print(X_mlp.shape)
    print(f'mlp: {mean_squared_error(X_mlp, X)}')
    Y_mlp = mlp_learner_XtoY.fit(X, Y).predict(X)
#     print(Y_mlp.shape)
    print(f'mlp: {mean_squared_error(Y_mlp, Y)}')

        
        
        