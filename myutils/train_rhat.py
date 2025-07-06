import os

from sklearn.base import clone
from sklearn.metrics import mean_squared_error

from data_generator import DataGenerator, DataGeneratorParam, TrainDataParam
from utils import timeit, read_file, write_file


@timeit
def train_rhat(
    *,
    data_generator_param: DataGeneratorParam,
    train_data_param: TrainDataParam,
    model_directory_uri: str,
    learner_name,
    learner,
    seed: int = 999,  # seed for train data generation
    fresh: bool = False,
):
    # Get paths for storing trained model and metadata
    model_uri = os.path.join(model_directory_uri, "trained_model.pkl")
    model_metadata_uri = os.path.join(model_directory_uri, "metadata.pkl")

    # Generate data for model training
    data_generator = DataGenerator(data_generator_param=data_generator_param)
    Z, X, Y = data_generator.generate_train_data(train_data_param, seed)

    # Check if the model has already been trained and stored properly
    if os.path.exists(model_uri) and os.path.exists(model_metadata_uri) and not fresh:
        print("Reading rhat...")
        rhat = read_file(model_uri)

    # If not
    else:
        print("Training rhat...")
        # Copy the learner, train and store the model
        r0_learner = clone(learner)
        rhat = r0_learner.fit(Z, X)
        write_file(rhat, model_uri)

        # Gather and store the metadata
        model_metadata = {
            "DataGeneratorParam": data_generator_param.to_dict(),
            "TrainDataParam": train_data_param.to_dict(),
            "r0_learner": learner_name,
            "g0_learner": None,
        }
        write_file(model_metadata, model_metadata_uri)

    # Print the training MSE
    print(f"{learner_name} training MSE = {mean_squared_error(X, rhat.predict(Z))}")

    return rhat


if __name__ == "__main__":
    import numpy as np

    from true_models import generate_true_models, generate_bias_models
    from data_generator import RealDataParam
    from learner import build_learner
    from utils import get_model_directory_uri

    ## 1-dim tests
    Q, P = 1, 1
    N_TRAIN = 100
    N_REAL = 100
    SEED = 999

    r0, g0, f0 = generate_true_models(Q, P)
    nn_bias_1, _ = generate_bias_models(Q, P)

    data_generator_param = DataGeneratorParam(p=P, q=Q, r0=r0, g0=g0, f0=f0)
    data_generator = DataGenerator(data_generator_param)

    train_data_param = TrainDataParam(n_train=N_TRAIN)
    train_data = data_generator.generate_train_data(train_data_param, seed=SEED)

    real_data_param = RealDataParam(bias_func=nn_bias_1, bias_scale=0, n_real=N_REAL)
    real_data = data_generator.generate_real_data(real_data_param, seed=1)

    assert not np.array_equal(train_data.Z[:100], real_data.Z)

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
        )

    ## multi-dim tests
    Q, P = 20, 10
    N_TRAIN = 10000
    N_REAL = 100
    SEED = 999

    r0, g0, f0 = generate_true_models(Q, P)
    nn_bias_1, _ = generate_bias_models(Q, P)

    data_generator_param = DataGeneratorParam(p=P, q=Q, r0=r0, g0=g0, f0=f0)
    data_generator = DataGenerator(data_generator_param)

    train_data_param = TrainDataParam(n_train=N_TRAIN)
    train_data = data_generator.generate_train_data(train_data_param, seed=SEED)

    real_data_param = RealDataParam(bias_func=nn_bias_1, bias_scale=0, n_real=N_REAL)
    real_data = data_generator.generate_real_data(real_data_param, seed=1)

    assert not np.array_equal(train_data.Z[:100], real_data.Z)

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
        )
