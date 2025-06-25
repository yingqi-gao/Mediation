import functools
import hashlib
import json
import os
import pickle
import random
import time
from typing import Optional

import numpy as np
import torch

from data_generator import (
    DataGeneratorParam,
    TrainDataParam,
    RealDataParam,
    ExpandedDataParam,
)


def timeit(func):
    """
    Decorator that measures and prints the execution time of the decorated function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__!r} executed in {end - start:.6f}s")
        return result
    return wrapper


def get_dict_hash(data: dict) -> str:
    encoded_data = json.dumps(data, sort_keys=True).encode('utf-8')
    hash_object = hashlib.md5(encoded_data)
    return hash_object.hexdigest()[:6]


def get_model_directory_uri(
    *,
    data_generator_param: DataGeneratorParam,
    train_data_param: TrainDataParam,
    r0_learner_name: str,
    g0_learner_name: Optional[str] = None,
    output_directory_uri: str,
) -> str:
    data_generator_param_dict = data_generator_param.to_dict()
    train_data_param_dict = train_data_param.to_dict()

    return os.path.join(
        output_directory_uri,
        f"data_generator_param={get_dict_hash(data_generator_param_dict)}",
        f"train_data_param={get_dict_hash(train_data_param_dict)}",
        f"r0_learner={r0_learner_name}"
        f"g0_learner={g0_learner_name}",
    )


def get_r0_CI_directory_uri(
    *,
    real_data_param: RealDataParam,
    expanded_data_param: ExpandedDataParam,
    model_directory_uri: str,
) -> str:
    real_data_param_dict = real_data_param.to_dict()
    expanded_data_param_dict = expanded_data_param.to_dict()

    return os.path.join(
        model_directory_uri,
        f"real_data_param={get_dict_hash(real_data_param_dict)}",
        f"expanded_data_param={get_dict_hash(expanded_data_param_dict)}",
    )


def read_pickle(uri):
    import warnings
    warnings.simplefilter("ignore", category=FutureWarning)
    with open(uri, 'rb') as file:
        return pickle.load(file)


def write_pickle(data, uri):
    dir_uri = os.path.dirname(uri)
    os.makedirs(dir_uri, exist_ok=True)
    with open(uri, "wb") as file:
        pickle.dump(data, file)


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
