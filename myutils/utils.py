import functools
import time
import json
import hashlib
import pickle
import os




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



def read_pickle(uri):
    with open(uri, 'rb') as file:
        return pickle.load(file)

    
def write_pickle(data, uri):
    dir_uri = os.path.dirname(uri)
    os.makedirs(dir_uri, exist_ok=True)
    with open(uri, "wb") as file:
        pickle.dump(data, file)
        

def set_seed(seed=0):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False