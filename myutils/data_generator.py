from abc import ABC
from dataclasses import asdict, dataclass
from typing import Callable, NamedTuple, Optional

import numpy as np


##################################
### Data Generating Parameters ###
##################################
@dataclass
class DataParam(ABC):

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


@dataclass
class TrainDataParam(DataParam):
    n_train: int


@dataclass
class RealDataParam(DataParam):
    bias_func: Callable
    bias_scale: float
    n_real: int

    def to_dict(self) -> dict:
        data = super().to_dict()
        data["bias_func"] = data["bias_func"].__name__
        return data

    @classmethod
    def from_dict(cls, data):
        obj = super().from_dict(data)
        bias_func_name = data["bias_func"]
        obj.bias_func = globals()[bias_func_name]
        return obj


@dataclass
class ExpandedDataParam(DataParam):
    n_expanded: int
    r_expanded: float


@dataclass
class DataGeneratorParam(DataParam):
    q: int
    p: int

    r0: Callable
    g0: Callable
    f0: Callable

    def to_dict(self) -> dict:
        data = super().to_dict()
        data["r0"] = data["r0"].__name__
        data["g0"] = data["g0"].__name__
        data["f0"] = data["f0"].__name__
        return data

    @classmethod
    def from_dict(cls, data):
        obj = super().from_dict(data)
        obj.r0 = globals()[data["r0"]]
        obj.g0 = globals()[data["g0"]]
        obj.f0 = globals()[data["f0"]]
        return obj


######################
### Generated Data ###
######################
class GeneratedData(NamedTuple):
    Z: np.ndarray
    X: np.ndarray
    Y: np.ndarray


class TargetPoint(NamedTuple):
    Z0: np.ndarray
    X0: np.ndarray
    Y0: np.ndarray
    wZ0: float


######################
### Data Generator ###
######################
@dataclass
class DataGenerator:
    data_generator_param: DataGeneratorParam

    def __post_init__(self):
        self.q = self.data_generator_param.q
        self.p = self.data_generator_param.p

        self.r0 = self.data_generator_param.r0
        self.g0 = self.data_generator_param.g0
        self.f0 = self.data_generator_param.f0

    def _generate_data(
        self,
        *,
        seed: int,
        n: int,
        noise: float = 1,
        bias_func: Optional[Callable] = None,
        bias_scale: Optional[float] = None,
    ) -> GeneratedData:

        rng = np.random.default_rng(seed)

        Z = rng.standard_normal((n, self.q))
        V = noise * rng.standard_normal((n, self.p))
        if self.p == 1:
            V = V.squeeze()
        X = self.r0(Z) + V
        # print(f"{Z.shape=}")
        # print(f"{V.shape=}")
        # print(f"{self.r0(Z).shape=}")

        if bias_func is not None and bias_scale is not None:
            X += bias_scale * bias_func(Z)

        U = noise * rng.standard_normal(n)
        Y = self.g0(Z) + self.f0(X) + U

        return GeneratedData(Z, X, Y)

    def generate_train_data(
        self, train_data_param: TrainDataParam, seed: int
    ) -> GeneratedData:
        return self._generate_data(seed=seed, n=train_data_param.n_train)

    def generate_real_data(
        self, real_data_param: RealDataParam, seed: int
    ) -> GeneratedData:

        return self._generate_data(
            seed=seed,
            n=real_data_param.n_real,
            bias_func=real_data_param.bias_func,
            bias_scale=real_data_param.bias_scale,
        )

    def generate_target_point(
        self, real_data_param: RealDataParam, seed: int
    ) -> TargetPoint:

        Z0, X0, Y0 = self._generate_data(
            seed=seed,
            n=1,
            bias_func=real_data_param.bias_func,
            bias_scale=real_data_param.bias_scale,
        )
        wZ0 = self.f0(self.r0(Z0))
        assert wZ0.shape == (1,)

        return TargetPoint(Z0, X0, Y0, wZ0[0])

    def generate_expanded_data(
        self, expanded_data_param: ExpandedDataParam, Z0: np.ndarray, seed: int
    ) -> np.ndarray:

        rng = np.random.default_rng(seed)

        # Sample W ~ N(0, 1)
        W = rng.standard_normal((expanded_data_param.n_expanded, self.q))

        # L2-normalize each row (along axis=1)
        W_norm = W / np.linalg.norm(W, axis=1, keepdims=True)

        # Create Zp
        Zp = Z0 + expanded_data_param.r_expanded * W_norm

        return Zp


if __name__ == "__main__":
    from true_models import generate_true_models, generate_bias_models

    ## 1-dim tests
    Q, P = 1, 1

    r0, g0, f0 = generate_true_models(Q, P)
    nn_bias_1, nn_bias_2 = generate_bias_models(Q, P)

    data_generator_param = DataGeneratorParam(p=P, q=Q, r0=r0, g0=g0, f0=f0)

    train_data_param = TrainDataParam(n_train=100)
    real_data_param = RealDataParam(bias_func=nn_bias_1, bias_scale=0, n_real=100)
    expanded_data_param = ExpandedDataParam(100, 0.1)

    data_generator = DataGenerator(data_generator_param)

    train_data = data_generator.generate_train_data(train_data_param, seed=0)
    real_data = data_generator.generate_real_data(real_data_param, seed=0)
    target_point = data_generator.generate_target_point(real_data_param, seed=0)
    print(target_point.wZ0)
    Zp = data_generator.generate_expanded_data(
        expanded_data_param, target_point.Z0, seed=0
    )
    print(np.mean(train_data.Z), np.mean(train_data.X), np.mean(train_data.Y))

    # Assertions for shape
    assert train_data.Z.shape == (train_data_param.n_train, Q)
    print(train_data.X.shape)
    assert train_data.X.shape == (
        (train_data_param.n_train, P) if P != 1 else (train_data_param.n_train,)
    )

    # Assertions for equality (NumPy)
    assert np.array_equal(train_data.Z, real_data.Z)
    assert np.array_equal(train_data.X, real_data.X)

    # Assertions for inequality
    assert not np.array_equal(train_data.Z[0], target_point.Z0)
    assert not np.array_equal(train_data.X[0], target_point.X0)

    assert not np.array_equal(train_data.Z, Zp)

    ## multi-dim tests
    Q, P = 20, 10

    r0, g0, f0 = generate_true_models(Q, P)
    nn_bias_1, nn_bias_2 = generate_bias_models(Q, P)

    data_generator_param = DataGeneratorParam(p=P, q=Q, r0=r0, g0=g0, f0=f0)

    train_data_param = TrainDataParam(n_train=100)
    real_data_param = RealDataParam(bias_func=nn_bias_1, bias_scale=0, n_real=100)
    expanded_data_param = ExpandedDataParam(100, 0.1)

    data_generator = DataGenerator(data_generator_param)

    train_data = data_generator.generate_train_data(train_data_param, seed=0)
    real_data = data_generator.generate_real_data(real_data_param, seed=0)
    target_point = data_generator.generate_target_point(real_data_param, seed=0)
    print(target_point.wZ0)
    Zp = data_generator.generate_expanded_data(
        expanded_data_param, target_point.Z0, seed=0
    )
    print(np.mean(train_data.Z), np.mean(train_data.X), np.mean(train_data.Y))

    # Assertions for shape
    assert train_data.Z.shape == (train_data_param.n_train, Q)
    assert train_data.X.shape == (
        (train_data_param.n_train, P) if P != 1 else (train_data_param.n_train,)
    )

    # Assertions for equality (NumPy)
    assert np.array_equal(train_data.Z, real_data.Z)
    assert np.array_equal(train_data.X, real_data.X)

    # Assertions for inequality
    assert not np.array_equal(train_data.Z[0], target_point.Z0)
    assert not np.array_equal(train_data.X[0], target_point.X0)

    assert not np.array_equal(train_data.Z, Zp)
