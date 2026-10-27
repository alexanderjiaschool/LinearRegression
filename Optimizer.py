import numpy as np


class Optimizer:
    def __init__(self, learning_rate: float, dimension: int):
        if Optimizer.isValidHyperParam(learning_rate, dimension):
            self._learning_rate = learning_rate
            self._dimension = dimension
        self._step_count = 0

    @staticmethod
    def isValidHyperParam(learning_rate, dimension):
        if learning_rate <= 0:
            raise ValueError("Invalid learning rate")
        if not isinstance(dimension, int):
            raise ValueError("Invalid dimension")
        return True

    def checkShape(self, param: dict, gradients: dict):
        if len(param["weight"]) == self._dimension == len(gradients["weight"]):
            return
        raise ValueError("Optimizer Error: Shape doesn't match")

    def step(self, param: dict, gradients: dict):
        self.checkShape(param, gradients)
        self._step_count += 1
        param["weight"] -= self._learning_rate * gradients["weight"]
        param["bias"] -= self._learning_rate * gradients["bias"]

    def updateHyperParam(self, learning_rate: float, dimension: int):
        if Optimizer.isValidHyperParam(learning_rate, dimension):
            self._learning_rate = learning_rate
            self._dimension = dimension

    def __str__(self):
        return f"{self.__class__.__name__}:\nlearning_rate={self._learning_rate}\ndimension={self._dimension}"


class Adam(Optimizer):
    def __init__(
        self,
        learning_rate: float,
        dimension: int,
        decay_for_weighted_moving_average_gradient: float = 0.9,
        decay_for_weighted_moving_average_squared_gradient: float = 0.99,
    ):
        super().__init__(learning_rate, dimension)

        if Adam.isValidAdamParam(
            decay_for_weighted_moving_average_gradient,
            decay_for_weighted_moving_average_squared_gradient,
        ):
            self.decay_for_weighted_average_gradient = (
                decay_for_weighted_moving_average_gradient
            )
            self.decay_for_weighted_average_squared_gradient = (
                decay_for_weighted_moving_average_squared_gradient
            )

        self._weighted_average_gradient = {"weight": np.zeros(dimension), "bias": 0.0}
        self._weighted_average_squared_gradient = {
            "weight": np.zeros(dimension),
            "bias": 0.0,
        }

        # an extremely small number used to prevent division by zero error
        self.padding = 1e-8

    def __str__(self):
        return (
            super().__str__()
            + f"\ndecay_for_weighted_average_gradient={self.decay_for_weighted_average_gradient}\ndecay_for_weighted_average_squared_gradient={self.decay_for_weighted_average_squared_gradient}"
        )

    def updateHyperParam(
        self,
        learning_rate: float,
        dimension: int,
        decay_for_weighted_moving_average_gradient: float = 0.9,
        decay_for_weighted_moving_average_squared_gradient: float = 0.99,
    ):
        if Adam.isValidAdamParam(
            decay_for_weighted_moving_average_gradient,
            decay_for_weighted_moving_average_squared_gradient,
        ):
            super().updateHyperParam(learning_rate, dimension)
            self.decay_for_weighted_average_gradient = (
                decay_for_weighted_moving_average_gradient
            )
            self.decay_for_weighted_average_squared_gradient = (
                decay_for_weighted_moving_average_squared_gradient
            )

    @staticmethod
    def isValidAdamParam(
        decay_for_weighted_moving_average_gradient,
        decay_for_weighted_moving_average_squared_gradient,
    ):
        if (
            decay_for_weighted_moving_average_gradient < 0
            or decay_for_weighted_moving_average_gradient >= 1
            or decay_for_weighted_moving_average_squared_gradient < 0
            or decay_for_weighted_moving_average_squared_gradient >= 1
        ):
            raise ValueError("Invalid decay value")
        return True

    @staticmethod
    def correction(x, decay, exponent):
        return x / (1 - decay**exponent)

    def step(self, param: dict, gradients: dict):
        self.checkShape(param, gradients)
        self._step_count += 1
        param["weight"] -= self._compute_update("weight", gradients["weight"])
        param["bias"] -= self._compute_update("bias", gradients["bias"])

    def _compute_update(self, param_name: str, gradients):
        self._weighted_average_gradient[param_name] = (
            self.decay_for_weighted_average_gradient
            * self._weighted_average_gradient[param_name]
            + (1 - self.decay_for_weighted_average_gradient) * gradients
        )
        self._weighted_average_squared_gradient[param_name] = (
            self.decay_for_weighted_average_squared_gradient
            * self._weighted_average_squared_gradient[param_name]
            + (1 - self.decay_for_weighted_average_squared_gradient) * (gradients**2)
        )

        steps = self._step_count
        corrected_weighted_average_gradient = Adam.correction(
            self._weighted_average_gradient[param_name],
            self.decay_for_weighted_average_gradient,
            steps,
        )
        corrected_weighted_average_squared_gradient = Adam.correction(
            self._weighted_average_squared_gradient[param_name],
            self.decay_for_weighted_average_squared_gradient,
            steps,
        )

        return (
            self._learning_rate
            * corrected_weighted_average_gradient
            / (np.sqrt(corrected_weighted_average_squared_gradient) + self.padding)
        )


class Momentum(Optimizer):
    def __init__(
        self, learning_rate: float, dimension: int, decay_for_velocity: float = 0.9
    ):
        super().__init__(learning_rate, dimension)
        if Momentum.isValidMomentumParam(decay_for_velocity):
            self._decay_for_velocity = decay_for_velocity
        self._velocity = {"weight": np.zeros(dimension), "bias": 0.0}

    @staticmethod
    def isValidMomentumParam(decay_for_velocity):
        if decay_for_velocity > 1 or decay_for_velocity <= 0:
            raise ValueError("Invalid velocity decay factor")
        return True

    def __str__(self):
        return super().__str__() + f"\ndecay_for_velocity={self._decay_for_velocity}"

    def updateHyperParam(
        self, learning_rate: float, dimension: int, decay_for_velocity: float = 0.9
    ):
        if Momentum.isValidMomentumParam(decay_for_velocity):
            super().updateHyperParam(learning_rate, dimension)
            self._decay_for_velocity = decay_for_velocity

    def step(self, param: dict, gradients: dict):
        self.checkShape(param, gradients)
        self._step_count += 1
        self._velocity["weight"] = (
            self._velocity["weight"] * self._decay_for_velocity + gradients["weight"]
        )

        self._velocity["bias"] = (
            self._velocity["bias"] * self._decay_for_velocity + gradients["bias"]
        )
        param["weight"] -= self._velocity["weight"] * self._learning_rate
        param["bias"] -= self._learning_rate * self._velocity["bias"]
