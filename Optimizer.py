import numpy as np


class Optimizer:
    def __init__(self, learning_rate: float, n: int):
        self._learning_rate = learning_rate
        self._step_count = 0
        self._n = n

    def checkShape(self, param: dict, gradients: dict):
        if len(param["w"]) == self._n == len(gradients["dw"]):
            return
        raise ValueError("Optimizer Error: Shape doesn't match")

    def step(self, param: dict, gradients: dict):
        self.checkShape(param, gradients)
        self._step_count += 1
        param["w"] -= self._learning_rate * gradients["dw"]
        param["b"] -= self._learning_rate * gradients["db"]


class Adam(Optimizer):
    def __init__(
        self, learning_rate: float, n: int, beta1: float = 0.9, beta2: float = 0.99
    ):
        super().__init__(learning_rate, n)
        self.beta1 = beta1
        self.beta2 = beta2

        self._m = {"w": np.zeros(n), "b": 0.0}
        self._v = {"w": np.zeros(n), "b": 0.0}

        # an extremely small number used to prevent division by zero error
        self.eps = 1e-8

    @staticmethod
    def correction(x, beta, exp):
        return x / (1 - beta**exp)

    def step(self, param: dict, gradients: dict):
        self.checkShape(param, gradients)
        self._step_count += 1
        param["w"] -= self._compute_update("w", gradients["dw"])
        param["b"] -= self._compute_update("b", gradients["db"])

    def _compute_update(self, param_name: str, gradients):
        self._m[param_name] = (
            self.beta1 * self._m[param_name] + (1 - self.beta1) * gradients
        )
        self._v[param_name] = self.beta2 * self._v[param_name] + (1 - self.beta2) * (
            gradients**2
        )

        steps = self._step_count
        m_hat = Adam.correction(self._m[param_name], self.beta1, steps)
        v_hat = Adam.correction(self._v[param_name], self.beta2, steps)

        return self._learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)


class Momentum(Optimizer):
    def __init__(self, learning_rate: float, n: int, beta: float = 0.9):
        super().__init__(learning_rate, n)
        self._beta = beta
        self._v = {"vw": np.zeros(n), "vb": 0.0}

    def step(self, param: dict, gradients: dict):
        self.checkShape(param, gradients)
        self._step_count += 1
        self._v["vw"] = self._v["vw"] * self._beta + gradients["dw"]

        self._v["vb"] = self._v["vb"] * self._beta + gradients["db"]
        param["w"] -= self._v["vw"] * self._learning_rate
        param["b"] -= self._learning_rate * self._v["vb"]
