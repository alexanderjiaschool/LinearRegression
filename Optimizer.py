class Optimizer:
    def __init__(self, learning_rate: float, n: int):
        self._learning_rate = learning_rate
        self._step_count = 0
        self._n = n

    def step(self, param: dict, gradients: dict):
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
        self._m_w = [0.0 for _ in range(n)]
        self._v_w = [0.0 for _ in range(n)]

        self._m_b = 0
        self._v_b = 0

    @property
    def m_w_corrected(self):
        return [
            self._m_w[i] / (1 - self.beta1**self._step_count) for i in range(self._n)
        ]

    @property
    def v_w_corrected(self):
        return [
            self._v_w[i] / (1 - self.beta2**self._step_count) for i in range(self._n)
        ]

    @property
    def m_b_corrected(self):
        return self._m_b / (1 - self.beta1**self._step_count)

    @property
    def v_b_corrected(self):
        return self._v_b / (1 - self.beta2**self._step_count)

    def step(self, param: dict, gradients: dict):
        self._step_count += 1
        for i in range(self._n):
            self._m_w[i] = (
                self.beta1 * self._m_w[i] + (1 - self.beta1) * gradients["dw"][i]
            )
            self._v_w[i] = self.beta2 * self._v_w[i] + (1 - self.beta2) * (
                gradients["dw"][i] ** 2
            )
            param["w"][i] -= (
                self._learning_rate
                * self.m_w_corrected[i]
                / (self.v_w_corrected[i] ** 0.5 + 1e-8)
            )
        self._m_b = self.beta1 * self._m_b + (1 - self.beta1) * gradients["db"]
        self._v_b = self.beta2 * self._v_b + (1 - self.beta2) * (gradients["db"] ** 2)
        param["b"] -= (
            self._learning_rate * self.m_b_corrected / (self.v_b_corrected**0.5 + 1e-8)
        )


class Momentum(Optimizer):
    def __init__(self, learning_rate: float, n: int, beta: float = 0.9):
        super().__init__(learning_rate, n)
        self._beta = beta
        self._v = {"vw": [0.0 for _ in range(n)], "vb": 0.0}

    def step(self, param: dict, gradients: dict):
        self._step_count += 1
        self._v["vw"] = (
            Momentum.array_num_multiply(self._v["vw"], self._beta) + gradients["dw"]
        )
        self._v["vb"] = self._v["vb"] * self._beta + gradients["db"]
        param["w"] -= Momentum.array_num_multiply(self._v["vw"], self._learning_rate)
        param["b"] -= self._learning_rate * self._v["vb"]

    @staticmethod
    def array_num_multiply(arr: list[float], num: float) -> list[float]:
        return [x * num for x in arr]
