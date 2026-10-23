from dataclasses import dataclass


@dataclass
class Point:
    x: float
    y: float


class Optimizer:
    def __init__(self, learning_rate: float, n: int):
        self._learning_rate = learning_rate
        self._step_count = 0
        self._n = n

    def step(self, param: dict, gradients: dict):
        self._step_count += 1
        param["w"] -= self._learning_rate * gradients["dw"]
        param["b"] -= self._learning_rate * gradients["db"]


class LinearRegression:
    total_model_count = 0

    def __init__(
        self,
        features: list[list[float]],
        y: list[float],
        optimizer: Optimizer,
    ):
        if self._isValidShape(features, y):
            self.features = features
            self.y = y
        else:
            raise ValueError("Incorrect data shape")
        self._param = {"w": [], "b": 0}
        self._optimizer = optimizer
        self._initializeParam()
        LinearRegression.total_model_count += 1

    def _initializeParam(self):
        self._param["w"] = [0.0] * len(self.features[0])
        self._param["b"] = 0

    def fit(self, epochs: int):
        for epoch in range(epochs):
            self._optimizer.step(self._param, self._gradients())
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {self._loss():.4f}")

    def predict(self, feature):
        result = (
            LinearRegression.dot_product(feature, self._param["w"]) + self._param["b"]
        )
        return result

    def _loss(self):
        n = len(self.features)
        loss = 0
        for i in range(n):
            loss += (self.predict(self.features[i]) - self.y[i]) ** 2
        return loss / n

    @staticmethod
    def dot_product(a, b):
        result = 0
        for i in range(len(a)):
            result += a[i] * b[i]
        return result

    def _gradients(self):
        weightLen = len(self._param["w"])
        grad = [0.0] * weightLen
        grad_b = 0
        # Gradients of w
        batch_size = len(self.features)
        for i in range(batch_size):
            for j in range(len(self._param["w"])):  # for each weight
                # partial derivative
                grad[j] += (
                    2
                    * (
                        LinearRegression.dot_product(self._param["w"], self.features[i])
                        + self._param["b"]
                        - self.y[i]
                    )
                    * self.features[i][j]
                ) / batch_size
        for i in range(batch_size):
            # partial derivative
            grad_b += (
                2
                * (
                    LinearRegression.dot_product(self._param["w"], self.features[i])
                    + self._param["b"]
                    - self.y[i]
                )
            ) / batch_size
        return {"dw": grad, "db": grad_b}

    def _isValidShape(self, features, y):
        if len(features) != len(y) or len(features) == 0:
            return False
        for i in range(len(features)):
            if len(features[i]) != len(features[0]):
                return False
        return True


# L2 regularization
class RidgeRegression(LinearRegression):
    def __init__(
        self,
        features: list[list[float]],
        y: list[float],
        optimizer: Optimizer,
        alpha=0,
    ):
        super().__init__(features, y, optimizer)
        self._a = alpha

    def _gradients(self):
        weightLen = len(self._param["w"])
        grad = [0.0] * weightLen
        grad_b = 0
        # Gradients of w
        batch_size = len(self.features)
        for i in range(batch_size):
            for j in range(len(self._param["w"])):  # for each weight
                # partial derivative
                grad[j] += (
                    2
                    * (
                        LinearRegression.dot_product(self._param["w"], self.features[i])
                        + self._param["b"]
                        - self.y[i]
                    )
                    * self.features[i][j]
                    + self._a * self._param["w"][j]
                ) / batch_size
        for i in range(batch_size):
            # partial derivative
            grad_b += (
                2
                * (
                    LinearRegression.dot_product(self._param["w"], self.features[i])
                    + self._param["b"]
                    - self.y[i]
                )
            ) / batch_size

        return {"dw": grad, "db": grad_b}


# L1
class LassoRegression(LinearRegression):
    def __init__(
        self,
        features: list[list[float]],
        y: list[float],
        optimizer: Optimizer,
        alpha=0,
    ):
        super().__init__(features, y, optimizer)
        self._a = alpha

    @staticmethod
    def sign(x):
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0

    def _gradients(self):
        weightLen = len(self._param["w"])
        grad = [0.0] * weightLen
        grad_b = 0
        # Gradients of w
        batch_size = len(self.features)
        for i in range(batch_size):
            for j in range(len(self._param["w"])):  # for each weight
                # partial derivative
                grad[j] += (
                    2
                    * (
                        LinearRegression.dot_product(self._param["w"], self.features[i])
                        + self._param["b"]
                        - self.y[i]
                    )
                    * self.features[i][j]
                    + self._a * LassoRegression.sign(self._param["w"][j])
                ) / batch_size
        for i in range(batch_size):
            # partial derivative
            grad_b += (
                2
                * (
                    LinearRegression.dot_product(self._param["w"], self.features[i])
                    + self._param["b"]
                    - self.y[i]
                )
            ) / batch_size

        return {"dw": grad, "db": grad_b}


class Momentum(Optimizer):
    def __init__(self, learning_rate: float, n: int, beta: float = 0.9):
        super().__init__(learning_rate, n)
        self._beta = beta
        self._v = {"vw": [0.0 for _ in range(n)], "vb": 0.0}

    def step(self, param: dict, gradients: dict):
        self._step_count += 1
        self._v["w"] = self._v["w"] * self._beta + gradients["dw"]
        self._v["b"] = self._v["b"] * self._beta + gradients["db"]
        param["w"] -= self._learning_rate * self._v["w"]
        param["b"] -= self._learning_rate * self._v["b"]


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


if __name__ == "__main__":
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [3.4, 4.7, 7.6, 8.5, 11.2, 13.7, 14.6, 16.4, 19.3, 20.8]
    data = []
    for i in range(len(x)):
        data.append(Point(x=x[i], y=y[i]))
