from Optimizer import Optimizer
import numpy as np


class LinearRegression:
    total_model_count = 0

    def __init__(
        self,
        features: list[list[float]],
        y: list[float],
        optimizer: Optimizer,
    ):
        if self._isValidShape(features, y):
            self.features = np.array(features, dtype=float)
            self.y = np.array(y, dtype=float)
        else:
            raise ValueError("Incorrect data shape")

        # weight is a numpy array.
        self._param = {"w": [], "b": 0}
        self._optimizer = optimizer
        self._initializeParam()
        LinearRegression.total_model_count += 1

    def _initializeParam(self):
        self._param["w"] = np.zeros(len(self.features[0]))
        self._param["b"] = 0

    def fit(self, epochs: int):
        for epoch in range(epochs):
            self._optimizer.step(self._param, self._gradients())
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {self._loss():.4f}")

    def predict(self, features):
        result = np.asarray(features) @ self._param["w"] + self._param["b"]
        return result

    def _loss(self):
        error = self.predict(self.features) - self.y
        return np.mean(error**2)

    @staticmethod
    def dot_product(a, b):
        result = 0
        for i in range(len(a)):
            result += a[i] * b[i]
        return result

    def _gradients(self):
        n = len(self.features)
        error = self.predict(self.features) - self.y
        # times the transpose of features (error length == features length)
        dw = 2 * (self.features.T @ error) / n
        db = 2 * error.sum() / n

        return {"dw": dw, "db": db}

    def _isValidShape(self, features, y):
        if len(features) != len(y) or len(features) == 0:
            return False
        for i in range(len(features)):
            if len(features[i]) != len(features[0]):
                return False
        return True


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

    """
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

    """

    def _gradients(self):
        n = len(self.features)
        error = self.predict(self.features) - self.y
        # times the transpose of features (error length == features length)
        dw = 2 * (self.features.T @ error) / n + self._a * np.sign(self._param["w"])
        db = 2 * error.sum() / n

        return {"dw": dw, "db": db}


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
        n = len(self.features)
        error = self.predict(self.features) - self.y
        # times the transpose of features (error length == features length)
        dw = 2 * (self.features.T @ error) / n + self._a * self._param["w"]
        db = 2 * error.sum() / n

        return {"dw": dw, "db": db}
