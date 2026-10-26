from Optimizer import Optimizer
import numpy as np


# Getter Setter for protected attribite
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

        self._param = {"w": np.zeros(len(self.features[0])), "b": 0.0}
        self.updateOptimizer(optimizer)
        LinearRegression._increaseModelCount()

    def updateOptimizer(self, optimizer: Optimizer):
        if not isinstance(optimizer, Optimizer):
            raise TypeError("Invalid Optimizer")
        self._optimizer = optimizer

    @classmethod
    def _increaseModelCount(cls):
        cls.total_model_count += 1

    @classmethod
    def getModelCount(cls):
        return cls.total_model_count

    @classmethod
    def standardTemplate(cls):
        return cls(
            features=[[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
            y=[5.0, 7.0, 9.0],
            optimizer=Optimizer(learning_rate=0.01, n=2),
        )

    def fit(self, epochs: int):
        for epoch in range(epochs):
            self._optimizer.step(self._param, self._gradients())
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {self._loss():.4f}")

    def predict(self, features):
        result = np.asarray(features) @ self._param["w"] + self._param["b"]
        return result

    def _loss(self):
        return LinearRegression.mean_squared_error(self.predict(self.features), self.y)

    @staticmethod
    def mean_squared_error(y, y_hat):
        return np.mean((y_hat - y) ** 2)

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
        alpha=0.0,
    ):
        super().__init__(features, y, optimizer)
        self._a = alpha

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
        alpha=0.0,
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
