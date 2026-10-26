from Optimizer import Optimizer
import numpy as np


# Getter Setter for protected attribite
class LinearRegression:
    total_model_count = 0

    def __init__(
        self,
        features: list[list[float]],
        expectedOutputs: list[float],
        optimizer: Optimizer,
    ):
        if self._isValidShape(features, expectedOutputs):
            self.features = np.array(features, dtype=float)
            self.expectedOutputs = np.array(expectedOutputs, dtype=float)
        else:
            raise ValueError("Incorrect data shape")

        self._param = {"weight": np.zeros(len(self.features[0])), "bias": 0.0}
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
            expectedOutputs=[5.0, 7.0, 9.0],
            optimizer=Optimizer(learning_rate=0.01, dimension=2),
        )

    def fit(self, epochs: int):
        for epoch in range(epochs):
            self._optimizer.step(self._param, self._gradients())
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {self._loss():.4f}")

    def predict(self, features):
        result = np.asarray(features) @ self._param["weight"] + self._param["bias"]
        return result

    def _loss(self):
        return LinearRegression.mean_squared_error(
            self.predict(self.features), self.expectedOutputs
        )

    @staticmethod
    def mean_squared_error(expectedOutput, predictedOutput):
        return np.mean((expectedOutput - predictedOutput) ** 2)

    def _gradients(self):
        batch_size = len(self.features)
        error = self.predict(self.features) - self.expectedOutputs
        # times the transpose of features (error length == features length)
        gradient_weight = 2 * (self.features.T @ error) / batch_size
        gradient_bias = 2 * error.sum() / batch_size

        return {"weight": gradient_weight, "bias": gradient_bias}

    def _isValidShape(self, inputMatrix, outputMatrix):
        if len(inputMatrix) != len(outputMatrix) or len(inputMatrix) == 0:
            return False
        for i in range(len(inputMatrix)):
            if len(inputMatrix[i]) != len(inputMatrix[0]):
                return False
        return True


# L1
class LassoRegression(LinearRegression):
    def __init__(
        self,
        features: list[list[float]],
        expectedOutputs: list[float],
        optimizer: Optimizer,
        penalty_factor=0.0,
    ):
        super().__init__(features, expectedOutputs, optimizer)
        self._penalty_factor = penalty_factor

    def _gradients(self):
        batch_size = len(self.features)
        error = self.predict(self.features) - self.expectedOutputs
        # times the transpose of features (error length == features length)
        gradient_weight = 2 * (
            self.features.T @ error
        ) / batch_size + self._penalty_factor * np.sign(self._param["weight"])
        gradient_bias = 2 * error.sum() / batch_size

        return {"weight": gradient_weight, "bias": gradient_bias}


# L2 regularization
class RidgeRegression(LinearRegression):
    def __init__(
        self,
        features: list[list[float]],
        expectedOutputs: list[float],
        optimizer: Optimizer,
        penalty_factor=0.0,
    ):
        super().__init__(features, expectedOutputs, optimizer)
        self._penalty_factor = penalty_factor

    def _gradients(self):
        batch_size = len(self.features)
        error = self.predict(self.features) - self.expectedOutputs
        # times the transpose of features (error length == features length)
        gradient_weight = (
            2 * (self.features.T @ error) / batch_size
            + self._penalty_factor * self._param["weight"]
        )
        gradient_bias = 2 * error.sum() / batch_size

        return {"weight": gradient_weight, "bias": gradient_bias}
