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
        self._initializeParam(self.dimension)
        self.updateOptimizer(optimizer)
        self.loss_history = []
        LinearRegression._increaseModelCount()

    @property
    def dimension(self):
        return len(self.features[0])

    def _initializeParam(self, dimension):
        self._param = {"weight": np.zeros(dimension), "bias": 0.0}

    def updateOptimizer(self, optimizer: Optimizer):
        if not isinstance(optimizer, Optimizer):
            raise TypeError("Invalid Optimizer")
        if optimizer._dimension != self.dimension:
            raise ValueError("Optimizer dimension doesn't match")
        self._optimizer = optimizer

    def removeOptimizer(self):
        if self._optimizer is None:
            raise Exception("Optimizer is already removed")
        self._optimizer = None

    def getOptimizer(self):
        return self._optimizer

    def getParam(self):
        return self._param

    def setParam(self, param):
        if len(param["weight"]) != self.dimension or (
            not isinstance(param["bias"], float)
        ):
            raise ValueError("Invalid parameters")
        else:
            self._param = param

    def getSamples(self):
        return self.features, self.expectedOutputs

    def setSamples(self, features, expectedOutputs):
        if self._isValidShape(features, expectedOutputs):
            self.features = np.array(features, dtype=float)
            self.expectedOutputs = np.array(expectedOutputs, dtype=float)
            if self._optimizer:
                self._optimizer._dimension = self.dimension
        else:
            raise ValueError("Invalid samples")

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
        if self._optimizer:
            for epoch in range(epochs):
                self._optimizer.step(self._param, self._gradients())
                self.loss_history.append(self._loss())
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {self._loss():.4f}")
        else:
            raise ValueError("Optimizer doesn't exist, cannot fit")

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
        if LassoRegression.isValidPenaltyFactor(penalty_factor):
            self._penalty_factor = penalty_factor

    @staticmethod
    def isValidPenaltyFactor(penalty_factor):
        if penalty_factor < 0.0:
            raise ValueError("Invalid penalty factor")
        return True

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
        if RidgeRegression.isValidPenaltyFactor(penalty_factor):
            self._penalty_factor = penalty_factor

    @staticmethod
    def isValidPenaltyFactor(penalty_factor):
        if penalty_factor < 0.0:
            raise ValueError("Invalid penalty factor")
        return True

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
