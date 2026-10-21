from dataclasses import dataclass


@dataclass
class Point:
    x: float
    y: float


class Optimizer:
    def __init__(self, learning_rate: float):

        self._learning_rate = learning_rate
        self._step_count = 0

    def step(self, param: dict, gradients: dict):
        self._step_count += 1


class MultivariableLinearRegression:
    def __init__(
        self, features: list[list[float]], y: list[float], optimizer: Optimizer
    ):
        if self._isValidShape(features, y):
            self.features = features
            self.y = y
        else:
            raise ValueError("Incorrect data shape")
        self._param = {"w": [], "b": 0}
        self._initializeParam()

    def _initializeParam(self):
        self._param["w"] = [0.0] * len(self.features[0])
        self._param["b"] = 0

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
                        MultivariableLinearRegression.dot_product(
                            self._param["w"], self.features[i]
                        )
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
                    MultivariableLinearRegression.dot_product(
                        self._param["w"], self.features[i]
                    )
                    + self._param["b"]
                    - self.y[i]
                )
            ) / batch_size
        return {"dw": grad, "db": grad_b}

    def _isValidShape(self, features, y):
        if len(features) != 2 or len(features) != len(y):
            return False
        for i in range(len(features)):
            if len(features[i]) != len(features[0]):
                return False
        return True


class MultiVariableOptimizer:
    def __init__(self, learning_rate: float):
        self._learning_rate = learning_rate
        self._step_count = 0

    def step(self, param: dict, gradients: dict):
        self._step_count += 1
        for i in range(len(param["w"])):
            param["w"][i] -= self._learning_rate * gradients["dw"][i]
        param["b"] -= self._learning_rate * gradients["db"]


class LinearRegression:
    def __init__(self, data: list[Point], optimizer: Optimizer):

        self.data = data
        self._param = {"w": 0.0, "b": 0.0}
        self._optimizer = optimizer

    def fit(self, epochs: int):
        for epoch in range(epochs):
            gradients = self._gradients()
            self._optimizer.step(self._param, gradients)
            loss = self._loss()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

    def predict(self, x):
        return self._param["w"] * x + self._param["b"]

    def _gradients(self):
        dw = 0
        db = 0
        n = len(self.data)
        for point in self.data:
            error = self.predict(point.x) - point.y
            # Derivative of loss with respect to w
            dw += 2 * error * point.x
            # Derivative of loss with respect to b
            db += 2 * error
        return {"dw": dw / n, "db": db / n}

    def _loss(self):
        loss_sum = 0
        for point in self.data:
            prediction = self.predict(point.x)
            loss_sum += (prediction - point.y) ** 2
        return loss_sum / len(self.data)


class NormalEquation(LinearRegression):
    def fit(self, epochs: int = 1):
        # dw: 2(wx + b - y)x = 0 or 2wx^2 + 2bx - 2xy = 0
        # db: 2(wx + b - y) = 0 or 2wx + 2b - 2y = 0
        eli = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        for point in self.data:
            eli[0][0] += 2 * point.x**2
            eli[1][0] += 2 * point.x
            eli[0][1] += 2 * point.x
            eli[1][1] += 2
            eli[0][2] += -2 * point.x * point.y
            eli[1][2] += -2 * point.y

        # Elimination for w
        quo = eli[0][1] / eli[1][1]
        for i in range(len(eli[0])):
            eli[0][i] -= quo * eli[1][i]

        w = -1 * eli[0][2] / eli[0][0]

        # Elimination for b
        quo = eli[1][0] / eli[0][0]

        for i in range(len(eli[1])):
            eli[1][i] -= quo * eli[0][i]

        b = -1 * eli[1][2] / eli[1][1]

        self._param["w"] = w
        self._param["b"] = b


# L2 regularization
class RidgeRegression(LinearRegression):
    def __init__(self, data: list[Point], optimizer: Optimizer, alpha=0):
        super().__init__(data, optimizer)
        self._a = alpha

    def _gradients(self):
        dw = 0
        db = 0
        n = len(self.data)
        for point in self.data:
            error = self.predict(point.x) - point.y
            # penalize large weight
            dw += 2 * error * point.x + 2 * self._a * self._param["w"]
            db += 2 * error
        return {"dw": dw / n, "db": db / n}


class Momentum(Optimizer):
    def __init__(self, learning_rate: float, beta: float = 0.9):
        super().__init__(learning_rate)
        self._beta = beta
        self._v = [0.0, 0.0]

    def step(self, param: dict, gradients: dict):
        self._v[0] = self._v[0] * self._beta + gradients["dw"]
        self._v[1] = self._v[1] * self._beta + gradients["db"]


class GradientDescent(Optimizer):
    def step(self, param: dict, gradients: dict):
        param["w"] -= self._learning_rate * gradients["dw"]
        param["b"] -= self._learning_rate * gradients["db"]


if __name__ == "__main__":
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [3.4, 4.7, 7.6, 8.5, 11.2, 13.7, 14.6, 16.4, 19.3, 20.8]
    data = []
    for i in range(len(x)):
        data.append(Point(x=x[i], y=y[i]))

    model = LinearRegression(data=data, optimizer=GradientDescent(learning_rate=0.01))
    model.fit(epochs=1000)
    print(model._param)
