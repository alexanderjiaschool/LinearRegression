from LinearRegression import LassoRegression, LinearRegression, RidgeRegression
from Optimizer import Adam, Optimizer, Momentum

if __name__ == "__main__":
    # Example usage of LinearRegression with Adam optimizer
    features = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
    y = [5.0, 7.0, 9.0]
    adam = Adam(learning_rate=0.01, n=2)
    model = LinearRegression(features, y, adam)
    model.fit(epochs=100)
    # prediction test
    print(model.predict([4.0, 5.0]))

    standard_model = LinearRegression.standardTemplate()

    lasso_model = LassoRegression(features, y, adam, alpha=0.1)
    lasso_model.fit(epochs=100)

    ridge_model = RidgeRegression(features, y, adam, alpha=0.1)
    ridge_model.fit(epochs=100)

    # model with standard optimizer
    optimizer = Optimizer(learning_rate=0.01, n=2)
    standard_optimizer_model = LinearRegression(features, y, optimizer)
    standard_optimizer_model.fit(epochs=100)

    momentum = Momentum(learning_rate=0.01, n=2, beta=0.9)
    optimizer_momentum_model = LinearRegression(features, y, momentum)
    optimizer_momentum_model.fit(epochs=100)

    # prediction test
    print(lasso_model.predict([4.0, 5.0]))
    print(ridge_model.predict([4.0, 5.0]))
    print(standard_optimizer_model.predict([4.0, 5.0]))
    print(optimizer_momentum_model.predict([4.0, 5.0]))
    print(standard_model.predict([4.0, 5.0]))
