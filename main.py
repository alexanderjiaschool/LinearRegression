from LinearRegression import LinearRegression
from Optimizer import Adam

if __name__ == "__main__":
    # Example usage of LinearRegression with Adam optimizer
    features = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
    y = [5.0, 7.0, 9.0]
    optimizer = Adam(learning_rate=0.01, n=2)
    model = LinearRegression(features, y, optimizer)
    model.fit(epochs=1000)
    """
    optimizer_momentum = Momentum(learning_rate=0.01, n=2, beta=0.9)
    model_momentum = LinearRegression(features, y, optimizer_momentum)
    model_momentum.fit(epochs=1000)
     """
    # prediction test
    print(model.predict([4.0, 5.0]))
