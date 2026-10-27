from LinearRegression import LassoRegression, LinearRegression, RidgeRegression
from Optimizer import Adam, Optimizer, Momentum
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

if __name__ == "__main__":
    features, expectedOutput = load_diabetes(return_X_y=True, scaled=False)
    features_train, features_test, outputs_train, outputs_test = train_test_split(
        features, expectedOutput, test_size=0.2, random_state=0
    )
    mean, std = features_train.mean(axis=0), features_train.std(axis=0)
    features_train = (features_train - mean) / std
    features_test = (features_test - mean) / std

    model = LinearRegression(
        features_train,
        outputs_train,
        Momentum(
            learning_rate=0.01, dimension=len(features_train[0]), decay_for_velocity=0.9
        ),
    )
    model.fit(epochs=1000)
    plt.plot(model.loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()

    model.updateOptimizer(Adam(learning_rate=0.01, dimension=len(features_train[0])))
    model.fit(epochs=1000)

    ridge_model = RidgeRegression(
        features_train,
        outputs_train,
        Optimizer(learning_rate=0.01, dimension=len(features_train[0])),
    )
    ridge_model.fit(epochs=1000)

    lasso_model = LassoRegression(
        features_train,
        outputs_train,
        Optimizer(learning_rate=0.01, dimension=len(features_train[0])),
    )
    lasso_model.fit(epochs=1000)

    result = model.predict(features_test)
    print("Predicted result:" + result)
