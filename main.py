from decisionTree import DecisionTree
import numpy as np
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
import time


def main():

    # Génération de 100 observations et 3 variables
    X, y = make_regression(
        n_samples=50000, n_features=3, n_informative=3, noise=15, random_state=42
    )

    # Normalisation de X entre 0 et 1
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Normalisation de y entre 0 et 1
    y_scaler = MinMaxScaler()
    y = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

    # Vérification
    print("X shape :", X.shape)
    print("y shape :", y.shape)
    print("X min/max :", X.min(), X.max())
    print("y min/max :", y.min(), y.max())

    print("\nX :\n", X[:5])
    print("\ny :\n", y[:5])

    max_depth = 4
    min_sample_leaf = 5
    tree = DecisionTree(max_depth, min_sample_leaf)

    tree_verif = DecisionTreeRegressor(max_depth=3, min_samples_leaf=2)

    start = time.time()
    tree.fit(X, y)

    tree_verif.fit(X, y)

    end = time.time()
    length = end - start

    start = time.time()

    tree_verif.fit(X, y)

    end = time.time()
    length2 = end - start

    y_pred = tree.predict(X)

    y_pred_verif = tree_verif.predict(X)

    print(y_pred)
    score = tree.score(y, y_pred)
    score_verif = mean_squared_error(y, y_pred_verif)
    tree.display_tree()
    print(y_pred)
    print("DecisionTree MSE:", score)
    print("DecisionTreeRegressor MSE:", score_verif)

    print("Length 1", length, "seconds!")

    print("Length 2 ", length2, "seconds!")


if __name__ == "__main__":
    main()
