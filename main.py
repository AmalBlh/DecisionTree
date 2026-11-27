from decisionTree import DecisionTree
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
import time


def main():

    # Génération de 1000 observations et 3 variables
    X, y = make_regression(
        n_samples=1000,
        n_features=3,
        n_informative=3,
        noise=15,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Ajuster les normaliseurs uniquement sur les données d'entraînement.
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_scaler = MinMaxScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    max_depth = 4
    min_sample_leaf = 5
    tree = DecisionTree(max_depth, min_sample_leaf)

    tree_verif = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_sample_leaf)

    start = time.time()
    tree.fit(X_train, y_train)
    end = time.time()
    length = end - start

    start = time.time()
    tree_verif.fit(X_train, y_train)
    end = time.time()
    length2 = end - start

    y_pred = tree.predict(X_test)
    y_pred_verif = tree_verif.predict(X_test)

    score = tree.score(y_test, y_pred)
    score_verif = mean_squared_error(y_test, y_pred_verif)
    tree.display_tree()
    print("DecisionTree MSE:", score)
    print("DecisionTreeRegressor MSE:", score_verif)

    print("DecisionTree training time:", length, "seconds")
    print("DecisionTreeRegressor training time:", length2, "seconds")


if __name__ == "__main__":
    main()
