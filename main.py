from decisionTree import DecisionTree
import numpy as np
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler

def main():

    #Génération de 100 observations et 3 variables
    X, y = make_regression(
        n_samples=100,
        n_features=3,
        n_informative=3,
        noise=15,
        random_state=42
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
    min_sample_leaf = 2
    tree = DecisionTree(max_depth, min_sample_leaf)

    tree.fit(X, y)

   
    y_pred = tree.predict(X)
    print(y_pred)
    score = tree.score(y ,y_pred)
    tree.display_tree()
    print(y_pred)
    print(score)


if __name__ == "__main__":
    main()
