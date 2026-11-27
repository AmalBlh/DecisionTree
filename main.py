from decisionTree import DecisionTree
import numpy as np
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler

def main():

    data = {
        "Edad": [
            22,
            25,
            47,
            52,
            46,
            56,
            23,
            35,
            40,
            60,
            48,
            29,
            31,
            45,
            50,
            27,
            33,
            55,
            38,
            42,
        ],
        "Salario": [
            18000,
            22000,
            50000,
            60000,
            45000,
            70000,
            20000,
            35000,
            40000,
            80000,
            55000,
            25000,
            30000,
            48000,
            65000,
            23000,
            32000,
            75000,
            42000,
            46000,
        ],
        "Experiencia": [
            0,
            1,
            15,
            20,
            12,
            25,
            1,
            8,
            10,
            30,
            18,
            3,
            5,
            14,
            22,
            2,
            6,
            27,
            11,
            13,
        ],
        "Comprara": [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1],
    }

    X = np.array(
        [
            [
                22,
                25,
                47,
                52,
                46,
                56,
                27,
                35,
                40,
                60,
                48,
                29,
                23,
                45,
                50,
                27,
                33,
                55,
                38,
                42,
            ],
            [
                18000,
                50000,
                20000,
                60000,
                45000,
                70000,
                20000,
                35000,
                40000,
                80000,
                35000,
                25000,
                30000,
                48000,
                60000,
                23000,
                32000,
                75000,
                42000,
                46000,
            ],
            [
                0,
                1,
                15,
                20,
                12,
                25,
                1,
                8,
                10,
                30,
                18,
                3,
                5,
                14,
                22,
                2,
                6,
                27,
                11,
                13,
            ],
        ]
    )

    y = np.array([[0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1]])
    # X = df[["Edad", "Salario", "Experiencia"]]
    # y = df["Comprara"]



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
    

    max_depth = 10
    min_sample_leaf = 2
    tree = DecisionTree(max_depth, min_sample_leaf)

    tree.fit(X, y)
    tree.display_tree()
    y = tree.predict(X)
    print(y)
    #tree.score()


if __name__ == "__main__":
    main()
