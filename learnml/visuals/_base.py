import matplotlib.pyplot as plt
import numpy as np


def plot_regression(X, y, model, poly_features=None, xlabel="x_1", ylabel="y"):
    plt.xlabel("${}$".format(xlabel), fontsize=18)
    plt.ylabel("${}$".format(ylabel), fontsize=18)

    X_new = np.linspace(np.min(X), np.max(X), 1000).reshape(1000, 1)

    if poly_features:
        X_new = poly_features.transform(X_new)

    y_new = model.predict(X_new)

    plt.plot(X, y, "b.")
    plt.plot(X_new[:, 0], y_new, "r-", linewidth=2, label="Predictions")
    plt.show()


def plot_w_path(w_path):
    plt.xlabel(r"$\theta_0$", fontsize=18)
    plt.ylabel(r"$\theta_1$", fontsize=18)
    plt.plot(w_path[:, 0], w_path[:, 1], "r-s")
    plt.show()
