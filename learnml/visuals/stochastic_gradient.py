from learnml.metrics import mean_squared_error
from learnml.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


def plot_regression(model, X, y, poly_features=None, xlabel="x_1", ylabel="y"):
    plt.xlabel("${}$".format(xlabel), fontsize=18)
    plt.ylabel("${}$".format(ylabel), fontsize=18)

    X_new = np.linspace(np.min(X), np.max(X), 1000).reshape(1000, 1)

    if poly_features:
        X_new = poly_features.transform(X_new)

    y_new = model.predict(X_new)

    plt.plot(X, y, "b.")
    plt.plot(X_new[:, 0], y_new, "r-", linewidth=2, label="Predictions")
    plt.legend(loc="upper left", fontsize=14)
    plt.show()


def plot_w_path(w_path):
    plt.xlabel(r"$\theta_0$", fontsize=18)
    plt.ylabel(r"$\theta_1$", fontsize=18)
    plt.plot(w_path[:, 0], w_path[:, 1], "r-s")
    plt.show()


def plot_learning_curves(model, X, y):
    X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
    y_train, y_val = train_test_split(y, test_size=0.2, random_state=42)
    train_errors, val_errors = [], []

    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))

    plt.xlabel("Training set size", fontsize=14)
    plt.ylabel("RMSE", fontsize=14)
    plt.plot(np.sqrt(train_errors), "r-", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=2, label="val")
    plt.legend(loc="upper right", fontsize=14)
    plt.show()
