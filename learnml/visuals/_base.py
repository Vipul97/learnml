import matplotlib.pyplot as plt

def plot_linear_regression(X_train, y_train, X_test=None, y_pred=None, xlabel="x_1", ylabel="y"):
    plt.xlabel("${}$".format(xlabel), fontsize=18)
    plt.ylabel("${}$".format(ylabel), fontsize=18)
    plt.plot(X_train, y_train, "b.")

    if X_test is not None and y_pred is not None:
        plt.plot(X_test, y_pred, "r-", linewidth=2, label="Predictions")
        plt.legend(loc="upper left", fontsize=14)

    plt.show()
