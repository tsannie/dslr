import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from data_analysis.utils import colorize_house, colorize_plot


class LogisticRegression:
    def __init__(self, house_name, learning_rate=0.01, batch_size=10, epochs=5):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.w = None
        self.b = None
        self.history_accuracy = []
        self.history_loss = []
        self.house_name = house_name

    def grandient_descent(self, X, y):
        m = X.shape[0]
        y_predicted = self.sigmoid(np.dot(X, self.w) + self.b)
        dz = y_predicted - y
        dw = 1 / m * np.dot(X.T, dz)
        db = 1 / m * np.sum(dz, axis=0).reshape(-1, 1)
        return dw, db

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.w = np.random.randn(n_features).reshape(-1, 1)
        self.b = np.random.randn(1).reshape(-1, 1)

        for i in range(self.epochs):
            for e in tqdm(
                range(0, n_samples, self.batch_size),
                leave=False,
                desc="{} epoch {:0{}}/{:0{}}, loss {:.4f}, accuracy {:.4f}".format(
                    colorize_house(self.house_name),
                    i + 1,
                    len(str(self.epochs)),
                    self.epochs,
                    len(str(self.epochs)),
                    self.history_loss[-1] if self.history_loss else 0,
                    self.history_accuracy[-1] if self.history_accuracy else 0,
                ),
            ):
                X_batch = X[e : e + self.batch_size]
                y_batch = y[e : e + self.batch_size]

                dw, db = self.grandient_descent(X_batch, y_batch)

                self.w -= self.learning_rate * dw
                self.b -= self.learning_rate * db

                y_predicted = self.sigmoid(np.dot(X, self.w) + self.b)
                y_predicted_class = [1 if i > 0.5 else 0 for i in y_predicted]
                self.history_accuracy.append(accuracy_score(y, y_predicted_class))
                self.history_loss.append(self.log_loss(y, y_predicted))

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def predict(self, X):
        y_predicted = self.sigmoid(np.dot(X, self.w) + self.b)
        y_predicted_class = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_class

    def log_loss(self, y, y_predicted):
        m = y.shape[0]
        epsilon = 1e-15
        y_predicted = np.clip(y_predicted, epsilon, 1 - epsilon)
        return (
            -1 / m * np.sum(y * np.log(y_predicted) + (1 - y) * np.log(1 - y_predicted))
        )

    def save(self, path, house):
        with open(path, mode="a") as f:
            f.write(house + ",")
            np.savetxt(f, self.w.T, delimiter=" ")
            f.close()

    def stats(self):
        _, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax[0].plot(self.history_loss)
        ax[0].set_title("Loss")
        ax[1].plot(self.history_accuracy)
        ax[1].set_title("Accuracy")

        plt.suptitle(
            "House {}".format(self.house_name),
            fontsize=20,
            fontweight="bold",
            color=colorize_plot(self.house_name),
        )
        plt.tight_layout()
        plt.show()
