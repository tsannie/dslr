import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score


class LogisticRegression:
    def __init__(self, learning_rate=0.01, batch_size=10, epochs=30):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.w = None
        self.b = None
        self.history_accuracy = []
        self.history_loss = []

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
            print("\nEpoch {}/{}".format(i + 1, self.epochs))
            for idx in tqdm(range(0, n_samples, self.batch_size)):
                X_batch = X[idx : idx + self.batch_size]
                y_batch = y[idx : idx + self.batch_size]

                dw, db = self.grandient_descent(X_batch, y_batch)

                self.w -= self.learning_rate * dw
                self.b -= self.learning_rate * db

                y_predicted = self.sigmoid(np.dot(X, self.w) + self.b)
                y_predicted_class = [1 if i > 0.5 else 0 for i in y_predicted]
                accuracy = accuracy_score(y, y_predicted_class)
                loss = self.log_loss(y, y_predicted)
                self.history_accuracy.append(accuracy)
                self.history_loss.append(loss)
            print("Loss: {:.4f}, Accuracy: {:.4f}".format(loss, accuracy))

        """

        plt.plot(self.history_accuracy)
        plt.title("Accuracy")

        plt.figure()
        plt.plot(self.history_loss)
        plt.title("Loss")

        plt.show() """

        y_predicted = self.sigmoid(np.dot(X, self.w) + self.b)
        y_predicted_class = [1 if i > 0.5 else 0 for i in y_predicted]
        print("Loss: ", self.log_loss(y, y_predicted))
        print("Accuracy: ", accuracy_score(y, y_predicted_class))

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
