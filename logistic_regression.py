import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score

class LogisticRegression:
    def __init__(self, learning_rate=1, batch_size=1, epochs=1):
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

        for _ in tqdm(range(self.epochs)):
            for idx in range(0, n_samples, self.batch_size):
                X_batch = X[idx:idx+self.batch_size]
                y_batch = y[idx:idx+self.batch_size]

                dw, db = self.grandient_descent(X_batch, y_batch)

                self.w -= self.learning_rate * dw
                self.b -= self.learning_rate * db

                """ y_predicted = self.sigmoid(np.dot(X, self.w) + self.b)
                y_predicted_class = [1 if i > 0.5 else 0 for i in y_predicted]
                self.history_accuracy.append(accuracy_score(y, y_predicted_class))
                self.history_loss.append(self.log_loss(y, y_predicted))

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
        return - 1 / m * np.sum(y * np.log(y_predicted)
                                + (1 - y) * np.log(1 - y_predicted))

    def save(self, path, house):
        with open(path, mode='a') as f:
            f.write(house + ",")
            np.savetxt(f, self.w.T, delimiter=" ")
            f.close()
