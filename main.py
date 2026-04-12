
import numpy as np

def sigmoid(z):
    # prevent overflow
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def compute_loss(y, y_hat):
    # prevent log(0)
    eps = 1e-9
    y_hat = np.clip(y_hat, eps, 1 - eps)
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


def train_logistic_regression(X, y, lr=0.01, epochs=1000):
    n_samples, n_features = X.shape

    # initial weights
    weights = np.zeros(n_features)
    bias = 0

    for epoch in range(epochs):
        # forward
        linear_output = np.dot(X, weights) + bias
        y_hat = sigmoid(linear_output)

        # loss
        loss = compute_loss(y, y_hat)

        # gradients
        dw = (1 / n_samples) * np.dot(X.T, (y_hat - y))
        db = (1 / n_samples) * np.sum(y_hat - y)

        # update
        weights -= lr * dw
        bias -= lr * db

        # print loss
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    return weights, bias


def predict(X, weights, bias):
    linear_output = np.dot(X, weights) + bias
    y_hat = sigmoid(linear_output)
    return (y_hat >= 0.5).astype(int)


# ======================
# test on synthetic data
# ======================
if __name__ == "__main__":
    np.random.seed(42)

    # create data
    X = np.random.randn(100, 2)
    true_weights = np.array([2, -3])
    true_bias = 0.5

    logits = np.dot(X, true_weights) + true_bias
    y = (sigmoid(logits) > 0.5).astype(int)

    # train
    weights, bias = train_logistic_regression(X, y, lr=0.1, epochs=1000)

    # predict
    y_pred = predict(X, weights, bias)

    accuracy = np.mean(y_pred == y)
    print(f"Accuracy: {accuracy:.2f}")


