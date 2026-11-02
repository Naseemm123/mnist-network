import numpy as np

INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM = 784, 128, 10


def init_params(rng=None):
    rng = rng or np.random.default_rng()
    w1 = rng.standard_normal((HIDDEN_DIM, INPUT_DIM), dtype=np.float32) * np.sqrt(2 / INPUT_DIM)
    b1 = np.zeros((HIDDEN_DIM, 1), dtype=np.float32)
    w2 = rng.standard_normal((OUTPUT_DIM, HIDDEN_DIM), dtype=np.float32) * np.sqrt(2 / HIDDEN_DIM)
    b2 = np.zeros((OUTPUT_DIM, 1), dtype=np.float32)
    return w1, b1, w2, b2


def relu(z):
    return np.maximum(z, 0.0)


def relu_deriv(z):
    return z > 0


def softmax(z):
    shifted = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)


def one_hot(y):
    return np.eye(OUTPUT_DIM, dtype=np.float32)[y].T


def forward_prop(w1, b1, w2, b2, x):
    z1 = w1 @ x + b1
    a1 = relu(z1)
    z2 = w2 @ a1 + b2
    a2 = softmax(z2)
    return z1, z2, a1, a2


def backward_prop(z1, a1, z2, a2, w1, w2, x, y):
    del z2, w1
    batch_size = y.size
    dz2 = a2 - one_hot(y)
    dw2 = (dz2 @ a1.T) / batch_size
    db2 = np.sum(dz2, axis=1, keepdims=True) / batch_size
    dz1 = (w2.T @ dz2) * relu_deriv(z1)
    dw1 = (dz1 @ x.T) / batch_size
    db1 = np.sum(dz1, axis=1, keepdims=True) / batch_size
    return dw1, db1, dw2, db2


def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 -= alpha * dw1
    b1 -= alpha * db1
    w2 -= alpha * dw2
    b2 -= alpha * db2
    return w1, b1, w2, b2


def get_predictions(a2):
    return np.argmax(a2, axis=0)


def get_accuracy(predictions, y):
    return float(np.mean(predictions == y))


def compute_loss(a2, y):
    return float(-np.mean(np.log(np.clip(a2[y, np.arange(y.size)], 1e-7, 1.0))))


def evaluate(w1, b1, w2, b2, x, y, batch_size=4096):
    total_loss = total_correct = 0.0
    for start in range(0, y.size, batch_size):
        end = min(start + batch_size, y.size)
        _, _, _, a2 = forward_prop(w1, b1, w2, b2, x[:, start:end])
        current_y = y[start:end]
        total_loss += compute_loss(a2, current_y) * current_y.size
        total_correct += np.sum(get_predictions(a2) == current_y)
    return total_loss / y.size, float(total_correct / y.size)


def gradient_descent(x_train, y_train, x_test=None, y_test=None, alpha=0.1, epochs=20,
                     batch_size=128, decay_rate=0.5, decay_every=5, seed=42):
    rng = np.random.default_rng(seed)
    w1, b1, w2, b2 = init_params(rng)
    history = []
    for epoch in range(epochs):
        learning_rate = alpha * decay_rate ** (epoch // decay_every)
        permutation = rng.permutation(y_train.size)
        shuffled_x, shuffled_y = x_train[:, permutation], y_train[permutation]
        for start in range(0, y_train.size, batch_size):
            end = min(start + batch_size, y_train.size)
            x_batch, y_batch = shuffled_x[:, start:end], shuffled_y[start:end]
            z1, z2, a1, a2 = forward_prop(w1, b1, w2, b2, x_batch)
            gradients = backward_prop(z1, a1, z2, a2, w1, w2, x_batch, y_batch)
            w1, b1, w2, b2 = update_params(w1, b1, w2, b2, *gradients, learning_rate)
        train_loss, train_accuracy = evaluate(w1, b1, w2, b2, x_train, y_train)
        metrics = {"epoch": epoch + 1, "learning_rate": learning_rate,
                   "train_loss": train_loss, "train_accuracy": train_accuracy}
        if x_test is not None and y_test is not None:
            metrics["test_loss"], metrics["test_accuracy"] = evaluate(w1, b1, w2, b2, x_test, y_test)
            print(f"Epoch {epoch + 1:02d}/{epochs} | accuracy: {metrics['test_accuracy']:.4%}")
        history.append(metrics)
    return (w1, b1, w2, b2), history
