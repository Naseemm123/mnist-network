import json
from pathlib import Path

from sklearn.datasets import fetch_openml

from model import evaluate, gradient_descent


def export_weights(w1, b1, w2, b2, destination=None):
    # Serialize NumPy parameters as browser-friendly nested JSON arrays.
    destination = Path(destination or Path(__file__).with_name("weights.json"))
    payload = {
        "version": 1,
        "input_dim": 784,
        "hidden_dim": 128,
        "output_dim": 10,
        "activation": "relu",
        "w1": w1.tolist(),
        "b1": b1.reshape(-1).tolist(),
        "w2": w2.tolist(),
        "b2": b2.reshape(-1).tolist(),
    }
    serialized = json.dumps(payload, separators=(",", ":"))
    destination.write_text(serialized)
    browser_copy = destination.parents[1] / "web" / "public" / "weights.json"
    if browser_copy.parent.exists():
        browser_copy.write_text(serialized)
    print(f"Exported weights to {destination}")


def main():
    print("Downloading MNIST dataset...")
    mnist = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=False,
        parser="liac-arff",
    )
    x = mnist.data.astype("float32").T / 255.0
    y = mnist.target.astype("int64")
    x_train, x_test = x[:, :60000], x[:, 60000:]
    y_train, y_test = y[:60000], y[60000:]
    (w1, b1, w2, b2), _ = gradient_descent(
        x_train, y_train, x_test=x_test, y_test=y_test,
        alpha=0.1, epochs=20, batch_size=128, decay_rate=0.5, decay_every=5, seed=42,
    )
    _, test_accuracy = evaluate(w1, b1, w2, b2, x_test, y_test)
    print(f"Final test accuracy: {test_accuracy:.4%}")
    export_weights(w1, b1, w2, b2)


if __name__ == "__main__":
    main()
