import json
import os
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

# Import PyTorch VGG architecture and execution device
from vgg_model import PyTorchVGG, device

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

N_SAMPLES = 4


def test_vgg(n_samples=N_SAMPLES):

    # 1. Load test dataset and class names
    X_test = np.load(f"{OUTPUT_DIR}/X_test.npy")
    y_test = np.load(f"{OUTPUT_DIR}/y_test.npy")
    with open(f"{OUTPUT_DIR}/classes.json") as f:
        classes = json.load(f)

    # 2. Randomly select sample images for inference
    index = np.random.choice(len(X_test), n_samples, replace=False)
    X_sample = X_test[index]
    y_sample = y_test[index]

    # 3. Load trained model weights and move model to GPU
    num_classes = len(classes)
    model = PyTorchVGG(num_classes=num_classes).to(device)
    model_path = os.path.join(OUTPUT_DIR, "vgg_model.pt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at: {model_path}. Please run main.py to train the model first."
        )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 4. Perform model inference on GPU
    inputs = torch.tensor(X_sample).to(device)
    with torch.no_grad():
        outputs = model(inputs).squeeze()

        if num_classes == 2:
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            if probabilities.ndim == 0:
                probabilities = np.array([probabilities.item()])
            predictions = (probabilities > 0.5).astype(int)
            confidence = np.where(predictions == 1, probabilities, 1 - probabilities)
        else:
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            predictions = probs.argmax(axis=1)
            confidence = probs.max(axis=1)

    # 5. Plot and save prediction results
    cols = int(np.ceil(np.sqrt(n_samples)))
    rows = int(np.ceil(n_samples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 4.0 * rows))
    axes = np.atleast_1d(axes).ravel()

    print(f"\n[PyTorch Inference on {device.type.upper()}]")

    for i, ax in enumerate(axes):
        if i >= n_samples:
            ax.axis("off")
            continue

        pred = classes[predictions[i]]
        true = classes[y_sample[i]]
        correct = predictions[i] == y_sample[i]
        color = "green" if correct else "red"

        ax.imshow(X_sample[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"Pred: {pred} ({confidence[i] * 100:.0f}%)\nTrue: {true}", color=color
        )

        print(
            f"[{i + 1}] Pred: {pred:<6} True: {true:<6} "
            f"conf {confidence[i] * 100:5.1f}%  "
            f"{'OK' if correct else 'WRONG'}"
        )

    correct_total = int((predictions == y_sample).sum())
    print(f"\nCorrect: {correct_total}/{n_samples}")

    fig.suptitle(f"Prediction: {correct_total}/{n_samples} correct")
    fig.tight_layout()

    save_path = f"{OUTPUT_DIR}/prediction_sample.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    test_vgg()