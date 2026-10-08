import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Detect and set execution device (CUDA GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VGG16 = None
VGG_SMALL = None


class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_conv):
        super().__init__()
        layers = []
        for i in range(n_conv):
            layers.append(
                nn.Conv2d(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                )
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class PyTorchVGG(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # VGG16 Architecture Layout
        self.features = nn.Sequential(
            VGGBlock(3, 64, 2),
            VGGBlock(64, 128, 2),
            VGGBlock(128, 256, 3),
            VGGBlock(256, 512, 3),
            VGGBlock(512, 512, 3),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 1 if num_classes == 2 else num_classes),
        )

    def forward(self, x):
        # Convert image layout from NHWC (OpenCV/NumPy) to NCHW (PyTorch) and rescale [0, 255] to [0.0, 1.0]
        x = x.permute(0, 3, 1, 2).float() / 255.0
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    num_classes,
    output_dir=None,
    epochs=50,
    batch_size=32,
    blocks=None,
):

    print(f"\n[PyTorch] Training on Device: {device}")

    # Convert NumPy arrays to PyTorch Tensor datasets
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))

    # drop_last=True helps avoid single-item batch shape mismatch issues during batching
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Instantiate model and transfer weights to target device
    model = PyTorchVGG(num_classes).to(device)

    criterion = (
        nn.BCEWithLogitsLoss() if num_classes == 2 else nn.CrossEntropyLoss()
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    history = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}

    # Model training loop
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            if num_classes == 2:
                # Ensure 1D shape alignment (batch_size,)
                outputs = outputs.view(-1)
                targets_loss = targets.float()
            else:
                targets_loss = targets

            loss = criterion(outputs, targets_loss)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            if num_classes == 2:
                preds = (torch.sigmoid(outputs) > 0.5).long()
            else:
                preds = outputs.argmax(1)

            correct += (preds == targets).sum().item()
            total += targets.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        # Validation evaluation step
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                if num_classes == 2:
                    outputs = outputs.view(-1)
                    targets_loss = targets.float()
                else:
                    targets_loss = targets

                loss = criterion(outputs, targets_loss)

                val_loss += loss.item() * inputs.size(0)

                if num_classes == 2:
                    preds = (torch.sigmoid(outputs) > 0.5).long()
                else:
                    preds = outputs.argmax(1)

                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)

        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total

        history["loss"].append(epoch_loss)
        history["accuracy"].append(epoch_acc)
        history["val_loss"].append(val_epoch_loss)
        history["val_accuracy"].append(val_epoch_acc)

        print(
            f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f} - val_loss: {val_epoch_loss:.4f} - val_acc: {val_epoch_acc:.4f}"
        )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        torch.save(
            model.state_dict(), os.path.join(output_dir, "vgg_model.pt")
        )

    class HistoryWrapper:
        def __init__(self, h):
            self.history = h

    return model, HistoryWrapper(history)


def predict_model(model, X_test):
    model.eval()
    inputs = torch.tensor(X_test).to(device)
    with torch.no_grad():
        outputs = model(inputs)
        if outputs.shape[-1] == 1 or outputs.ndim == 1:
            outputs = outputs.view(-1)
            return (torch.sigmoid(outputs) > 0.5).cpu().numpy().astype(int)
        return outputs.argmax(1).cpu().numpy()