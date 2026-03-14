"""Reusable training pipeline for MNIST CNN models.

Supports arbitrary label mappings so the same pipeline can train both
a digit classifier (10 classes) and an even/odd classifier (2 classes).

Hyperparameters (based on common MNIST CNN best practices):
    - Optimizer: Adam, lr=0.001
    - Batch size: 64
    - Early stopping patience: 10 epochs
    - Uses 10% of training data as validation split for early stopping
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from src.model import MnistCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_train_dataset(label_fn=None):
    """Load MNIST training dataset with optional label remapping.

    Args:
        label_fn: Optional callable that maps original label → new label.
                  e.g. lambda y: 0 if y % 2 == 0 else 1 for even/odd.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
    if label_fn is not None:
        dataset.targets = torch.tensor([label_fn(y.item()) for y in dataset.targets])
    return dataset


def train_model(
    num_classes: int,
    label_fn=None,
    lr: float = 0.001,
    batch_size: int = 64,
    patience: int = 10,
    max_epochs: int = 100,
) -> MnistCNN:
    """Train a CNN model on MNIST with early stopping.

    Args:
        num_classes: Number of output classes.
        label_fn: Optional label remapping function.
        lr: Learning rate for Adam optimizer.
        batch_size: Training batch size.
        patience: Early stopping patience (epochs without val loss improvement).
        max_epochs: Maximum number of training epochs.

    Returns:
        Trained MnistCNN model (on CPU, eval mode).
    """
    dataset = get_train_dataset(label_fn)

    # 90/10 train/val split for early stopping
    val_size = 6000
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=1000, shuffle=False, num_workers=2)

    model = MnistCNN(num_classes=num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= train_size

        # Validate
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                output = model(images)
                val_loss += criterion(output, labels).item() * images.size(0)
                correct += (output.argmax(1) == labels).sum().item()
        val_loss /= val_size
        val_acc = correct / val_size

        print(f"  Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

    model.load_state_dict(best_state)
    model.cpu().eval()
    return model


if __name__ == "__main__":
    import os
    os.makedirs("models", exist_ok=True)

    # Train digit classifier (10 classes)
    print("=" * 60)
    print("Training DIGIT classifier (10 classes)")
    print("=" * 60)
    digit_model = train_model(num_classes=10)
    torch.save(digit_model.state_dict(), "models/digit_classifier.pt")
    print("Saved models/digit_classifier.pt\n")

    # Train even/odd classifier (2 classes)
    print("=" * 60)
    print("Training EVEN/ODD classifier (2 classes)")
    print("=" * 60)
    even_odd_fn = lambda y: 0 if y % 2 == 0 else 1  # 0=even, 1=odd
    eo_model = train_model(num_classes=2, label_fn=even_odd_fn)
    torch.save(eo_model.state_dict(), "models/even_odd_classifier.pt")
    print("Saved models/even_odd_classifier.pt")
