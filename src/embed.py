"""Embedding extraction and UMAP projection.

Loads trained models, extracts 128-dim embeddings from the penultimate FC
layer for a subsample of 1000 test images, then projects to 2D with UMAP.
"""

import numpy as np
import torch
from torchvision import datasets, transforms

from src.model import MnistCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
N_SAMPLES = 1000


def load_test_subsample(n: int = N_SAMPLES, seed: int = SEED):
    """Load MNIST test set and subsample n random points.

    Returns:
        images: Tensor of shape (n, 1, 28, 28), normalized.
        labels: Tensor of shape (n,), original digit labels.
        raw_images: numpy array (n, 28, 28) of unnormalized pixel values [0,1]
                    for thumbnail rendering.
        indices: The indices selected from the test set.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    dataset = datasets.MNIST("data", train=False, download=True, transform=transform)

    rng = np.random.RandomState(seed)
    indices = rng.choice(len(dataset), size=n, replace=False)

    images = torch.stack([dataset[i][0] for i in indices])
    labels = torch.tensor([dataset[i][1] for i in indices])

    # Raw images for thumbnail overlays
    raw_dataset = datasets.MNIST("data", train=False, download=True,
                                 transform=transforms.ToTensor())
    raw_images = np.stack([raw_dataset[i][0].squeeze().numpy() for i in indices])

    return images, labels, raw_images, indices


def extract_embeddings(model: MnistCNN, images: torch.Tensor) -> np.ndarray:
    """Extract 128-dim embeddings from model's penultimate layer.

    Args:
        model: Trained MnistCNN in eval mode.
        images: Batch of input images.

    Returns:
        Numpy array of shape (n, 128).
    """
    model.eval()
    model.to(DEVICE)
    with torch.no_grad():
        embeddings = model.embed(images.to(DEVICE))
    return embeddings.cpu().numpy()


def project_umap(embeddings: np.ndarray, seed: int = SEED) -> np.ndarray:
    """Project embeddings to 2D using UMAP.

    Args:
        embeddings: Array of shape (n, d).
        seed: Random seed for reproducibility.

    Returns:
        Array of shape (n, 2).
    """
    import umap
    reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=15, min_dist=0.1)
    return reducer.fit_transform(embeddings)


def load_model(path: str, num_classes: int) -> MnistCNN:
    """Load a trained model from disk."""
    model = MnistCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


if __name__ == "__main__":
    import os
    os.makedirs("plots", exist_ok=True)

    images, labels, raw_images, indices = load_test_subsample()
    print(f"Loaded {len(images)} test samples")

    # Digit model embeddings
    digit_model = load_model("models/digit_classifier.pt", num_classes=10)
    digit_emb = extract_embeddings(digit_model, images)
    digit_2d = project_umap(digit_emb)
    print(f"Digit embeddings shape: {digit_emb.shape} → UMAP: {digit_2d.shape}")

    # Even/odd model embeddings
    eo_model = load_model("models/even_odd_classifier.pt", num_classes=2)
    eo_emb = extract_embeddings(eo_model, images)
    eo_2d = project_umap(eo_emb)
    print(f"Even/odd embeddings shape: {eo_emb.shape} → UMAP: {eo_2d.shape}")

    # Save for plotting
    np.savez(
        "plots/embedding_data.npz",
        digit_2d=digit_2d,
        eo_2d=eo_2d,
        labels=labels.numpy(),
        raw_images=raw_images,
    )
    print("Saved plots/embedding_data.npz")
