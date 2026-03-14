# Methodology

## Overview

This project demonstrates how the **training objective** shapes what a neural network learns — and therefore what it considers "similar." We train two CNNs on the same MNIST digits dataset but with different tasks, then visualize how each model's internal representations (embeddings) organize the data.

## Architecture

**Model:** A basic CNN with two convolutional blocks followed by a fully-connected head.

```
Input (1×28×28)
  → Conv2d(1→32, 3×3, pad=1) → ReLU
  → Conv2d(32→64, 3×3, pad=1) → ReLU → MaxPool(2) → Dropout(0.25)
  → Flatten (64×14×14 = 12,544)
  → FC(12544→128) → ReLU → Dropout(0.5)    ← embedding layer
  → FC(128→num_classes)                      ← output layer
```

The 128-dimensional penultimate FC layer serves as the **embedding layer** — the learned representation we visualize.

## Training

Two models are trained on the 60,000 MNIST training images:

| Model | Task | Classes | Val Accuracy |
|-------|------|---------|-------------|
| Digit classifier | Predict digit (0–9) | 10 | ~99.1% |
| Even/odd classifier | Predict even vs. odd | 2 | ~99.5% |

### Hyperparameters

- **Optimizer:** Adam (lr=0.001)
- **Batch size:** 64
- **Early stopping:** Patience of 10 epochs on validation loss
- **Validation split:** 10% of training data (6,000 samples)
- **Normalization:** Mean=0.1307, Std=0.3081 (MNIST standard)

These values are based on widely-used MNIST CNN configurations. No hyperparameter search was performed — the goal is a competent model, not state-of-the-art.

## Embedding Extraction

1. Subsample 1,000 random points from the 10,000 MNIST test images (seed=42)
2. Forward-pass through each model, extracting the 128-dim activations after the first FC layer (before the output layer)
3. This gives two sets of 1,000 embeddings — one per model

## Dimensionality Reduction

**UMAP** (Uniform Manifold Approximation and Projection) is used to project 128-dim embeddings to 2D for visualization.

- `n_neighbors=15` (default)
- `min_dist=0.1` (default)
- `random_state=42` for reproducibility

UMAP preserves local neighborhood structure, so points that are close in 128-dim space tend to stay close in 2D.

## Visualization

Four plots are generated:

1. **Digit classifier scatter** — All 1,000 points with digit 7 highlighted in red. The digit model clusters 7s tightly together, separate from other digits.

2. **Even/odd classifier scatter** — Same points, same highlighting. The even/odd model clusters 7s with other odd digits (1, 3, 5, 9) since it doesn't need to distinguish between them.

3. **Digit classifier with thumbnails** — Two 7s that are close in embedding space, showing the model considers them similar (they are the same digit).

4. **Even/odd classifier with thumbnails** — A 7 and a 9 that are close in embedding space. Different digits, but the model treats them the same (both are odd).

## Key Takeaway

> **"Similarity" is defined by the training objective.**
>
> The same image of a 7 is embedded differently depending on what the model was trained to do. The digit model groups it with other 7s. The even/odd model groups it with all odd digits. The embedding space reflects what the model was asked to learn — not some intrinsic truth about the data.
