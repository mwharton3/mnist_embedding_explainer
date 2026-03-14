"""Plotting module for MNIST embedding visualizations.

This module generates publication-quality scatterplots of UMAP-projected
embeddings. It is designed to be configurable and separate from the
data pipeline so you can iterate on visual style independently.

Main entry points:
    - plot_highlight_scatter(): Basic scatterplot with digit 7 highlighted.
    - plot_with_thumbnails(): Scatterplot with sample digit images overlaid.
    - generate_all_plots(): Convenience function to produce all four figures.

Style configuration is centralized in the STYLE dict at the top of this file.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# ---------------------------------------------------------------------------
# Style configuration — tweak these to change the look of all plots.
# ---------------------------------------------------------------------------
STYLE = {
    # Figure dimensions (inches)
    "figsize": (8, 8),

    # Dot appearance
    "dot_size": 12,
    "dot_alpha_bg": 0.35,
    "dot_alpha_fg": 0.85,
    "color_bg": "#B0B0B0",
    "color_fg": "#E63946",

    # Thumbnail overlay settings
    "thumb_zoom": 0.8,
    "thumb_border_color": "#264653",
    "thumb_border_width": 1.5,
    "line_color": "#264653",
    "line_width": 1.2,
    "line_alpha": 0.7,

    # Typography
    "title_fontsize": 16,
    "title_fontweight": "bold",
    "subtitle_fontsize": 11,
    "label_fontsize": 10,

    # Layout
    "dpi": 200,
    "bg_color": "#FAFAFA",
    "spine_color": "#DDDDDD",
}


def _setup_axes(ax, title: str, subtitle: str = ""):
    """Apply consistent styling to an axes object.

    Args:
        ax: Matplotlib Axes.
        title: Main title string.
        subtitle: Optional subtitle displayed below the title.
    """
    ax.set_facecolor(STYLE["bg_color"])
    ax.set_title(
        title,
        fontsize=STYLE["title_fontsize"],
        fontweight=STYLE["title_fontweight"],
        pad=18 if subtitle else 12,
        color="#1D3557",
    )
    if subtitle:
        ax.text(
            0.5, 1.01, subtitle,
            transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=STYLE["subtitle_fontsize"],
            color="#457B9D",
            style="italic",
        )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(STYLE["spine_color"])
        spine.set_linewidth(0.5)


def plot_highlight_scatter(
    coords_2d: np.ndarray,
    labels: np.ndarray,
    title: str,
    subtitle: str = "",
    highlight_digit: int = 7,
    save_path: str | None = None,
):
    """Generate a scatterplot with one digit class highlighted in color.

    All points are rendered in a muted gray except for the target digit,
    which is shown in a vivid accent color.

    Args:
        coords_2d: UMAP coordinates, shape (n, 2).
        labels: Original digit labels, shape (n,).
        title: Plot title.
        subtitle: Optional subtitle.
        highlight_digit: Which digit to highlight (default 7).
        save_path: If provided, save the figure to this path.

    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=STYLE["figsize"], facecolor=STYLE["bg_color"])
    _setup_axes(ax, title, subtitle)

    mask = labels == highlight_digit
    # Background points
    ax.scatter(
        coords_2d[~mask, 0], coords_2d[~mask, 1],
        s=STYLE["dot_size"],
        c=STYLE["color_bg"],
        alpha=STYLE["dot_alpha_bg"],
        edgecolors="none",
        zorder=1,
    )
    # Highlighted points
    ax.scatter(
        coords_2d[mask, 0], coords_2d[mask, 1],
        s=STYLE["dot_size"] * 1.8,
        c=STYLE["color_fg"],
        alpha=STYLE["dot_alpha_fg"],
        edgecolors="white",
        linewidths=0.3,
        zorder=2,
        label=f"Digit {highlight_digit}",
    )
    ax.legend(
        loc="lower right",
        fontsize=STYLE["label_fontsize"],
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        edgecolor=STYLE["spine_color"],
    )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=STYLE["dpi"], bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Saved {save_path}")
    return fig


def _add_thumbnail(ax, image: np.ndarray, xy_point, xy_offset, label_text: str = ""):
    """Overlay a digit thumbnail on the plot with a connecting line.

    Args:
        ax: Matplotlib Axes.
        image: 28×28 grayscale image array (values in [0, 1]).
        xy_point: (x, y) coordinates of the data point.
        xy_offset: (x, y) coordinates where the thumbnail should be placed.
        label_text: Optional text label below the thumbnail.
    """
    im = OffsetImage(
        image, cmap="gray_r", zoom=STYLE["thumb_zoom"],
    )
    im.image.axes = ax

    ab = AnnotationBbox(
        im, xy_offset,
        frameon=True,
        bboxprops=dict(
            edgecolor=STYLE["thumb_border_color"],
            linewidth=STYLE["thumb_border_width"],
            facecolor="white",
            boxstyle="round,pad=0.1",
        ),
        zorder=10,
    )
    ax.add_artist(ab)

    # Connecting line
    ax.annotate(
        "",
        xy=xy_point,
        xytext=xy_offset,
        arrowprops=dict(
            arrowstyle="-",
            color=STYLE["line_color"],
            linewidth=STYLE["line_width"],
            alpha=STYLE["line_alpha"],
        ),
        zorder=9,
    )

    # Data-point marker
    ax.scatter(
        [xy_point[0]], [xy_point[1]],
        s=60, c=STYLE["thumb_border_color"],
        marker="o", edgecolors="white", linewidths=1.0, zorder=11,
    )

    if label_text:
        ax.text(
            xy_offset[0], xy_offset[1] - 3.0,
            label_text,
            ha="center", va="top",
            fontsize=9, fontweight="bold",
            color=STYLE["thumb_border_color"],
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
            zorder=12,
        )


def plot_with_thumbnails(
    coords_2d: np.ndarray,
    labels: np.ndarray,
    raw_images: np.ndarray,
    title: str,
    subtitle: str = "",
    highlight_digit: int = 7,
    thumbnail_indices: list[int] = None,
    thumbnail_labels: list[str] = None,
    save_path: str | None = None,
):
    """Scatterplot with digit thumbnail overlays linked to data points.

    This is the 'experimental' plot that shows actual MNIST digit images
    overlaid on the embedding space, connected to their data points by
    line segments.

    Args:
        coords_2d: UMAP coordinates, shape (n, 2).
        labels: Original digit labels, shape (n,).
        raw_images: Unnormalized images (n, 28, 28) for thumbnails.
        title: Plot title.
        subtitle: Optional subtitle.
        highlight_digit: Which digit to highlight.
        thumbnail_indices: List of indices (into the subsample) to show as thumbnails.
        thumbnail_labels: List of text labels for each thumbnail.
        save_path: If provided, save figure to this path.

    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=STYLE["figsize"], facecolor=STYLE["bg_color"])
    _setup_axes(ax, title, subtitle)

    mask = labels == highlight_digit
    ax.scatter(
        coords_2d[~mask, 0], coords_2d[~mask, 1],
        s=STYLE["dot_size"],
        c=STYLE["color_bg"],
        alpha=STYLE["dot_alpha_bg"],
        edgecolors="none",
        zorder=1,
    )
    ax.scatter(
        coords_2d[mask, 0], coords_2d[mask, 1],
        s=STYLE["dot_size"] * 1.8,
        c=STYLE["color_fg"],
        alpha=STYLE["dot_alpha_fg"],
        edgecolors="white",
        linewidths=0.3,
        zorder=2,
        label=f"Digit {highlight_digit}",
    )

    if thumbnail_indices is not None:
        x_min, x_max = coords_2d[:, 0].min(), coords_2d[:, 0].max()
        y_min, y_max = coords_2d[:, 1].min(), coords_2d[:, 1].max()
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2

        for i, idx in enumerate(thumbnail_indices):
            pt = coords_2d[idx]
            # Push thumbnail away from data center so it doesn't overlap
            dx = 1.0 if pt[0] >= x_center else -1.0
            dy = 1.0 if pt[1] >= y_center else -1.0
            # Alternate: first goes upper, second goes lower-opposite
            if i % 2 == 1:
                dx = -dx
            offset = (
                np.clip(pt[0] + dx * 0.18 * x_range, x_min + 0.05 * x_range, x_max - 0.05 * x_range),
                np.clip(pt[1] + dy * 0.18 * y_range, y_min + 0.05 * y_range, y_max - 0.05 * y_range),
            )
            lbl = thumbnail_labels[i] if thumbnail_labels else ""
            _add_thumbnail(ax, raw_images[idx], tuple(pt), offset, label_text=lbl)

    ax.legend(
        loc="lower right",
        fontsize=STYLE["label_fontsize"],
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        edgecolor=STYLE["spine_color"],
    )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=STYLE["dpi"], bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Saved {save_path}")
    return fig


def find_similar_pair(coords_2d, labels, digit, n_closest=1):
    """Find two points of the same digit that are close together in UMAP space.

    Args:
        coords_2d: UMAP coordinates.
        labels: Digit labels.
        digit: Target digit.
        n_closest: Return the n-th closest pair (1 = closest).

    Returns:
        Tuple of (idx_a, idx_b) into the subsample array.
    """
    mask = np.where(labels == digit)[0]
    from scipy.spatial.distance import cdist
    dists = cdist(coords_2d[mask], coords_2d[mask])
    np.fill_diagonal(dists, np.inf)
    flat_idx = np.argsort(dists.ravel())
    # Pick the n_closest-th pair
    pair_flat = flat_idx[2 * (n_closest - 1)]  # factor of 2 for symmetry
    i, j = np.unravel_index(pair_flat, dists.shape)
    return mask[i], mask[j]


def find_different_pair_close_in_embedding(coords_2d, labels, digit_a=7, digit_b_parity="odd"):
    """Find two different-digit points that are close in UMAP space.

    For the even/odd model, this finds a digit_a point and a point with
    the specified parity that are close together — showing that the model
    considers them 'similar' despite being different digits.

    Args:
        coords_2d: UMAP coordinates.
        labels: Digit labels.
        digit_a: First digit (default 7).
        digit_b_parity: 'odd' or 'even' — the parity of the second point.

    Returns:
        Tuple of (idx_a, idx_b).
    """
    mask_a = np.where(labels == digit_a)[0]
    if digit_b_parity == "odd":
        odd_digits = {1, 3, 5, 9}  # exclude 7 itself
        mask_b = np.where(np.isin(labels, list(odd_digits)))[0]
    else:
        even_digits = {0, 2, 4, 6, 8}
        mask_b = np.where(np.isin(labels, list(even_digits)))[0]

    from scipy.spatial.distance import cdist
    dists = cdist(coords_2d[mask_a], coords_2d[mask_b])
    flat_min = np.argmin(dists)
    i, j = np.unravel_index(flat_min, dists.shape)
    return mask_a[i], mask_b[j]


def generate_all_plots(data_path: str = "plots/embedding_data.npz", output_dir: str = "plots"):
    """Generate all four plots from saved embedding data.

    Produces:
        1. digit_scatter.png — Digit model, 7s highlighted
        2. evenodd_scatter.png — Even/odd model, 7s highlighted
        3. digit_thumbnails.png — Digit model with two similar 7s overlaid
        4. evenodd_thumbnails.png — Even/odd model with a 7 and another odd digit

    Args:
        data_path: Path to the .npz file from embed.py.
        output_dir: Directory to write PNG files.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    data = np.load(data_path)
    digit_2d = data["digit_2d"]
    eo_2d = data["eo_2d"]
    labels = data["labels"]
    raw_images = data["raw_images"]

    print("Generating plots...")

    # Plot 1: Digit model scatter
    plot_highlight_scatter(
        digit_2d, labels,
        title="Digit Classifier Embeddings",
        subtitle="CNN trained to recognize digits 0–9 · UMAP projection",
        save_path=os.path.join(output_dir, "digit_scatter.png"),
    )
    plt.close()

    # Plot 2: Even/odd model scatter
    plot_highlight_scatter(
        eo_2d, labels,
        title="Even/Odd Classifier Embeddings",
        subtitle="CNN trained to classify even vs. odd · UMAP projection",
        save_path=os.path.join(output_dir, "evenodd_scatter.png"),
    )
    plt.close()

    # Plot 3: Digit model with two similar 7s
    idx_a, idx_b = find_similar_pair(digit_2d, labels, digit=7, n_closest=5)
    plot_with_thumbnails(
        digit_2d, labels, raw_images,
        title="Digit Classifier — Similar Embeddings",
        subtitle="Two 7s that are close in embedding space (both recognized as the same digit)",
        thumbnail_indices=[idx_a, idx_b],
        thumbnail_labels=[f"Digit {labels[idx_a]}", f"Digit {labels[idx_b]}"],
        save_path=os.path.join(output_dir, "digit_thumbnails.png"),
    )
    plt.close()

    # Plot 4: Even/odd model with 7 and another odd digit nearby
    idx_a, idx_b = find_different_pair_close_in_embedding(
        eo_2d, labels, digit_a=7, digit_b_parity="odd"
    )
    plot_with_thumbnails(
        eo_2d, labels, raw_images,
        title="Even/Odd Classifier — Similar Embeddings",
        subtitle="A 7 and another odd digit nearby (same class to this model, different digits)",
        thumbnail_indices=[idx_a, idx_b],
        thumbnail_labels=[f"Digit {labels[idx_a]}", f"Digit {labels[idx_b]}"],
        save_path=os.path.join(output_dir, "evenodd_thumbnails.png"),
    )
    plt.close()

    print("All plots generated!")


if __name__ == "__main__":
    generate_all_plots()
