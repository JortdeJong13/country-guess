import matplotlib.pyplot as plt

from .data import geom_to_img


def _create_figure(n_subplots):
    """Create a figure with black background and specified number of subplots."""
    fig, axs = plt.subplots(1, n_subplots, figsize=(10, 5))
    fig.patch.set_facecolor("black")
    return fig, axs


def _plot_image(ax, image, title):
    """Plot a single image with given title and styling."""
    ax.imshow(image, interpolation="nearest", origin="lower", cmap="copper")
    ax.set_title(title, color="white")
    ax.axis("off")


def plot_training_sample(train_data, idx=None):
    """Plot training sample with drawing, positive and negative examples."""
    sample = train_data[idx]

    _, axs = _create_figure(3)

    _plot_image(axs[0], sample["drawing"], "Generated drawing")
    _plot_image(axs[1], sample["pos_img"], "Positive country")
    _plot_image(axs[2], sample["neg_img"], "Negative country")

    plt.show()


def plot_sample(data, idx=None):
    """Plot sample from the test of validation set with drawing and reference shape."""
    sample = data[idx]
    ref_img = data.from_country_name(sample["country_name"])

    _, axs = _create_figure(2)

    _plot_image(axs[0], sample["drawing"], f"Drawing of {sample['country_name']}")
    _plot_image(axs[1], ref_img, "Reference shape")

    plt.show()
