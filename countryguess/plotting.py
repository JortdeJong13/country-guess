"""Plotting functions for the country guess app."""

import matplotlib.pyplot as plt


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


def plot_evaluation(country_names, ranking, conf_scores):
    """Plot the ranking and the confidence score for each country."""
    plt.figure(figsize=(8, 4 + len(country_names) / 5))

    # Group rankings by country
    country_to_rankings = {country: ([], []) for country in set(country_names)}
    for country, rank, score in zip(country_names, ranking, conf_scores):
        country_to_rankings[country][0].append(rank + 1)
        country_to_rankings[country][1].append(score)

    # Calculate total number of bars and positions
    current_pos = 0
    y_ticks = []
    y_labels = []

    for country, item in country_to_rankings.items():
        rankings, scores = item
        country_center = current_pos + (len(rankings) - 1) / 2
        y_ticks.append(country_center)
        y_labels.append(country)

        for rank, score in zip(rankings, scores):
            # Color coding
            if rank == 1:
                color = "green"
            elif rank <= 10:
                color = "yellow"
            else:
                color = "red"

            plt.barh(current_pos, rank, height=0.85, color=color)
            plt.text(
                min(rank, 9) + 0.2,
                current_pos,
                f"{100 * score:.1f}%",
                va="center",
                fontsize=10,
            )
            current_pos += 1
        current_pos += 0.3

    # Formatting
    plt.xlabel("Rank", fontsize=14)
    plt.ylabel("Countries", fontsize=14)
    plt.title("Test data ranking with confidence", fontsize=16)
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.xlim(0, 10)
    plt.xticks(range(1, 11))
    plt.yticks(y_ticks, y_labels, fontsize=12)
    plt.ylim(current_pos - 0.3 - 0.5, -0.5)
    plt.tight_layout()
    plt.show()
