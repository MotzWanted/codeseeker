import typing as typ
from matplotlib.axes import Axes
import seaborn as sns
import numpy as np


niceblue = "#1F5B93"
nicered = "#BE533B"
nicegreen = "#54AD72"


def compute_calibration_curve(
    probabilities: list[float],
    labels: list[int],
    predictions: list[int],
    n_bins: int,
) -> tuple[list[float], list[float]]:
    """Compute the calibration curve."""
    # Create bins for probabilities
    x = [i / n_bins for i in range(n_bins + 1)]
    y = [0.0] * n_bins

    # Iterate through the bins and compute the empirical fraction
    for j in range(n_bins):
        # Select indices where probabilities fall into the current bin
        bin_indices = [idx for idx, p in enumerate(probabilities) if x[j] <= p < x[j + 1]]

        # Extract predictions and labels for the current bin
        bin_preds = [predictions[idx] for idx in bin_indices]
        bin_labels = [labels[idx] for idx in bin_indices]

        # Compute the empirical fraction (correct predictions) for the current bin
        if bin_preds:
            correct_preds = [1 if pred == label else 0 for pred, label in zip(bin_preds, bin_labels)]
            y[j] = sum(correct_preds) / len(correct_preds)
        else:
            y[j] = 0.0  # Handle empty bin case

    # Calculate midpoints of bins for x-axis
    x_centers = [(x[j] + x[j + 1]) / 2 for j in range(n_bins)]
    return x_centers, y


def plot_probability_distribution(ax: Axes, prob_true: np.ndarray, prob_false: np.ndarray, dset: str, i: int) -> None:
    # plot the distribution
    sns.kdeplot(prob_false, color=nicered, label="Incorrect", fill=True, ax=ax)
    sns.kdeplot(prob_true, color=niceblue, label="Correct", fill=True, ax=ax)
    ax.axvline(np.mean(prob_false), color=nicered, linestyle=":")
    ax.axvline(np.mean(prob_true), color=niceblue, linestyle=":")
    # ax.set_xlim(0.2, 1.2)
    ax.set_ylim(0, 1.0)
    ax.set_title(dset)
    if i == 2:
        ax.legend(loc="upper left")


def plot_calibration_curve(
    ax_cal: Axes, x: np.ndarray, y: np.ndarray, i: int, _type: typ.Literal["positive", "negative"]
) -> None:
    ax_cal.plot(x, y, color="black", marker="o", label="Calibration")
    ax_cal.plot(x, x, color="gray", linestyle=":", label="Perfectly calibrated")
    ax_cal.set_ylim(0.0, 1)
    ax_cal.set_xlim(0.3, 1)
    ax_cal.set_xlabel(r"$\max_{x}\, p(x)$")
    if i == 0:
        ax_cal.set_ylabel(f"Fraction of {_type}")
    if i == 2:
        ax_cal.legend(loc="upper left")


# def plot_binary_calibration_curve(data: list[dict], n_bins: int) -> None:
#     """Plot the binary calibration curve."""
#     df = pd.DataFrame(data)
#     probabilities = df["probabilities"]
#     labels = df["labels_matrix"]
#     predictions = df["sparse_matrix"]

#     negative_mask = labels == 0
#     positive_mask = labels == 1

#     # Compute calibration curves for positive and negative classes
#     x_pos, y_pos = compute_calibration_curve(
#         probabilities[positive_mask], labels[positive_mask], predictions[positive_mask], n_bins
#     )
#     x_neg, y_neg = compute_calibration_curve(
#         probabilities[negative_mask], labels[negative_mask], predictions[negative_mask], n_bins
#     )

#     fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
#     for i, _type in enumerate(["positive", "negative"]):
#         mask = labels == i
#         ax_cal = axes[i]
#         x, y = compute_calibration_curve(
#             dataframe["probabilities"], dataframe["labels"], dataframe["predictions"], n_bins, _type
#         )
#         plot_calibration_curve(ax_cal, x, y, i, _type)
#     plt.tight_layout()
#     plt.savefig("binary-calibration.png", dpi=600)
#     plt.show()


# def plot_results(datadirs: Dict[str, Path], n_samples: Dict[str, int]) -> None:
#     fig, axes = plt.subplots(3, 2, figsize=(9, 6), sharey="row", sharex="col")  # Adjust layout to 3x2

#     for i, (dset, datadir) in enumerate(datadirs.items()):
#         ax = axes[0, i]
#         ax_cal = axes[1, i]
#         ax_cal_neg = axes[2, i]  # New row for negative calibration

#         # Load data
#         probs, labels, preds = load_data(datadir, n_samples, dset)

#         # Compute probabilities
#         prob_true, prob_false = compute_probabilities(probs, labels, preds)

#         # Plot distribution
#         plot_distribution(ax, prob_true, prob_false, dset, i)

#         # Compute and plot calibration curve
#         x, y = compute_calibration_curve(probs, labels, preds, n_bins)
#         plot_positive_calibration_curve(ax_cal, x, y, i)
#         plot_negative_calibration_curve(ax_cal_neg, x, y, i)  # New negative calibration plot

#     # Adjust layout and save the plot
#     plt.tight_layout()
#     plt.savefig("med-uncertainty-k40-negative-calibration.png", dpi=600)
#     plt.show()
