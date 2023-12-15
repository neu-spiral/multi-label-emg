import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import plotly.graph_objects as go
from loguru import logger
from scipy.spatial.distance import cdist, pdist

from multi_label_emg.data import load_data_dict
from multi_label_emg.train import (
    balance_classes,
    get_augmented_doubles,
    get_augmented_singles,
    remove_double_gestures,
)
from multi_label_emg.utils import NO_DIR_IDX, NO_MOD_IDX, PROJECT_ROOT, canonical_coords

HEATMAP_WIDTH = 1000
HEATMAP_HEIGHT = 1000

PROJECTION_WIDTH = 1000
PROJECTION_HEIGHT = 1000

SCALE = 2
FONT_SIZE = 18


def plot_heatmap(data: np.ndarray, ticktext: List[str], tril: bool = False):
    def make_text(cm):
        text = []
        for v in cm.flatten():
            text.append(f"{round(v, 2)}")
        return np.array(text).reshape(cm.shape)

    if tril:
        data = np.copy(data)
        data[np.triu_indices(data.shape[0], k=1)] = None

    text = make_text(data)

    fig = go.Figure()
    fig.update_layout(
        # margin=margin,
        template="simple_white",
        xaxis=dict(
            tickangle=-45,
            tickmode="array",
            ticktext=ticktext,
            tickvals=list(range(len(ticktext))),
            constrain="domain",
        ),
        yaxis=dict(
            tickmode="array",
            ticktext=ticktext,
            tickvals=list(range(len(ticktext))),
            autorange="reversed",
            scaleanchor="x",
            scaleratio=1,
            constrain="domain",
        ),
        width=HEATMAP_WIDTH,
        height=HEATMAP_HEIGHT,
    )
    fig.add_trace(
        go.Heatmap(z=data, text=text, texttemplate="%{text}", zmin=0, zmax=1, colorscale="Blues", showscale=False)
    )
    return fig


def compute_feature_similarity_no_labels(features: List[np.ndarray], method: str, gamma: float) -> np.ndarray:
    """Computes some notion of similarity between each pair of classes in feature space.

    Returns:
        np.ndarray: similarity between each pair of classes
    """
    if not isinstance(features, list):
        raise ValueError("features must be list of chunks of data")
    # Start with NaN so that missing gestures will not skew towards default value.
    # NOTE - Later, need to use functions like np.nanmean(), etc
    n_class = len(features)
    pairwise_similarity = np.nan * np.ones((n_class, n_class))

    if method == "rbf":
        for (idx1, feat1), (idx2, feat2) in combinations(enumerate(features), 2):
            # Match rows based on their full label vector
            rbf_similarities = np.exp(-gamma * cdist(feat1, feat2, "sqeuclidean"))
            pairwise_similarity[idx1, idx2] = pairwise_similarity[idx2, idx1] = rbf_similarities.mean()
        # Fill diagonal.  different because we must ignore zeros due to d(x1, x1) == 0
        for idx, feat in enumerate(features):
            if len(feat) <= 1:
                continue
            rbf_similarities = np.exp(-gamma * pdist(feat, "sqeuclidean"))
            pairwise_similarity[idx, idx] = rbf_similarities.mean()
    else:
        raise NotImplementedError()

    return pairwise_similarity


def compute_feature_similarity_timeseries(
    timeseries_features: np.ndarray, static_features: List[np.ndarray], method: str, gamma: float = 50.0
) -> np.ndarray:
    """
    For each point in the timeseries, compute the avg RBF similarity to points in a static class.

    Result:
        np.ndarray, shape (n_static_classes, n_time_steps)
    """
    assert isinstance(timeseries_features, np.ndarray)
    assert isinstance(static_features, list)

    n_time_steps, n_features = timeseries_features.shape

    # For each class that we are comparing to, we will obtain a similarity curve
    result = np.nan * np.ones((len(static_features), n_time_steps))
    # Start with NaN so that missing gestures will not skew towards default value.
    # NOTE - Later, need to use functions like np.nanmean(), etc

    if method == "rbf":
        result = []
        for static_feat in static_features:
            # Compare this particular class to the timeseries.
            # For each point in the timeseries, we want the avg similarity to points in the class.
            result.append(np.exp(-gamma * cdist(timeseries_features, static_feat, metric="sqeuclidean")).mean(1))
        result = np.array(result)
    else:
        raise NotImplementedError()

    return result


def make_similarity_matrix(
    real_features: np.ndarray,
    real_dir_labels: np.ndarray,
    real_mod_labels: np.ndarray,
    fake_features: np.ndarray,
    fake_dir_labels: np.ndarray,
    fake_mod_labels: np.ndarray,
    gamma: Union[float, str],
):
    # make a matrix of shape (15+8, 15+8)
    # where we compare all real classes to the fake doubles classes
    # group the data by class
    real_labels_2d = np.stack([real_dir_labels.argmax(-1), real_mod_labels.argmax(-1)], axis=1)
    fake_labels_2d = np.stack([fake_dir_labels.argmax(-1), fake_mod_labels.argmax(-1)], axis=1)

    coords, coords_str = canonical_coords()
    # Add all the real features in groups, 1 class at a time

    ticktext = []
    all_features_grouped = []
    for (d, m), s in zip(coords, coords_str):
        idx = (real_labels_2d == (d, m)).all(-1)
        all_features_grouped.append(real_features[idx])
        ticktext.append(f"{s} - Real")

    # Add the fake doubles, 1 class at a time
    for (d, m), s in zip(coords, coords_str):
        # If this is a singles class, skip
        if d == NO_DIR_IDX or m == NO_MOD_IDX:
            continue
        idx = (fake_labels_2d == (d, m)).all(-1)
        all_features_grouped.append(fake_features[idx])
        ticktext.append(f"{s} - Fake")

    # compute similarity between each pair of classes
    if gamma == "median":
        gamma = 1 / np.median(pdist(np.concatenate(all_features_grouped), "sqeuclidean"))
    similarity_matrix = compute_feature_similarity_no_labels(all_features_grouped, method="rbf", gamma=gamma)
    return similarity_matrix, ticktext


def make_heatmap_plot(similarity_matrix, ticktext):
    fig = plot_heatmap(similarity_matrix, ticktext, tril=True)
    full_fig = fig.full_figure_for_development(warn=False)
    x_lo, x_hi = full_fig.layout.xaxis.range
    y_hi, y_lo = full_fig.layout.yaxis.range  # NOTE - y-axis range is reversed for heatmap

    n_classes = len(ticktext)
    box_size = (y_hi - y_lo) / n_classes

    # Add a line after the single gesture classes
    def add_hline(n):
        # Line from the y-axis, travling horizontall, until it hits the diagonal
        x = [x_lo, x_lo + n * box_size]
        # compute y-values in the normal way
        y = [y_hi - n * box_size, y_hi - n * box_size]
        # Then adjust y values to account for reversed axis
        y = [y_hi - y_ + y_lo for y_ in y]
        fig.add_trace(
            go.Scatter(x=x, y=y, mode="lines", line=dict(color="black", dash="dot", width=4), showlegend=False)
        )

    def add_vline(n):
        # Line from the diagonal, traveling vertically down, until it hits x-axis
        # after moving over n boxes, the y value of the diagonal is
        x = [x_lo + n * box_size, x_lo + n * box_size]
        # compute y-values in the normal way
        y = [y_hi - n * box_size, y_lo]
        # Then adjust y values to account for reversed axis
        y = [y_hi - y_ + y_lo for y_ in y]
        fig.add_trace(
            go.Scatter(x=x, y=y, mode="lines", line=dict(color="black", dash="dot", width=4), showlegend=False)
        )

    # Add lines for easier interpretation
    add_hline(6)
    add_hline(15)
    add_vline(6)
    add_vline(15)

    fig.update_layout(
        title="Average RBF Similarity",
        template="simple_white",
        height=HEATMAP_HEIGHT,
        width=HEATMAP_WIDTH,
        boxmode="group",
        title_x=0.5,
        title_y=0.95,
        xaxis_range=[x_lo, x_hi],
        yaxis_range=[y_hi, y_lo],
        yaxis_autorange=False,
        margin=dict(l=0, r=0, t=0, b=0),
        font_size=FONT_SIZE,
    )

    return fig


def prepare_data(data: Dict, singles_method, rel_frac_singles_per_class, doubles_method, frac_doubles_per_class):
    calib_features = data["Calibration_features"]
    calib_dir_labels = data["Calibration_dir_labels"]
    calib_mod_labels = data["Calibration_mod_labels"]

    # Remove any double gestures that occured due to bad participant behavior
    calib_features, calib_dir_labels, calib_mod_labels = remove_double_gestures(
        calib_features, calib_dir_labels, calib_mod_labels
    )
    # Subset the "Rest" class so that classes have the same number of samples
    calib_features, calib_dir_labels, calib_mod_labels = balance_classes(
        calib_features, calib_dir_labels, calib_mod_labels
    )

    # make augmented doubles
    double_features_aug, double_dir_labels_aug, double_mod_labels_aug = get_augmented_doubles(
        doubles_method, "avg", frac_doubles_per_class, calib_features, calib_dir_labels, calib_mod_labels
    )

    n_singles_per_class = 0
    if singles_method != "none":
        doubles_labels_2d = np.stack((double_dir_labels_aug.argmax(-1), double_mod_labels_aug.argmax(-1)), axis=-1)
        class_sizes = np.unique(doubles_labels_2d, axis=0, return_counts=True)[-1]
        n_singles_per_class = int(np.round(np.mean(class_sizes) * rel_frac_singles_per_class))

    # Make augmented singles
    single_features_aug, single_dir_labels_aug, single_mod_labels_aug = get_augmented_singles(
        singles_method, n_singles_per_class, calib_features, calib_dir_labels, calib_mod_labels
    )

    # Load all the rest of their data
    # Concatenate all the real data into one collection, and all the fake data into another.
    task_features, task_dir_labels, task_mod_labels = [], [], []
    all_block_prefixes = []
    for i in [1, 2, 3]:
        all_block_prefixes.append(f"HoldPulse{i}_WithFeedBack")
        all_block_prefixes.append(f"SimultaneousPulse{i}_WithFeedBack")
    for block_prefix in all_block_prefixes:
        task_features.append(data[f"{block_prefix}_features"])
        task_dir_labels.append(data[f"{block_prefix}_dir_labels"])
        task_mod_labels.append(data[f"{block_prefix}_mod_labels"])
    task_features = np.concatenate(task_features, axis=0)
    task_dir_labels = np.concatenate(task_dir_labels, axis=0)
    task_mod_labels = np.concatenate(task_mod_labels, axis=0)

    logger.info(f"Real shapes: {task_features.shape=}, {task_dir_labels.shape=}, {task_mod_labels.shape=}")
    shapes = f"{double_features_aug.shape=}, {double_dir_labels_aug.shape=}, {double_mod_labels_aug.shape=}"
    logger.info(f"Synthetic doubles shapes: {shapes}")
    shapes = f"{single_features_aug.shape=}, {single_dir_labels_aug.shape=}, {single_mod_labels_aug.shape=}"
    logger.info(f"Aug singles shapes: {shapes}")

    # items from calibration
    calib = (calib_features, calib_dir_labels, calib_mod_labels)
    # real items from all combos blocks
    task = (task_features, task_dir_labels, task_mod_labels)
    fake_doubles = (double_features_aug, double_dir_labels_aug, double_mod_labels_aug)
    aug_singles = (single_features_aug, single_dir_labels_aug, single_mod_labels_aug)
    return calib, task, fake_doubles, aug_singles


def main(
    subjects: List[str],
    singles_method: str,
    rel_frac_singles_per_class: float,
    doubles_method: str,
    frac_doubles_per_class: float,
    output_dir: Path,
    gamma: Union[float, str],
):
    # load all data
    logger.info("Load data...")
    data_dict = load_data_dict()

    "__".join(map(str, [singles_method, rel_frac_singles_per_class, doubles_method, frac_doubles_per_class]))
    all_heatmaps = []
    for subj in subjects:
        logger.info(f"Prepare data for subject: {subj}...")
        calib, task, fake_doubles, aug_singles = prepare_data(
            data_dict[subj], singles_method, rel_frac_singles_per_class, doubles_method, frac_doubles_per_class
        )
        calib_features, calib_dir_labels, calib_mod_labels = calib
        task_features, task_dir_labels, task_mod_labels = task
        double_features_aug, double_dir_labels_aug, double_mod_labels_aug = fake_doubles
        # single_features_aug, single_dir_labels_aug, single_mod_labels_aug = aug_singles
        # NOTE - for now, we don't use any aug singles here.

        # make heatmap
        logger.info("Make heatmap...")
        heatmap_similarity_matrix, heatmap_ticktext = make_similarity_matrix(
            real_features=np.concatenate((calib_features, task_features)),
            real_dir_labels=np.concatenate((calib_dir_labels, task_dir_labels)),
            real_mod_labels=np.concatenate((calib_mod_labels, task_mod_labels)),
            fake_features=double_features_aug,
            fake_dir_labels=double_dir_labels_aug,
            fake_mod_labels=double_mod_labels_aug,
            gamma=gamma,
        )
        heatmap_fig = make_heatmap_plot(heatmap_similarity_matrix, heatmap_ticktext)
        heatmap_fig.write_image(output_dir / f"{subj}_heatmap.png", scale=2)
        heatmap_fig.write_html(output_dir / f"{subj}_heatmap.html", include_plotlyjs="cdn")

        all_heatmaps.append(heatmap_similarity_matrix)

    logger.info("Make average heatmap...")
    avg_heatmap = np.mean(all_heatmaps, axis=0)
    avg_heatmap_fig = make_heatmap_plot(avg_heatmap, heatmap_ticktext)
    avg_heatmap_fig.write_image(output_dir / "avg_heatmap.png", scale=2)
    avg_heatmap_fig.write_html(output_dir / "avg_heatmap.html", include_plotlyjs="cdn")


if __name__ == "__main__":
    import argparse

    logger.remove()
    logger.add(sys.stderr, level="INFO", colorize=True)

    parser = argparse.ArgumentParser()
    doubles_methods = [
        "none",
        "subset_uniform",
        "subset_near_mean",
        "subset_spaced_quantiles",
        "subsetInput_uniform",
        "subsetInput_near_mean",
        "subsetInput_spaced_quantiles",
        "all",
    ]
    parser.add_argument("--doubles_method", type=str, choices=doubles_methods, required=True)
    parser.add_argument("--frac_doubles_per_class", type=float, default=1.0)
    singles_methods = [
        "none",
        "add-gaussian-0.05",
        "add-gaussian-0.1",
        "add-gaussian-0.2",
        "add-gaussian-0.3",
        "add-gaussian-0.4",
        "add-gaussian-0.5",
        "add-gaussian-0.6",
        "fit-gaussian",
        "fit-gmm-3",
        "fit-gmm-5",
        "fit-gmm-10",
        "fit-kde-gaussian-scott",
        "fit-kde-gaussian-silverman",
        "fit-kde-gaussian-0.01",
        "fit-kde-gaussian-0.1",
        "fit-kde-gaussian-1.0",
        "fit-kde-gaussian-10.0",
    ]
    parser.add_argument("--singles_method", type=str, choices=singles_methods, required=True)
    parser.add_argument("--rel_frac_singles_per_class", type=float, default=1.0)
    parser.add_argument("--gamma", default=50.0)
    args = parser.parse_args()
    print(args)

    output_dir = PROJECT_ROOT.parent / "results" / "figures" / f"feature_similarity_gamma={args.gamma}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # NOTE - seed affects how data is subsampled to balance classes
    np.random.seed(0)

    # Get list of subjects to process
    subjects = [f"Subj{i}" for i in range(11)]
    if args.gamma != "median":
        args.gamma = float(args.gamma)
    main(
        subjects=subjects,
        singles_method=args.singles_method,
        rel_frac_singles_per_class=args.rel_frac_singles_per_class,
        doubles_method=args.doubles_method,
        frac_doubles_per_class=args.frac_doubles_per_class,
        output_dir=output_dir,
        gamma=args.gamma,
    )

    print("finished")
