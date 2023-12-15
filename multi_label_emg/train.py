import sys

import numpy as np
import plotly.graph_objects as go
from loguru import logger
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity, KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

from multi_label_emg.data import load_data_dict
from multi_label_emg.models import AvgPairs, ElementwiseMaxPairs, ParallelA, ParallelB
from multi_label_emg.utils import (
    NO_DIR_IDX,
    NO_MOD_IDX,
    RESULTS_DIR,
    canonical_coords,
    confusion_matrix,
    str2bool,
)


def get_name(
    subject: str,
    seed: int,
    parallel_model_type: str,
    clf_name: str,
    doubles_method: str,
    fraction_doubles_per_class: float,
    singles_method: str,
    rel_fraction_singles_per_class: float,
    include_doubles_in_train: bool,
    feature_combine_type: str,
):
    return "__".join(
        [
            f"subj={subject}",
            f"seed={seed}",
            f"par={parallel_model_type}",
            f"clf={clf_name}",
            f"doubles={doubles_method}",
            f"frac_doubles={fraction_doubles_per_class}",
            f"singles={singles_method}",
            f"frac_singles={rel_fraction_singles_per_class}",
            f"incl_doubles={include_doubles_in_train}",
            f"feat_type={feature_combine_type}",
        ]
    )


def plot_confusion_matrix(data: np.ndarray):
    def make_text(cm):
        text = []
        for v in cm.flatten():
            text.append(f"{round(v, 2)}")
        return np.array(text).reshape(cm.shape)

    coords, coords_str = canonical_coords()
    text = make_text(data)

    fig = go.Figure()
    fig.update_layout(
        # margin=margin,
        xaxis=dict(
            title="Predicted",
            tickangle=-45,
            tickmode="array",
            ticktext=coords_str,
            tickvals=list(range(len(coords_str))),
            constrain="domain",
        ),
        yaxis=dict(
            title="Actual",
            tickmode="array",
            ticktext=coords_str,
            tickvals=list(range(len(coords_str))),
            autorange="reversed",
            scaleanchor="x",
            scaleratio=1,
            constrain="domain",
        ),
    )
    fig.add_trace(
        go.Heatmap(z=data, text=text, texttemplate="%{text}", zmin=0, zmax=1, colorscale="Blues", showscale=False)
    )
    return fig


def subset_doubles_uniform(
    n_per_class: int, features_aug: np.ndarray, dir_labels_aug: np.ndarray, mod_labels_aug: np.ndarray
):
    """For each class, take n_per_class items uniformly at random"""
    res_x, res_y_dir, res_y_mod = [], [], []
    labels_2d = np.stack([dir_labels_aug.argmax(-1), mod_labels_aug.argmax(-1)], axis=-1)
    for d, m in np.unique(labels_2d, axis=0):
        idx = np.where((labels_2d == (d, m)).all(-1))[0]
        subset_idx = np.random.choice(idx, size=n_per_class, replace=False)
        res_x.append(features_aug[subset_idx])
        res_y_dir.append(dir_labels_aug[subset_idx])
        res_y_mod.append(mod_labels_aug[subset_idx])

    features_aug = np.concatenate(res_x)
    dir_labels_aug = np.concatenate(res_y_dir)
    mod_labels_aug = np.concatenate(res_y_mod)

    return features_aug, dir_labels_aug, mod_labels_aug


def subset_doubles_near_mean(
    n_per_class: int, features_aug: np.ndarray, dir_labels_aug: np.ndarray, mod_labels_aug: np.ndarray
):
    """For each class, take n_per_class items closest to the mean of these synthetic items"""
    # Find class means
    class_means = {}
    labels_2d = np.stack([dir_labels_aug.argmax(-1), mod_labels_aug.argmax(-1)], axis=-1)

    for d, m in np.unique(labels_2d, axis=0):
        idx = np.where((labels_2d == (d, m)).all(-1))[0]
        class_means[(d, m)] = np.mean(features_aug[idx], axis=0)

    # Subset each class by taking items closest to mean
    res_x, res_y_dir, res_y_mod = [], [], []
    for d, m in np.unique(labels_2d, axis=0):
        class_mean = class_means[(d, m)]
        idx = np.where((labels_2d == (d, m)).all(-1))[0]
        dists = np.linalg.norm(features_aug[idx] - class_mean, axis=-1)
        k_smallest_idx = np.argpartition(dists, n_per_class)[:n_per_class]

        subset_idx = idx[k_smallest_idx]
        res_x.append(features_aug[subset_idx])
        res_y_dir.append(dir_labels_aug[subset_idx])
        res_y_mod.append(mod_labels_aug[subset_idx])

    features_aug = np.concatenate(res_x)
    dir_labels_aug = np.concatenate(res_y_dir)
    mod_labels_aug = np.concatenate(res_y_mod)

    return features_aug, dir_labels_aug, mod_labels_aug


def subset_doubles_spaced_quantiles(
    n_per_class: int, features_aug: np.ndarray, dir_labels_aug: np.ndarray, mod_labels_aug: np.ndarray
):
    """For each class, rank items by their distance to the class mean,
    and take items with ranks 1, K+1, 2K+1.
    The spacing K will be approx (class_size / n_per_class)
    """
    # Find class means
    class_means = {}
    labels_2d = np.stack([dir_labels_aug.argmax(-1), mod_labels_aug.argmax(-1)], axis=-1)
    for d, m in np.unique(labels_2d, axis=0):
        idx = np.where((labels_2d == (d, m)).all(-1))[0]
        class_means[(d, m)] = np.mean(features_aug[idx], axis=0)

    # Subset each class by taking items closest to mean
    res_x, res_y_dir, res_y_mod = [], [], []
    for d, m in np.unique(labels_2d, axis=0):
        class_mean = class_means[(d, m)]
        idx = np.where((labels_2d == (d, m)).all(-1))[0]
        dists = np.linalg.norm(features_aug[idx] - class_mean, axis=-1)
        ranked_distances = np.argsort(dists)
        spacing = int(np.floor(len(idx) / n_per_class))
        # Since we use floor, we step slightly too little.
        # In case this gives us extra items, we also truncate.
        subset_idx = idx[ranked_distances[::spacing][:n_per_class]]
        n_subset = len(subset_idx)
        assert abs(n_subset - n_per_class) <= 1

        res_x.append(features_aug[subset_idx])
        res_y_dir.append(dir_labels_aug[subset_idx])
        res_y_mod.append(mod_labels_aug[subset_idx])

    features_aug = np.concatenate(res_x)
    dir_labels_aug = np.concatenate(res_y_dir)
    mod_labels_aug = np.concatenate(res_y_mod)

    return features_aug, dir_labels_aug, mod_labels_aug


def subset_dir_mod(
    method: str, fraction_doubles_per_class: float, features: np.ndarray, dir_labels: np.ndarray, mod_labels: np.ndarray
):
    # Should have 1-hot vector labels
    assert dir_labels.ndim == 2
    assert mod_labels.ndim == 2

    # check these are all singles
    items_with_dir = dir_labels.argmax(-1) != NO_DIR_IDX
    items_with_mod = mod_labels.argmax(-1) != NO_MOD_IDX
    items_with_both = np.logical_and(items_with_dir, items_with_mod)
    assert np.sum(items_with_both) == 0

    labels_2d = np.stack([dir_labels.argmax(-1), mod_labels.argmax(-1)], axis=-1)

    # Figure out how many items we have per class
    # Then use fraction_doubles_per_class to figure out how many doubles we want
    class_sizes = np.unique(labels_2d, axis=0, return_counts=True)[-1]
    n_per_class = int(np.round(fraction_doubles_per_class * np.mean(class_sizes)))
    n_per_class = min(n_per_class, np.min(class_sizes))
    logger.info(f"Initial class sizes: {class_sizes}, n_per_class: {n_per_class}")

    # For each class, fit a multivariate gaussian and sample the requested number of points
    res_x, res_y_dir, res_y_mod = [], [], []
    for d, m in np.unique(labels_2d, axis=0):
        idx = np.where((labels_2d == (d, m)).all(-1))[0]
        class_mean = np.mean(features[idx], axis=0)

        if method == "subsetInput_uniform":
            subset_idx = np.random.choice(idx, n_per_class, replace=False)
        elif method == "subsetInput_near_mean":
            dists = np.linalg.norm(features[idx] - class_mean, axis=-1)
            ranked_distances = np.argsort(dists)
            subset_idx = idx[ranked_distances[:n_per_class]]
        elif method == "subsetInput_spaced_quantiles":
            dists = np.linalg.norm(features[idx] - class_mean, axis=-1)
            ranked_distances = np.argsort(dists)
            spacing = int(np.floor(len(idx) / n_per_class))
            # Since we use floor, we step slightly too little.
            # In case this gives us extra items, we also truncate.
            subset_idx = idx[ranked_distances[::spacing][:n_per_class]]
            n_subset = len(subset_idx)
            assert abs(n_subset - n_per_class) <= 1

        res_x.append(features[subset_idx])
        res_y_dir.append(dir_labels[subset_idx])
        res_y_mod.append(mod_labels[subset_idx])

    res_x = np.concatenate(res_x)
    res_y_dir = np.concatenate(res_y_dir)
    res_y_mod = np.concatenate(res_y_mod)
    labels_2d = np.stack([res_y_dir.argmax(-1), res_y_mod.argmax(-1)], axis=-1)
    class_sizes = np.unique(labels_2d, axis=0, return_counts=True)[-1]
    logger.info(f"Class sizes after subset: {class_sizes}")

    return res_x, res_y_dir, res_y_mod


def get_augmented_doubles(
    method: str,
    feature_combine_type: str,
    fraction_doubles_per_class: float,
    features: np.ndarray,
    dir_labels: np.ndarray,
    mod_labels: np.ndarray,
):
    if feature_combine_type == "avg":
        aug = AvgPairs(-1)
    elif feature_combine_type == "max":
        aug = ElementwiseMaxPairs(-1)
    else:
        raise ValueError(f"Unknown feature_combine_type: {feature_combine_type}")

    if method == "none":
        logger.info("No synthetic doubles")
        # We create nothing and return early
        features_aug = np.empty((0, *features.shape[1:]))
        dir_labels_aug = np.empty((0, *dir_labels.shape[1:]))
        mod_labels_aug = np.empty((0, *mod_labels.shape[1:]))
        return features_aug, dir_labels_aug, mod_labels_aug

    if method.startswith("subsetInput"):
        # NOTE - here, n_per_class means how many items in each INPUT class
        # Do the subsetting before making combinations
        logger.info("Subset before creating doubles")
        features_subset, dir_labels_subset, mod_labels_subset = subset_dir_mod(
            method, fraction_doubles_per_class, features, dir_labels, mod_labels
        )
        features_aug, dir_labels_aug, mod_labels_aug = aug(features_subset, dir_labels_subset, mod_labels_subset)

        labels_2d = np.stack([dir_labels_aug.argmax(-1), mod_labels_aug.argmax(-1)], axis=-1)
        class_sizes = np.unique(labels_2d, axis=0, return_counts=True)[-1]
        logger.info(f"Final synthetic double class sizes: {class_sizes}")

        return features_aug, dir_labels_aug, mod_labels_aug

    # Other methods create all combinations and THEN subset

    # First, create all augmented items
    logger.info("Subset after creating doubles")
    features_aug, dir_labels_aug, mod_labels_aug = aug(features, dir_labels, mod_labels)
    labels_2d = np.stack([dir_labels_aug.argmax(-1), mod_labels_aug.argmax(-1)], axis=-1)
    class_sizes = np.unique(labels_2d, axis=0, return_counts=True)[-1]
    logger.info(f"Initial synthetic double class sizes: {class_sizes}")

    # check these are all doubles
    items_with_dir = dir_labels_aug.argmax(-1) != NO_DIR_IDX
    items_with_mod = mod_labels_aug.argmax(-1) != NO_MOD_IDX
    items_with_both = np.logical_and(items_with_dir, items_with_mod)
    assert np.sum(items_with_both) == len(features_aug)

    # Figure out how many items we want per class
    n_per_class = int(np.round(fraction_doubles_per_class * np.mean(class_sizes)))
    n_per_class = min(n_per_class, np.min(class_sizes))

    # Then, subset as requested
    if method == "all":
        pass
    elif method == "subset_uniform":
        features_aug, dir_labels_aug, mod_labels_aug = subset_doubles_uniform(
            n_per_class, features_aug, dir_labels_aug, mod_labels_aug
        )
    elif method == "subset_near_mean":
        features_aug, dir_labels_aug, mod_labels_aug = subset_doubles_near_mean(
            n_per_class, features_aug, dir_labels_aug, mod_labels_aug
        )
    elif method == "subset_spaced_quantiles":
        features_aug, dir_labels_aug, mod_labels_aug = subset_doubles_spaced_quantiles(
            n_per_class, features_aug, dir_labels_aug, mod_labels_aug
        )
    else:
        raise ValueError(f"Unknown augmentation method: {method}")

    labels_2d = np.stack([dir_labels_aug.argmax(-1), mod_labels_aug.argmax(-1)], axis=-1)
    class_sizes = np.unique(labels_2d, axis=0, return_counts=True)[-1]
    logger.info(f"Final synthetic double class sizes: {class_sizes}")
    return features_aug, dir_labels_aug, mod_labels_aug


def get_noise_simple(x, relative_std):
    """Add noise to x, where the noise standard deviation is relative_std * x.std()"""
    return np.random.randn(*x.shape) * relative_std * x.std(0)


def balanced_sample_singles(features, dir_labels, mod_labels, n_per_class):
    # Should have 1-hot vector labels
    assert dir_labels.ndim == 2
    assert mod_labels.ndim == 2

    # check these are all singles
    items_with_dir = dir_labels.argmax(-1) != NO_DIR_IDX
    items_with_mod = mod_labels.argmax(-1) != NO_MOD_IDX
    items_with_both = np.logical_and(items_with_dir, items_with_mod)
    assert np.sum(items_with_both) == 0

    labels_2d = np.stack([dir_labels.argmax(-1), mod_labels.argmax(-1)], axis=-1)

    res_x, res_y_dir, res_y_mod = [], [], []
    for d, m in np.unique(labels_2d, axis=0):
        idx = np.where((labels_2d == (d, m)).all(-1))[0]

        n_needed = n_per_class
        selected_idx = []
        while True:
            if n_needed >= len(idx):
                # Take all items in this class 1 more time
                selected_idx.append(idx)
                n_needed -= len(idx)
            else:
                # Take the remaining items randomly
                selected_idx.append(np.random.choice(idx, n_needed, replace=False))
                break
        selected_idx = np.concatenate(selected_idx)

        res_x.append(features[selected_idx])
        res_y_dir.append(dir_labels[selected_idx])
        res_y_mod.append(mod_labels[selected_idx])

    return np.concatenate(res_x), np.concatenate(res_y_dir), np.concatenate(res_y_mod)


def sample_singles_gmm(features, dir_labels, mod_labels, n_per_class, n_components):
    """Fit a GMM to each class, then sample as requested"""
    assert dir_labels.ndim == 2
    assert mod_labels.ndim == 2

    # check these are all singles
    items_with_dir = dir_labels.argmax(-1) != NO_DIR_IDX
    items_with_mod = mod_labels.argmax(-1) != NO_MOD_IDX
    items_with_both = np.logical_and(items_with_dir, items_with_mod)
    assert np.sum(items_with_both) == 0

    labels_2d = np.stack([dir_labels.argmax(-1), mod_labels.argmax(-1)], axis=-1)

    # For each class, fit a multivariate gaussian and sample the requested number of points
    res_x, res_y_dir, res_y_mod = [], [], []
    for d, m in np.unique(labels_2d, axis=0):
        # NOTE - d and m are now integer values. We need to convert them to 1-hot vectors for the output
        d_onehot = np.zeros(dir_labels.shape[1])
        d_onehot[d] = 1
        m_onehot = np.zeros(mod_labels.shape[1])
        m_onehot[m] = 1
        idx = np.where((labels_2d == (d, m)).all(-1))[0]

        gmm = GaussianMixture(n_components=n_components)
        gmm.fit(features[idx])

        res_x.append(gmm.sample(n_per_class)[0])
        res_y_dir.append(np.tile(d_onehot, (n_per_class, 1)))
        res_y_mod.append(np.tile(m_onehot, (n_per_class, 1)))

    return np.concatenate(res_x), np.concatenate(res_y_dir), np.concatenate(res_y_mod)


def sample_singles_kde(features, dir_labels, mod_labels, n_per_class, bandwidth):
    """Fit a GMM to each class, then sample as requested"""
    assert dir_labels.ndim == 2
    assert mod_labels.ndim == 2

    # check these are all singles
    items_with_dir = dir_labels.argmax(-1) != NO_DIR_IDX
    items_with_mod = mod_labels.argmax(-1) != NO_MOD_IDX
    items_with_both = np.logical_and(items_with_dir, items_with_mod)
    assert np.sum(items_with_both) == 0

    labels_2d = np.stack([dir_labels.argmax(-1), mod_labels.argmax(-1)], axis=-1)

    # For each class, fit a multivariate gaussian and sample the requested number of points
    res_x, res_y_dir, res_y_mod = [], [], []
    for d, m in np.unique(labels_2d, axis=0):
        # NOTE - d and m are now integer values. We need to convert them to 1-hot vectors for the output
        d_onehot = np.zeros(dir_labels.shape[1])
        d_onehot[d] = 1
        m_onehot = np.zeros(mod_labels.shape[1])
        m_onehot[m] = 1
        idx = np.where((labels_2d == (d, m)).all(-1))[0]

        kde = KernelDensity(bandwidth=bandwidth)
        kde.fit(features[idx])

        res_x.append(kde.sample(n_per_class))
        res_y_dir.append(np.tile(d_onehot, (n_per_class, 1)))
        res_y_mod.append(np.tile(m_onehot, (n_per_class, 1)))

    return np.concatenate(res_x), np.concatenate(res_y_dir), np.concatenate(res_y_mod)


def get_augmented_singles(
    method: str, n_per_class: int, features: np.ndarray, dir_labels: np.ndarray, mod_labels: np.ndarray
):
    if method == "none":
        logger.info("No augmented singles")
        # Return empties so we can just concatenate and not worry about it
        features_aug = np.empty((0, *features.shape[1:]))
        dir_labels_aug = np.empty((0, *dir_labels.shape[1:]))
        mod_labels_aug = np.empty((0, *mod_labels.shape[1:]))
        return features_aug, dir_labels_aug, mod_labels_aug

    logger.info(f"Augmenting singles with method {method}")
    if method.startswith("add-gaussian"):
        # First, choose a subset of items according to n_per_class
        features, dir_labels_aug, mod_labels_aug = balanced_sample_singles(
            features, dir_labels, mod_labels, n_per_class
        )
        if method == "add-gaussian-0.05":
            factor = 0.05
        elif method == "add-gaussian-0.1":
            factor = 0.1
        elif method == "add-gaussian-0.2":
            factor = 0.2
        elif method == "add-gaussian-0.3":
            factor = 0.3
        elif method == "add-gaussian-0.4":
            factor = 0.4
        elif method == "add-gaussian-0.5":
            factor = 0.5
        elif method == "add-gaussian-0.6":
            factor = 0.6
        else:
            raise ValueError(f"Unknown gaussian factor: {method}")
        features_aug = features + get_noise_simple(features, factor)
    elif method.startswith("fit-gmm"):
        if method == "fit-gmm-1":
            nc = 1
        elif method == "fit-gmm-3":
            nc = 3
        elif method == "fit-gmm-5":
            nc = 5
        elif method == "fit-gmm-10":
            nc = 10
        features_aug, dir_labels_aug, mod_labels_aug = sample_singles_gmm(
            features, dir_labels, mod_labels, n_per_class, n_components=nc
        )
    elif method.startswith("fit-kde"):
        if method == "fit-kde-gaussian-scott":
            bandwidth = "scott"
        if method == "fit-kde-gaussian-silverman":
            bandwidth = "silverman"
        if method == "fit-kde-gaussian-0.01":
            bandwidth = 0.01
        if method == "fit-kde-gaussian-0.1":
            bandwidth = 0.1
        if method == "fit-kde-gaussian-1.0":
            bandwidth = 1.0
        if method == "fit-kde-gaussian-10.0":
            bandwidth = 10.0
        features_aug, dir_labels_aug, mod_labels_aug = sample_singles_kde(
            features, dir_labels, mod_labels, n_per_class, bandwidth=bandwidth
        )

    else:
        raise NotImplementedError()

    labels_2d = np.stack([dir_labels_aug.argmax(-1), mod_labels_aug.argmax(-1)], axis=-1)
    class_sizes = np.unique(labels_2d, axis=0, return_counts=True)[-1]
    logger.info(f"Augmented singles class sizes: {class_sizes}")
    return features_aug, dir_labels_aug, mod_labels_aug


def get_clf(name: str, num_classes: int):
    if name == "mlp":
        return make_pipeline(
            RobustScaler(), MLPClassifier(hidden_layer_sizes=[100, 100, 100], early_stopping=True, max_iter=200)
        )
    elif name == "logr":
        return make_pipeline(RobustScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, n_jobs=-1))
    elif name == "svc":
        return make_pipeline(RobustScaler(), SVC(class_weight="balanced", probability=True))
    elif name == "rf":
        return make_pipeline(RobustScaler(), RandomForestClassifier(class_weight="balanced", n_jobs=-1))
    elif name == "knn":
        return make_pipeline(RobustScaler(), KNeighborsClassifier())
    elif name == "lda":
        return make_pipeline(RobustScaler(), LinearDiscriminantAnalysis())
    elif name == "gbc":
        return make_pipeline(RobustScaler(), GradientBoostingClassifier())

    else:
        raise ValueError(f"Unknown model name: {name}")


def balance_classes(train_features, train_dir_labels, train_mod_labels):
    # Subsample the "Rest" class since it will be overrepresented
    assert train_dir_labels.ndim == 2
    assert train_mod_labels.ndim == 2
    labels_2d = np.stack([train_dir_labels.argmax(-1), train_mod_labels.argmax(-1)], axis=-1)

    class_sizes = np.unique(labels_2d, axis=0, return_counts=True)[-1]
    logger.info(f"Before pruning 'Rest' items, class sizes: {class_sizes}")

    rest_idx = np.where((labels_2d == [NO_DIR_IDX, NO_MOD_IDX]).all(-1))[0]
    active_idx = np.where((labels_2d != [NO_DIR_IDX, NO_MOD_IDX]).any(-1))[0]

    active_counts = np.unique(labels_2d[active_idx], axis=0, return_counts=True)[-1]
    avg_n_active = int(np.mean(active_counts))

    subset_rest_idx = np.random.choice(rest_idx, avg_n_active, replace=False)

    res_x = np.concatenate((train_features[active_idx], train_features[subset_rest_idx]))
    res_y_dir = np.concatenate((train_dir_labels[active_idx], train_dir_labels[subset_rest_idx]))
    res_y_mod = np.concatenate((train_mod_labels[active_idx], train_mod_labels[subset_rest_idx]))

    res_labels_2d = np.stack([res_y_dir.argmax(-1), res_y_mod.argmax(-1)], axis=-1)
    res_class_sizes = np.unique(res_labels_2d, axis=0, return_counts=True)[-1]
    logger.info(f"After pruning 'Rest' items, class sizes: {res_class_sizes}")
    return res_x, res_y_dir, res_y_mod


def remove_double_gestures(train_features, train_dir_labels, train_mod_labels):
    labels_2d = np.stack([train_dir_labels.argmax(-1), train_mod_labels.argmax(-1)], axis=-1)
    class_sizes = np.unique(labels_2d, axis=0, return_counts=True)[-1]
    logger.info(f"Before removing double gestures, class sizes: {class_sizes}")
    items_with_dir = train_dir_labels.argmax(-1) != NO_DIR_IDX
    items_with_mod = train_mod_labels.argmax(-1) != NO_MOD_IDX
    # Remove items with both direction and modifier
    singles_idx = ~np.logical_and(items_with_dir, items_with_mod)

    res_features = train_features[singles_idx]
    res_dir_labels = train_dir_labels[singles_idx]
    res_mod_labels = train_mod_labels[singles_idx]
    res_labels_2d = np.stack([res_dir_labels.argmax(-1), res_mod_labels.argmax(-1)], axis=-1)
    res_class_sizes = np.unique(res_labels_2d, axis=0, return_counts=True)[-1]
    logger.info(f"After removing double gestures, class sizes: {res_class_sizes}")
    return res_features, res_dir_labels, res_mod_labels


@logger.catch(onerror=lambda _: sys.exit(1))
def run_training(
    subject: str,
    parallel_model_type: str,
    clf_name: str,
    doubles_method: str,
    fraction_doubles_per_class: float,
    singles_method: str,
    rel_fraction_singles_per_class: float,
    include_doubles_in_train: bool,
    feature_combine_type: str,
):
    # We don't want to modify code in the gest module itself.
    # Thus, we'll do augmentation manually here, and tell the model not to do
    # any further augmentation.

    # Load train data
    data_dict = load_data_dict()
    try:
        data = data_dict[subject]
    except KeyError:
        raise ValueError(f"Unknown subject: {subject}")

    train_features = data["Calibration_features"]
    train_dir_labels = data["Calibration_dir_labels"]
    train_mod_labels = data["Calibration_mod_labels"]

    # First, reduce amount of "Rest" items in train set
    train_features, train_dir_labels, train_mod_labels = balance_classes(
        train_features, train_dir_labels, train_mod_labels
    )
    # Remove any double gestures that occured due to bad participant behavior
    train_features, train_dir_labels, train_mod_labels = remove_double_gestures(
        train_features, train_dir_labels, train_mod_labels
    )

    # NOTE - we use HoldPulse1_NoFeedback and SimultaneousPulse1_NoFeedback for train set in the "upper bound"
    # otherwise, these blocks are not used

    # Load test data
    if include_doubles_in_train:
        # We use blocks 1 and 2 of the "NoFeedBack" portion of experiment
        # Double check that we're not using augmentation
        assert doubles_method == "none"
        assert singles_method == "none"

        # Add real combos to train set
        train_features = np.concatenate(
            [
                train_features,
                data["HoldPulse1_NoFeedBack_features"],
                data["SimultaneousPulse1_NoFeedBack_features"],
                data["HoldPulse2_NoFeedBack_features"],
                data["SimultaneousPulse2_NoFeedBack_features"],
            ]
        )
        train_dir_labels = np.concatenate(
            [
                train_dir_labels,
                data["HoldPulse1_NoFeedBack_dir_labels"],
                data["SimultaneousPulse1_NoFeedBack_dir_labels"],
                data["HoldPulse2_NoFeedBack_dir_labels"],
                data["SimultaneousPulse2_NoFeedBack_dir_labels"],
            ]
        )
        train_mod_labels = np.concatenate(
            [
                train_mod_labels,
                data["HoldPulse1_NoFeedBack_mod_labels"],
                data["SimultaneousPulse1_NoFeedBack_mod_labels"],
                data["HoldPulse2_NoFeedBack_mod_labels"],
                data["SimultaneousPulse2_NoFeedBack_mod_labels"],
            ]
        )

    logger.info(f"Initial train set: {train_features.shape=}, {train_dir_labels.shape=}, {train_mod_labels.shape=}")

    # Don't use "Feedback" blocks for this analysis
    test_blocks = ["HoldPulse3_NoFeedBack", "SimultaneousPulse3_NoFeedBack"]
    test_features = np.concatenate([data[f"{block}_features"] for block in test_blocks])
    test_dir_labels = np.concatenate([data[f"{block}_dir_labels"] for block in test_blocks])
    test_mod_labels = np.concatenate([data[f"{block}_mod_labels"] for block in test_blocks])
    logger.info(f"test set: {test_features.shape=}, {test_dir_labels.shape=}, {test_mod_labels.shape=}")

    # Vary strategy for augmented doubles
    double_features_aug, double_dir_labels_aug, double_mod_labels_aug = get_augmented_doubles(
        doubles_method,
        feature_combine_type,
        fraction_doubles_per_class,
        train_features,
        train_dir_labels,
        train_mod_labels,
    )

    # Make augmented singles
    # Figure out how many doubles per class. Take avg and then apply rel_fraction_singles_per_class to
    # get the number of singles per class
    n_singles_per_class = 0
    if singles_method != "none":
        doubles_labels_2d = np.stack((double_dir_labels_aug.argmax(-1), double_mod_labels_aug.argmax(-1)), axis=-1)
        class_sizes = np.unique(doubles_labels_2d, axis=0, return_counts=True)[-1]
        n_singles_per_class = int(np.round(np.mean(class_sizes) * rel_fraction_singles_per_class))

    single_features_aug, single_dir_labels_aug, single_mod_labels_aug = get_augmented_singles(
        singles_method, n_singles_per_class, train_features, train_dir_labels, train_mod_labels
    )

    # Merge all train data
    train_features = np.concatenate([train_features, double_features_aug, single_features_aug])
    train_dir_labels = np.concatenate([train_dir_labels, double_dir_labels_aug, single_dir_labels_aug])
    train_mod_labels = np.concatenate([train_mod_labels, double_mod_labels_aug, single_mod_labels_aug])

    logger.info(f"Augmented train set: {train_features.shape=}, {train_dir_labels.shape=}, {train_mod_labels.shape=}")
    # Create model
    if parallel_model_type == "ParallelA":
        model = ParallelA(
            get_clf(clf_name, num_classes=5),
            get_clf(clf_name, num_classes=3),
            use_augmentation=False,
            include_rest_data_for_clf=True,
        )
    elif parallel_model_type == "ParallelB":
        model = ParallelB(
            dir_clf=get_clf(clf_name, num_classes=4),
            mod_clf=get_clf(clf_name, num_classes=2),
            has_dir_clf=get_clf(clf_name, num_classes=2),
            has_mod_clf=get_clf(clf_name, num_classes=2),
            use_augmentation=False,
            # include_rest_data_for_clf=True,  # NOTE - always using true, flag is not in model
        )
    elif parallel_model_type == "SerialControl":
        model = get_clf(clf_name, num_classes=15)
    else:
        raise ValueError(f"Unknown parallel model type: {parallel_model_type}")

    # Train
    logger.info("Train...")
    if parallel_model_type == "SerialControl":
        # Convert labels to integer by making 2-digit numbers,
        # where the 10s place is the dir label and the 1s place is the mod label
        train_labels = train_dir_labels.argmax(-1) * 10 + train_mod_labels.argmax(-1)
        model.fit(train_features, train_labels)
    else:
        model.fit(train_features, train_dir_labels, train_mod_labels)

    # Evaluate
    logger.info("Evaluate")
    if parallel_model_type == "SerialControl":
        combined_preds = model.predict(test_features)
        dir_preds = combined_preds // 10
        mod_preds = combined_preds % 10
    else:
        dir_preds, mod_preds = model.predict(test_features)
    preds_2d = np.stack([dir_preds, mod_preds], axis=-1)
    true_labels_2d = np.stack([test_dir_labels.argmax(-1), test_mod_labels.argmax(-1)], axis=-1)
    return confusion_matrix(true_labels_2d, preds_2d)


if __name__ == "__main__":
    import argparse

    logger.remove()
    logger.add(sys.stdout, level="INFO", colorize=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--parallel_model_type", choices=["ParallelA", "ParallelB", "SerialControl"], required=True)
    clf_names = ["mlp", "rf", "logr"]
    parser.add_argument("--clf_name", type=str, choices=clf_names, required=True)
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
    parser.add_argument("--fraction_doubles_per_class", type=float, required=True)
    singles_methods = [
        "none",
        "add-gaussian-0.3",
        "add-gaussian-0.4",
        "add-gaussian-0.5",
        "fit-gmm-1",
        "fit-gmm-5",
        "fit-gmm-10",
        "fit-kde-gaussian-silverman",
        "fit-kde-gaussian-0.01",
        "fit-kde-gaussian-0.1",
        "fit-kde-gaussian-1.0",
    ]
    parser.add_argument("--singles_method", type=str, choices=singles_methods, required=True)
    parser.add_argument("--rel_fraction_singles_per_class", type=float, required=True)
    parser.add_argument("--include_doubles_in_train", type=str2bool, required=True)
    parser.add_argument("--feature_combine_type", type=str, choices=["avg", "max"], required=True)
    args = parser.parse_args()

    if args.include_doubles_in_train:
        # When we do the "upper bound" model - we should not try to do any augmentation
        if args.doubles_method != "none" or args.singles_method != "none":
            raise ValueError("When including doubles in train, don't use augmentation")

    # We either use no augmentation, doubles aug, or doubles + singles.
    # We never use singles alone.
    if args.doubles_method == "none" and args.singles_method != "none":
        raise ValueError("Can't use singles augmentation without doubles augmentation")

    if not 0 <= args.fraction_doubles_per_class <= 1:
        raise ValueError(f"Invalid fraction_doubles_per_class: {args.fraction_doubles_per_class}")
    if not 0 <= args.rel_fraction_singles_per_class <= 1:
        raise ValueError(f"Invalid rel_fraction_singles_per_class: {args.rel_fraction_singles_per_class}")
    logger.info(args)

    # NOTE - seed affects how data is subsampled to balance classes
    np.random.seed(args.seed)

    cm = run_training(
        subject=args.subject,
        parallel_model_type=args.parallel_model_type,
        clf_name=args.clf_name,
        doubles_method=args.doubles_method,
        fraction_doubles_per_class=args.fraction_doubles_per_class,
        singles_method=args.singles_method,
        rel_fraction_singles_per_class=args.rel_fraction_singles_per_class,
        include_doubles_in_train=args.include_doubles_in_train,
        feature_combine_type=args.feature_combine_type,
    )

    output_dir = RESULTS_DIR / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Make a title suitable for filename
    title = get_name(
        subject=args.subject,
        seed=args.seed,
        parallel_model_type=args.parallel_model_type,
        clf_name=args.clf_name,
        doubles_method=args.doubles_method,
        fraction_doubles_per_class=args.fraction_doubles_per_class,
        singles_method=args.singles_method,
        rel_fraction_singles_per_class=args.rel_fraction_singles_per_class,
        include_doubles_in_train=args.include_doubles_in_train,
        feature_combine_type=args.feature_combine_type,
    )
    np.save(output_dir / f"{title}.npy", cm)
    fig = plot_confusion_matrix(cm)
    fig.write_image(output_dir / f"{title}.png", width=1000, height=1000, scale=2)

    logger.info("finished")
