import pickle
from abc import ABC, abstractmethod
from itertools import product
from pathlib import Path
from typing import Any, Tuple

import numpy as np
from sklearn.base import ClassifierMixin

from multi_label_emg.utils import NO_DIR_IDX, NO_MOD_IDX


class InsufficientDataError(Exception):
    ...


def split_dir_mod(x: np.ndarray, y_dir: np.ndarray, y_mod: np.ndarray):
    """Given single and double data:
    - isolate the singles
    - split into direction and modifier gestures
    - split these by class

    Returns:
        x_dir List[Union[torch.tensor, np.ndarray]]: each item shape (n_samples, n_features)
        x_mod List[Union[torch.tensor, np.ndarray]]: each item shape (n_samples, n_features)
        y_dir List[Union[torch.tensor, np.ndarray]]: each item shape (n_samples) - integer labels
        y_mod List[Union[torch.tensor, np.ndarray]]: each item shape (n_samples) - integer labels
    """
    assert y_dir.ndim == 2
    assert y_mod.ndim == 2
    # Ignore all combo gestures
    singles_idx = np.where(np.logical_xor((y_dir.argmax(-1) == NO_DIR_IDX), (y_mod.argmax(-1) == NO_MOD_IDX)))
    x_single, y_dir_single, y_mod_single = x[singles_idx], y_dir[singles_idx], y_mod[singles_idx]

    # Separate into direction gestures and modifier gestures
    y_dir_all = y_dir_single[y_dir_single.argmax(-1) != NO_DIR_IDX]
    x_dir_all = x_single[y_dir_single.argmax(-1) != NO_DIR_IDX]

    y_mod_all = y_mod_single[y_mod_single.argmax(-1) != NO_MOD_IDX]
    x_mod_all = x_single[y_mod_single.argmax(-1) != NO_MOD_IDX]

    # Sanity check that we have single gestures
    if len(y_dir_all) == 0:
        raise InsufficientDataError()
    if len(y_mod_all) == 0:
        raise InsufficientDataError()

    return x_dir_all, x_mod_all, y_dir_all, y_mod_all


class AvgPairs:
    """Create fake doubles by averaging pairs of singles. New items have hard labels including both classes"""

    def __init__(self, n_per_class: int):
        self.n_per_class = n_per_class

    def __call__(self, x: np.ndarray, y_dir: np.ndarray, y_mod: np.ndarray):
        """
        Args:
            x_single: (n_samples_in, n_features) - data/features from single gestures
            y_dir_single: (n_samples_in, DIR_PROBS_SHAPE) - one-hot labels of direction gestures
            y_mod_single: (n_samples_in, MOD_PROBS_SHAPE) - one-hot labels of modifier gestures

        Returns:
            x_prime: (n_samples_aug, n_features) - augmented data
            y_prime_dir: (n_samples_aug, len(DIRECTION_GESTURES)) - augmented labels
            y_prime_mod: (n_samples_aug, len(MODIFIER_GESTURES)) - augmented labels
        """
        x_dir, x_mod, y_dir, y_mod = split_dir_mod(x, y_dir, y_mod)
        x_aug, y_dir_aug, y_mod_aug = [], [], []
        for (x1, y1), (x2, y2) in product(zip(x_dir, y_dir), zip(x_mod, y_mod)):
            x_aug.append((x1 + x2) / 2)
            y_dir_aug.append(y1)
            y_mod_aug.append(y2)
        x_aug = np.stack(x_aug)
        y_dir_aug = np.stack(y_dir_aug)
        y_mod_aug = np.stack(y_mod_aug)

        if self.n_per_class > 0:
            # For each combination class, truncate to self.n_per_class
            res_x, res_y_dir, res_y_mod = [], [], []
            for d in np.unique(y_dir_aug, axis=0):
                for m in np.unique(y_mod_aug, axis=0):
                    idx = np.where(np.logical_and((y_dir_aug == d).all(-1), (y_mod_aug == m).all(-1)))[0]
                    perm = np.random.permutation(len(idx))
                    res_x.append(x_aug[idx[perm[: self.n_per_class]]])
                    res_y_dir.append(y_dir_aug[idx[perm[: self.n_per_class]]])
                    res_y_mod.append(y_mod_aug[idx[perm[: self.n_per_class]]])

            x_aug = np.concatenate(res_x)
            y_dir_aug = np.concatenate(res_y_dir)
            y_mod_aug = np.concatenate(res_y_mod)

        return x_aug, y_dir_aug, y_mod_aug

    def __repr__(self):
        return f"{type(self).__name__}(n_per_class={self.n_per_class})"


class ElementwiseMaxPairs:
    """Create fake doubles by taking elementwise max of each feature.
    New items have hard labels including both classes"""

    def __init__(self, n_per_class: int):
        self.n_per_class = n_per_class

    def __call__(self, x: np.ndarray, y_dir: np.ndarray, y_mod: np.ndarray):
        """
        Args:
            x_single: (n_samples_in, n_features) - data/features from single gestures
            y_dir_single: (n_samples_in, DIR_PROBS_SHAPE) - one-hot labels of direction gestures
            y_mod_single: (n_samples_in, MOD_PROBS_SHAPE) - one-hot labels of modifier gestures

        Returns:
            x_prime: (n_samples_aug, n_features) - augmented data
            y_prime_dir: (n_samples_aug, len(DIRECTION_GESTURES)) - augmented labels
            y_prime_mod: (n_samples_aug, len(MODIFIER_GESTURES)) - augmented labels
        """
        x_dir, x_mod, y_dir, y_mod = split_dir_mod(x, y_dir, y_mod)
        x_aug, y_dir_aug, y_mod_aug = [], [], []
        for (x1, y1), (x2, y2) in product(zip(x_dir, y_dir), zip(x_mod, y_mod)):
            x_aug.append(np.maximum(x1, x2))
            y_dir_aug.append(y1)
            y_mod_aug.append(y2)
        x_aug = np.stack(x_aug)
        y_dir_aug = np.stack(y_dir_aug)
        y_mod_aug = np.stack(y_mod_aug)

        if self.n_per_class > 0:
            # For each combination class, truncate to self.n_per_class
            res_x, res_y_dir, res_y_mod = [], [], []
            for d in np.unique(y_dir_aug, axis=0):
                for m in np.unique(y_mod_aug, axis=0):
                    idx = np.where(np.logical_and((y_dir_aug == d).all(-1), (y_mod_aug == m).all(-1)))[0]
                    perm = np.random.permutation(len(idx))
                    res_x.append(x_aug[idx[perm[: self.n_per_class]]])
                    res_y_dir.append(y_dir_aug[idx[perm[: self.n_per_class]]])
                    res_y_mod.append(y_mod_aug[idx[perm[: self.n_per_class]]])

            x_aug = np.concatenate(res_x)
            y_dir_aug = np.concatenate(res_y_dir)
            y_mod_aug = np.concatenate(res_y_mod)

        return x_aug, y_dir_aug, y_mod_aug

    def __repr__(self):
        return f"{type(self).__name__}(n_per_class={self.n_per_class})"


class BaseParallelModel(ABC, ClassifierMixin):
    @abstractmethod
    def fit(self, features, dir_labels, mod_labels, data=None) -> None:
        ...

    @abstractmethod
    def predict_proba(self, features) -> Tuple[np.ndarray, np.ndarray]:
        ...

    @abstractmethod
    def predict(self, features) -> Tuple[np.ndarray, np.ndarray]:
        ...

    @abstractmethod
    def save(self, path) -> Path:
        """Save the model to the given path."""

    @classmethod
    @abstractmethod
    def load(cls, path) -> Any:
        """Load the model from the given path."""

    @abstractmethod
    def get_params(self, deep=True) -> Any:
        """Return dictionary of kwargs for __init__ fn"""

    @abstractmethod
    def __repr__(self) -> str:
        ...


class ParallelA(BaseParallelModel):
    DEFAULT_SAVE_NAME = "ParallelA.pkl"

    def __init__(
        self,
        dir_clf,
        mod_clf,
        use_augmentation: bool,
        n_aug_per_class: int = -1,
        include_rest_data_for_clf: bool = False,
    ):
        self.dir_clf = dir_clf
        self.mod_clf = mod_clf
        self.use_augmentation = use_augmentation
        self.n_aug_per_class = n_aug_per_class
        self._n_aug_created = None
        self.include_rest_data_for_clf = include_rest_data_for_clf

    def get_params(self, deep=True):
        return {
            "dir_clf": self.dir_clf,
            "mod_clf": self.mod_clf,
            "use_augmentation": self.use_augmentation,
            "n_aug_per_class": self.n_aug_per_class,
            "include_rest_data_for_clf": self.include_rest_data_for_clf,
        }

    def fit(self, features, y_dir, y_mod):
        if self.use_augmentation:
            aug = AvgPairs(self.n_aug_per_class)
            aug_features, aug_dir_labels, aug_mod_labels = aug(features, y_dir, y_mod)
            features = np.concatenate([features, aug_features])
            y_dir = np.concatenate([y_dir, aug_dir_labels])
            y_mod = np.concatenate([y_mod, aug_mod_labels])
            self._n_aug_created = len(aug_features)

        if y_dir.ndim == 2:
            y_dir = y_dir.argmax(-1)
        if y_mod.ndim == 2:
            y_mod = y_mod.argmax(-1)

        if self.include_rest_data_for_clf:
            # In this case, the label (NoDir, NoMod) could mean "active and doesn't fit our classes" or "resting"
            self.dir_clf.fit(features, y_dir)
            self.mod_clf.fit(features, y_mod)
        else:
            # In this case, the label (NoDir, NoMod) means "active and doesn't fit classes".
            # "Rest" data is out-of-domain
            active_idx = np.logical_or(y_dir != NO_DIR_IDX, y_mod != NO_MOD_IDX)
            active_features = features[active_idx]
            active_y_dir = y_dir[active_idx]
            active_y_mod = y_mod[active_idx]

            self.dir_clf.fit(active_features, active_y_dir)
            self.mod_clf.fit(active_features, active_y_mod)
        return self

    def predict_proba(self, features):
        """Only for gestures"""
        dir_probs = self.dir_clf.predict_proba(features)
        mod_probs = self.mod_clf.predict_proba(features)
        return dir_probs, mod_probs

    def predict(self, features):
        """features.shape == (n_channels, n_samples) or (n_trials, n_channels, n_samples)"""
        dir_probs = self.dir_clf.predict_proba(features)
        mod_probs = self.mod_clf.predict_proba(features)
        return dir_probs.argmax(-1), mod_probs.argmax(-1)

    def save(self, save_dir: Path) -> Path:
        assert save_dir.exists() and save_dir.is_dir()
        file_path = save_dir / self.DEFAULT_SAVE_NAME
        with open(file_path, "wb") as f:
            pickle.dump(self, f)
        return file_path

    @classmethod
    def load(cls, file_path: Path) -> "ParallelA":
        with open(file_path, "rb") as f:
            return pickle.load(f)

    def __repr__(self):
        return (
            f"{type(self).__name__}(dir_clf={self.dir_clf}, "
            f"use_augmentation={self.use_augmentation}, "
            f"n_aug_per_class={self.n_aug_per_class}, "
            + f"mod_clf={self.mod_clf}, "
            + f"include_rest_data_for_clf={self.include_rest_data_for_clf})"
        )


class ParallelB(BaseParallelModel):
    DEFAULT_SAVE_NAME = "ParallelB.pkl"

    def __init__(
        self,
        dir_clf,
        mod_clf,
        has_dir_clf,
        has_mod_clf,
        use_augmentation: bool,
        n_aug_per_class: int = -1,
    ):
        self.has_dir_clf = has_dir_clf
        self.has_mod_clf = has_mod_clf
        self.dir_clf = dir_clf
        self.mod_clf = mod_clf
        self.use_augmentation = use_augmentation
        self.n_aug_per_class = n_aug_per_class
        self._n_aug_created = None

    def get_params(self, deep=True):
        return {
            "dir_clf": self.dir_clf,
            "mod_clf": self.mod_clf,
            "has_dir_clf": self.dir_clf,
            "has_mod_clf": self.mod_clf,
            "use_augmentation": self.use_augmentation,
            "n_aug_per_class": self.n_aug_per_class,
        }

    def fit(self, features, y_dir, y_mod):
        if self.use_augmentation:
            aug = AvgPairs(self.n_aug_per_class)
            aug_features, aug_dir_labels, aug_mod_labels = aug(features, y_dir, y_mod)
            features = np.concatenate([features, aug_features])
            y_dir = np.concatenate([y_dir, aug_dir_labels])
            y_mod = np.concatenate([y_mod, aug_mod_labels])
            self._n_aug_created = len(aug_features)

        if y_dir.ndim == 2:
            y_dir = y_dir.argmax(-1)
        if y_mod.ndim == 2:
            y_mod = y_mod.argmax(-1)
        has_direction = y_dir != NO_DIR_IDX
        has_modifier = y_mod != NO_MOD_IDX
        # Event check
        self.has_dir_clf.fit(features, has_direction.astype(int))
        self.has_mod_clf.fit(features, has_modifier.astype(int))
        # Direction and modifier
        self.dir_clf.fit(features[has_direction], y_dir[has_direction])
        self.mod_clf.fit(features[has_modifier], y_mod[has_modifier])
        return self

    def predict_proba(self, features):
        p_has_direction = self.has_dir_clf.predict_proba(features)
        p_has_modifier = self.has_mod_clf.predict_proba(features)

        p_dir_probs = self.dir_clf.predict_proba(features)
        p_mod_probs = self.mod_clf.predict_proba(features)

        # Check probs
        dir_probs = np.zeros((features.shape[0], 5))
        mod_probs = np.zeros((features.shape[0], 3))
        dir_probs[:, NO_DIR_IDX] = p_has_direction[:, 0]  # p(no_direction | x)
        mod_probs[:, NO_MOD_IDX] = p_has_modifier[:, 0]  # p(no_modifier | x)
        dir_probs[:, :NO_DIR_IDX] = np.multiply(
            p_dir_probs, p_has_direction[:, 1][..., None]
        )  # p(direction | has_direction)
        mod_probs[:, :NO_MOD_IDX] = np.multiply(
            p_mod_probs, p_has_modifier[:, 1][..., None]
        )  # p(modifier | has_modifier)
        assert np.allclose(dir_probs.sum(-1), 1) and np.allclose(mod_probs.sum(-1), 1), "Probabilities should sum to 1"
        # return probs
        """Only for gestures"""
        return dir_probs, mod_probs

    def predict(self, features):
        dir_probs, mod_probs = self.predict_proba(features)
        return dir_probs.argmax(-1), mod_probs.argmax(-1)

    def save(self, save_dir: Path) -> Path:
        assert save_dir.exists() and save_dir.is_dir()
        file_path = save_dir / self.DEFAULT_SAVE_NAME
        with open(file_path, "wb") as f:
            pickle.dump(self, f)
        return file_path

    @classmethod
    def load(cls, file_path: Path) -> "ParallelB":
        with open(file_path, "rb") as f:
            return pickle.load(f)

    def __repr__(self):
        return (
            f"{type(self).__name__}(has_dir_clf={self.has_dir_clf}, "
            f"dir_clf={self.dir_clf}, "
            f"use_augmentation={self.use_augmentation}, "
            f"n_aug_per_class={self.n_aug_per_class}, "
            f"has_mod_clf={self.has_mod_clf}),"
            f"mod_clf={self.mod_clf})"
        )
