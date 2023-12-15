import subprocess
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT.parent / "data"
DATASET_DIR.mkdir(exist_ok=True)
RESULTS_DIR = PROJECT_ROOT.parent / "results"  # For experiment outputs and figures
RESULTS_DIR.mkdir(exist_ok=True)
DIRECTION_GESTURES = ["Up", "Down", "Left", "Right"]
MODIFIER_GESTURES = ["Pinch", "Thumb"]
NO_DIR_IDX = len(DIRECTION_GESTURES)  # When predicting direction, we have an extra class representing "None"
NO_MOD_IDX = len(MODIFIER_GESTURES)
DATA_DTYPE = np.float32


def onehot(y, C: int):
    y_onehot = np.zeros((len(y), C), dtype=DATA_DTYPE)
    y_onehot[np.arange(len(y)), y] = 1
    return y_onehot


def canonical_coords():
    """NOTE - order does not matter: (Up, Pinch) and (Pinch, Up) are both labeled as (Up, Pinch)
    Make a list table so we can convert:
    from integer labels such as (0, 1),
    to an index in confusion matrix and a string label"""
    result_int = []
    result_str = []

    # Add (<DIR>, NoMod) items
    for i, d in enumerate(DIRECTION_GESTURES):
        result_int.append((i, NO_MOD_IDX))
        result_str.append(f"({d}, NoMod)")

    # Add (NoDir, <MOD>) items
    for i, m in enumerate(MODIFIER_GESTURES):
        result_int.append((NO_DIR_IDX, i))
        result_str.append(f"(NoDir, {m})")

    # Add (<DIR>, <MOD>) items
    for i, d in enumerate(DIRECTION_GESTURES):
        for j, m in enumerate(MODIFIER_GESTURES):
            result_int.append((i, j))
            result_str.append(f"({d}, {m})")

    # Add the (NoDir, NoMod) item
    result_int.append((NO_DIR_IDX, NO_MOD_IDX))
    result_str.append("(NoDir, NoMod)")

    return result_int, result_str


def confusion_matrix(y_true_2d, y_pred_2d, normalize_rows=True):
    """
    Number of classes = 4 direction + 2 modifier + 4*2 combinations + (NoDir, NoMod) = 15
    Create a confusion matrix of shape (15, 15), arranged according to the canonical
    coordinates above

    NOTE - result may contain nans - use nanmean later
    """
    coords, coords_str = canonical_coords()

    cm = np.zeros((len(coords), len(coords)), dtype=int)
    for yt, yp in zip(y_true_2d, y_pred_2d):
        cm[coords.index(tuple(yt)), coords.index(tuple(yp))] += 1
    if normalize_rows:
        cm = cm.astype(float)
        with np.errstate(all="ignore"):  # Ignore division by zero for empty rows
            cm /= cm.sum(axis=-1, keepdims=True)
    return cm


def get_git_hash():
    """Get short git hash, with "+" suffix if local files modified"""
    # Need to update index, otherwise can give incorrect results
    subprocess.run(["git", "update-index", "-q", "--refresh"])
    h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip().decode("utf-8")

    # Add '+' suffix if local files are modified
    exitcode, _ = subprocess.getstatusoutput("git diff-index --quiet HEAD")
    if exitcode != 0:
        h += "+"
    return "git" + h


def str2bool(s):
    if s.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif s.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError("Boolean value expected.")
