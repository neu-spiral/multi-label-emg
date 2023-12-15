"""Extract features from raw data and save to files."""

import numpy as np
from tqdm import tqdm, trange

from multi_label_emg.data import extract_features
from multi_label_emg.utils import DATASET_DIR


def main():
    blocks = ["Calibration"]
    for i in [1, 2, 3]:
        for feedback in ["NoFeedback", "WithFeedback"]:
            blocks.append(f"SimultaneousPulse{i}_{feedback}")
            blocks.append(f"HoldPulse{i}_{feedback}")

    for i in trange(11, desc="Subjects", leave=True):  # Subjcts
        for block in tqdm(blocks, desc="Blocks", leave=False, position=1):  # Experimental blocks
            path = DATASET_DIR / "python" / f"Subj{i}" / block
            data = np.load(path / "data.npy")
            features = extract_features(data)
            np.save(path / "features.npy", features)


if __name__ == "__main__":
    main()
