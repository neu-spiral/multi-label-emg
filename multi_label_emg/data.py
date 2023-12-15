import numpy as np
from scipy import signal
from scipy.integrate import cumulative_trapezoid
from tqdm import tqdm, trange

from multi_label_emg.utils import DATASET_DIR

EMG_PACKET_DURATION = 26
DELSYS_NUM_CHANNELS = 8
DELSYS_STREAM_FREQ_HZ = 1926.0
DATA_DTYPE = np.float32


def load_data_dict():
    """
    Loads features and labels from subject folders into a single dictionary as described below.
    NOTE - preprocessing should be been done first to extract features from raw data (see README).

    data_dict = {
        Subj0: {
            Calibration_features: ...,
            Calibration_dir_labels: ...,
            Calibration_mod_labels: ...,
            Calibration_visual_dir_labels: ...,
            Calibration_visual_mod_labels: ...,
            SimultaneousPulse1_NoFeedback_features: ...,
            ...
        },
        ...
    }
    """

    blocks = ["Calibration"]
    for i in [1, 2, 3]:
        for feedback in ["NoFeedBack", "WithFeedBack"]:
            blocks.append(f"SimultaneousPulse{i}_{feedback}")
            blocks.append(f"HoldPulse{i}_{feedback}")

    results = {}
    for i in trange(11, desc="Load Subjects", leave=True):
        results[f"Subj{i}"] = {}
        for block in tqdm(blocks, leave=False, position=1):
            path = DATASET_DIR / "python" / f"Subj{i}" / block
            # NOTE - features.npy is created during preprocessing script
            results[f"Subj{i}"][f"{block}_features"] = np.load(path / "features.npy")
            results[f"Subj{i}"][f"{block}_dir_labels"] = np.load(path / "joystick_direction_labels.npy")
            results[f"Subj{i}"][f"{block}_mod_labels"] = np.load(path / "joystick_modifier_labels.npy")
            results[f"Subj{i}"][f"{block}_visual_dir_labels"] = np.load(path / "visual_direction_labels.npy")
            results[f"Subj{i}"][f"{block}_visual_mod_labels"] = np.load(path / "visual_modifier_labels.npy")
    return results


def extract_features(data):
    """From data of a single channel, produce 2 features: median frequency and RMS value."""

    def extract_features_one_trial(one_data):
        MedFList = []
        muList = []
        nperseg = min(one_data.shape[-1], 128)
        noverlap = min(nperseg - 1, 80)
        for i in range(len(one_data)):
            f, Pxx_den = signal.welch(one_data[i, :], fs=DELSYS_STREAM_FREQ_HZ, nperseg=nperseg, noverlap=noverlap)
            Pxx_den_norm = Pxx_den / Pxx_den.max()
            f_norm = f / f.max()
            area_freq = cumulative_trapezoid(Pxx_den_norm, f_norm, initial=0)

            total_power = area_freq[-1]
            half_power_idx = np.where(area_freq >= total_power / 2)[0]
            if len(half_power_idx) > 0:
                MedF = f_norm[half_power_idx[0]]
            else:
                MedF = 0
            # MedF = f_norm[np.where(area_freq >= total_power / 2)[0][0]]
            MedFList.append(MedF)
            mu = np.sqrt(np.mean(np.square(one_data[i, :])))
            muList.append(mu)
        features = np.array(muList + MedFList, dtype=DATA_DTYPE)
        return features

    if data.ndim == 2:  # (channels, time)
        return extract_features_one_trial(data)
    elif data.ndim == 3:  # (trials, channels, time)
        return np.array([extract_features_one_trial(one_data) for one_data in data])
