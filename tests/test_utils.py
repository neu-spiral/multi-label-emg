from multi_label_emg.data import load_data_dict


def test_load_data_dict():
    data = load_data_dict()

    expected_keys = []
    blocks = ["Calibration"]
    for i in [1, 2, 3]:
        for feedback in ["NoFeedBack", "WithFeedBack"]:
            blocks.append(f"SimultaneousPulse{i}_{feedback}")
            blocks.append(f"HoldPulse{i}_{feedback}")

    for b in blocks:
        expected_keys.append(f"{b}_features")
        expected_keys.append(f"{b}_dir_labels")
        expected_keys.append(f"{b}_mod_labels")
        expected_keys.append(f"{b}_visual_dir_labels")
        expected_keys.append(f"{b}_visual_mod_labels")

    expected_n_items_prefix = {
        "Calibration": 8960,
        "SimultaneousPulse": 1920,
        "HoldPulse": 6720,
    }

    expected_shapes_suffix = {
        "data": (8, 494),
        "features": (16,),
        "dir_labels": (5,),
        "mod_labels": (3,),
        "visual_dir_labels": (5,),
        "visual_mod_labels": (3,),
    }

    def get_by_prefix(d, query):
        for prefix, v in d.items():
            if query.startswith(prefix):
                return v
        raise ValueError(f"Could not find prefix: {query=} {prefix=}")

    def get_by_suffix(d, query):
        for suffix, v in d.items():
            if query.endswith(suffix):
                return v
        raise ValueError(f"Could not find suffix: {query=} {suffix=}")

    expected_shapes = {}
    expected_shapes["Calibration_features"] = (8960, 8, 494)
    expected_shapes["Calibration_dir_labels"] = (8960, 5)

    # Should have all 11 subjects
    for i in range(11):
        subj_data = data[f"Subj{i}"]

        # Should have all blocks
        assert set(subj_data.keys()) == set(expected_keys)

        for k, v in subj_data.items():
            # Should have the correct number of items
            expected_n = get_by_prefix(expected_n_items_prefix, k)
            assert v.shape[0] == expected_n

            # Should have the correct shape
            expected_shape = get_by_suffix(expected_shapes_suffix, k)
            assert v.shape[1:] == expected_shape
