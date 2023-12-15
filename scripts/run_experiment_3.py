"""
Experiment 3:
Using previous best models and method for synthetic doubles,
Vary methods of generating augmented singles and how many to use.
"""
import itertools

import numpy as np
from run_experiment_1 import Setting

from multi_label_emg.slurm_utils import run_one
from multi_label_emg.utils import PROJECT_ROOT

DRY_RUN = True

script = PROJECT_ROOT / "train.py"
python = PROJECT_ROOT.parent / "venv" / "bin" / "python"
assert script.exists()
assert python.exists()

subjects = [f"Subj{i}" for i in range(11)]
parallel_model_type = "ParallelA"
clf = "mlp"

singles_methods = [
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
settings = []
for doubles_method, doubles_frac in [
    ("all", 1.0),
    ("subset_uniform", 0.1),
]:
    for subj, seed, singles_method, singles_rel_frac in itertools.product(
        subjects,
        np.arange(3),
        singles_methods,
        [0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
    ):
        settings.append(
            Setting(
                subject=subj,
                seed=seed,
                parallel_model_type=parallel_model_type,
                clf_name=clf,
                doubles_method=doubles_method,
                fraction_doubles_per_class=doubles_frac,
                singles_method=singles_method,
                rel_fraction_singles_per_class=singles_rel_frac,
                include_doubles_in_train=False,
                feature_combine_type="avg",
            )
        )


if __name__ == "__main__":
    if DRY_RUN:
        print("#" * 80)
        print("DRY RUN")

    running_job_count = 0
    for setting in settings:
        job = f"{python} {script} "
        job += f"--subject {setting.subject} "
        job += f"--seed {setting.seed} "
        job += f"--parallel_model_type {setting.parallel_model_type} "
        job += f"--clf_name {setting.clf_name} "
        job += f"--doubles_method {setting.doubles_method} "
        job += f"--fraction_doubles_per_class {setting.fraction_doubles_per_class} "
        job += f"--singles_method {setting.singles_method} "
        job += f"--rel_fraction_singles_per_class {setting.rel_fraction_singles_per_class} "
        job += f"--include_doubles_in_train {setting.include_doubles_in_train} "
        run_one(job, running_job_count, dry_run=DRY_RUN)
        running_job_count += 1

    print(f"Total jobs: {running_job_count}")
