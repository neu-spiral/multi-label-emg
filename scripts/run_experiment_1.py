"""
Experiment 1:
Vary parallel model type and classifier
Use "all" synthetic doubles and no augmented singles.
"""
import itertools
from dataclasses import dataclass

import numpy as np

from multi_label_emg.slurm_utils import run_one
from multi_label_emg.utils import PROJECT_ROOT

DRY_RUN = True

script = PROJECT_ROOT / "train.py"
python = PROJECT_ROOT.parent / "venv" / "bin" / "python"
assert script.exists()
assert python.exists()


@dataclass
class Setting:
    subject: str
    seed: int
    parallel_model_type: str
    clf_name: str
    doubles_method: str
    fraction_doubles_per_class: float
    singles_method: str
    rel_fraction_singles_per_class: float
    include_doubles_in_train: bool
    feature_combine_type: str


subjects = [f"Subj{i}" for i in range(11)]
settings = []
for subj, seed, parallel_model_type, clf, feat_type in itertools.product(
    subjects,
    np.arange(3),
    ["ParallelA", "ParallelB"],
    ["mlp", "rf", "logr"],
    ["avg"],
):
    # actual model
    settings.append(
        Setting(
            subject=subj,
            seed=seed,
            parallel_model_type=parallel_model_type,
            clf_name=clf,
            doubles_method="all",
            fraction_doubles_per_class=1.0,
            singles_method="none",
            rel_fraction_singles_per_class=1.0,
            include_doubles_in_train=False,
            feature_combine_type=feat_type,
        )
    )

    # "lower bound" baseline
    if parallel_model_type != "SerialControl":
        # Serial model needs to see all classes at train time.
        # To get a "lower bound" estimate for it, we will use the "upper bound"
        # and discard the predictions for doubles classes
        settings.append(
            Setting(
                subject=subj,
                seed=seed,
                parallel_model_type=parallel_model_type,
                clf_name=clf,
                doubles_method="none",
                fraction_doubles_per_class=1.0,
                singles_method="none",
                rel_fraction_singles_per_class=1.0,
                include_doubles_in_train=False,
                feature_combine_type=feat_type,
            )
        )

    # "upper bound" baseline
    settings.append(
        Setting(
            subject=subj,
            seed=seed,
            parallel_model_type=parallel_model_type,
            clf_name=clf,
            doubles_method="none",
            fraction_doubles_per_class=1.0,
            singles_method="none",
            rel_fraction_singles_per_class=1.0,
            include_doubles_in_train=True,
            feature_combine_type=feat_type,
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
        job += f"--feature_combine_type {setting.feature_combine_type} "
        run_one(job, running_job_count, dry_run=DRY_RUN)
        running_job_count += 1

    print(f"Total jobs: {running_job_count}")
