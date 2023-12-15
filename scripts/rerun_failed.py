from pathlib import Path

from multi_label_emg.slurm_utils import run_one
from multi_label_emg.utils import PROJECT_ROOT
from scripts.run_experiment_1 import Setting

DRY_RUN = True

script = PROJECT_ROOT / "train.py"
python = PROJECT_ROOT.parent / "venv" / "bin" / "python"
assert script.exists()
assert python.exists()


def parse_one_run(s: str) -> Setting:
    name = Path(s).stem
    parts = name.split("__")

    incl_doubles = True if parts[8].split("=")[1] == "True" else False
    return Setting(
        subject=parts[0].split("=")[1],
        seed=int(parts[1].split("=")[1]),
        parallel_model_type=parts[2].split("=")[1],
        clf_name=parts[3].split("=")[1],
        doubles_method=parts[4].split("=")[1],
        fraction_doubles_per_class=float(parts[5].split("=")[1]),
        singles_method=parts[6].split("=")[1],
        rel_fraction_singles_per_class=float(parts[7].split("=")[1]),
        include_doubles_in_train=incl_doubles,
    )


if __name__ == "__main__":
    from argparse import ArgumentParser, FileType

    parser = ArgumentParser()
    parser.add_argument("--missing_file", type=FileType("r"), required=True)
    args = parser.parse_args()
    failed_settings_str = args.missing_file.read().splitlines()

    # Parse each one into a "SettingAndSubj" object
    failed_settings = [parse_one_run(s) for s in failed_settings_str]
    if DRY_RUN:
        print("#" * 80)
        print("DRY RUN")

    running_job_count = 0
    for setting in failed_settings:
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
