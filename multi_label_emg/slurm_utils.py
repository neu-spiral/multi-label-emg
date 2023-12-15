import subprocess
from pathlib import Path
from textwrap import dedent

from multi_label_emg.utils import PROJECT_ROOT

ON_SLURM_CLUSTER = subprocess.run("which sbatch", shell=True, check=False, capture_output=True).returncode == 0
if ON_SLURM_CLUSTER:
    slurm_logs_dir = PROJECT_ROOT.parent / "slurm_logs"
    slurm_logs_dir.mkdir(exist_ok=True, parents=True)


def _run_one_slurm(inner_cmd: str, global_job_count: int, slurm_logs_dir: Path, dry_run: bool):
    partitions = ["short", "express"]
    partition = partitions[global_job_count % len(partitions)]
    wrapped_cmd = dedent(
        f"""
        sbatch
         --nodes=1
         --ntasks=1
         --cpus-per-task=8
         --time=1:00:00
         --job-name=gest
         --partition={partition}
         --mem=16Gb
         --output={slurm_logs_dir / "slurm-%j.out"}
         --open-mode=truncate
         --wrap=" {inner_cmd} "
        """
    ).replace("\n", " ")
    print(wrapped_cmd)
    if not dry_run:
        subprocess.run(wrapped_cmd, shell=True, check=True)


def _run_one_local(inner_cmd: str, global_job_count: int, dry_run):
    # use sem to limit the number of concurrent jobs
    # ALLOWED_CUDA_DEVICE_IDS = [0, 1]
    ALLOWED_CUDA_DEVICE_IDS = [0]
    cuda_device_id = ALLOWED_CUDA_DEVICE_IDS[global_job_count % len(ALLOWED_CUDA_DEVICE_IDS)]

    wrapped_cmd = dedent(
        f"""
        CUDA_VISIBLE_DEVICES={cuda_device_id}
         sem --id {cuda_device_id} --jobs 1
         {inner_cmd}
         >/dev/null 2>&1 &
        """
    ).replace("\n", " ")
    print(wrapped_cmd)
    if not dry_run:
        subprocess.run(wrapped_cmd, shell=True, check=True)


def run_one(job: str, running_job_count: int, dry_run: bool):
    if ON_SLURM_CLUSTER:
        _run_one_slurm(job, running_job_count, slurm_logs_dir, dry_run)
    else:
        _run_one_local(job, running_job_count, dry_run)
