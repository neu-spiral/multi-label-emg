from pathlib import Path

from setuptools import find_packages, setup

requirements = Path("requirements.txt").read_text().splitlines()

setup(
    name="multi_label_emg",
    version="0.1.0",
    # python_requires=">=3.X",
    install_requires=requirements,
    packages=find_packages(),
)
