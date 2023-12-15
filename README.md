Code for "**A Multi-label Classification Approach to Increase Expressivity of EMG-based Gesture Recognition**" by Niklas Smedemark-Margulies, Yunus Bicer, Elifnur Sunger, Stephanie Naufel, Tales Imbiriba, Eugene Tunik, Deniz Erdoğmuş, and Mathew Yarossi

# Setup and Usage

Use `make` to create python environment, install dependencies, and install `pre-commit` hooks.

To reproduce our experiments:
1. Download the dataset from Zenodo using `bash scripts/fetch_dataset.sh`
2. Preprocess data to extract features using `python scripts/preprocess_data.py`
- To check that all data have been downloaded and preprocessed successfully, use `pytest` to run `test_load_data_dict`

3. Run experiments. Note:
- `DRY_RUN=True` at the beginning of each script causes jobs to be printed, but not executed; change it to `DRY_RUN=False` to actually launch.
- Each script is a large grid of jobs, and all results are simple stored flat in one folder.
- To run fewer jobs at a time, truncate the list of jobs in the script.
```shell
python scripts/run_experiment_1.py
python scripts/run_experiment_2.py
python scripts/run_experiment_3.py
```
4. Create plots. Note:
- The list of expected results is found from each experiment launch script. If some results are missing (e.g. due to runs that timed-out or failed), they will be counted and printed to a text file.
- Failed runs can be re-launched using `python scripts/rerun_failed path/to/missing.txt`, by providing the path to this text file of missing runs.
```shell
python scripts/process_experiment_1.py
python scripts/process_experiment_2.py
python scripts/process_experiment_3.py
```

# PDF

To read our paper, see: https://arxiv.org/pdf/2309.12217.pdf

# Dataset

To use our dataset, see: https://zenodo.org/records/10358039

# Citation

If you use this code or dataset, please use one of the citations below.

Article citation:
```bibtex
@article{smedemark2023multi,
    title={A Multi-label Classification Approach to Increase Expressivity of EMG-based Gesture Recognition},
    author={
        Smedemark-Margulies, Niklas and 
        Bicer, Yunus and
        Sunger, Elifnur and
        Naufel, Stephanie and
        Imbiriba, Tales and
        Tunik, Eugene and
        Erdo{\u{g}}mu{\c{s}}, Deniz and
        Yarossi, Mathew
    },
    journal={arXiv preprint arXiv:2309.12217},
    year={2023},
    month={09},
    day={13},
    url={https://arxiv.org/abs/2309.12217},
}
```

Dataset citation:
```bibtex
@dataset{smedemarkmargulies_2023_10358039,
  title={{EMG from Combination Gestures with Ground-truth Joystick Labels}},
  author={
    Smedemark-Margulies, Niklas and 
    Bicer, Yunus and
    Sunger, Elifnur and
    Naufel, Stephanie and
    Imbiriba, Tales and
    Tunik, Eugene and
    Erdo{\u{g}}mu{\c{s}}, Deniz and
    Yarossi, Mathew
  },
  year={2023},
  month={12},
  day={15},
  publisher={Zenodo},
  version={1.0.0},
  doi={10.5281/zenodo.10358039},
  url={https://doi.org/10.5281/zenodo.10358039}
}
