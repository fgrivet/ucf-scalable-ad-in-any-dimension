# UCF: A Univariate Christoffel Function For Scalable Anomaly Detection In Any Dimension <!-- omit from toc -->

## Overview

This repository contains all the necessary code and resources to reproduce the benchmark results presented in our article.

### Repository Contents

| File | Description |
|------|-------------|
| `analyze_benchmark_results.ipynb` | Jupyternotebook containing the code to analyze benchmark results and generate tables/figures for the article. Requires `datasets_properties.csv` and `results_benchmark.csv`. |
| `benchmark.py` | Python script to execute the benchmark and generate `results_benchmark.csv`. Run with: `python benchmark.py -d path/to/ADBench/datasets/dir` |
| `datasets_properties.csv` | CSV file containing dataset descriptions including: dataset number, name, sample count, dimensions, anomaly count, anomaly rate, category, and anomaly ratio category. |
| `dataset_properties.ipynb` | Jupyter notebook for categorizing datasets and generating tables/figures about datasets for the article. Outputs `datasets_properties.csv`. |
| `requirements.txt` | List of all Python libraries required to execute the files in this repository. Install with: `pip install -r requirements.txt` |
| `results_benchmark.csv` | CSV file containing benchmark results including: dataset name, algorithm name, repetition number, training time, inference time, total execution time, AUROC score, AUPRC score, and execution status. |


## Table of Contents <!-- omit from toc -->

- [Overview](#overview)
  - [Repository Contents](#repository-contents)
- [Main results](#main-results)
- [Installation](#installation)
- [Usage](#usage)
  - [Benchmark](#benchmark)
    - [Parameters](#parameters)
  - [Dataset properties](#dataset-properties)
  - [Analyze benchmark results](#analyze-benchmark-results)
- [Citation](#citation)


## Main results


## Installation

Note that this code works on linux only.

1. Clone the repository 
```
git clone https://github.com/fgrivet/ucf-scalable-ad-in-any-dimension
```
2. Move to the folder 
```
cd ucf-scalable-ad-in-any-dimension
```
3. Install the dependencies
```
pip install -r requirements.txt
```

You may have to install the [PyOD](https://github.com/yzhao062/pyod) and [CRISTAL](https://github.com/fgrivet/CRISTAL) libraries from GitHub directly.


## Usage

### Benchmark
Note that this code works on linux only.

To run the benchmark, execute the following command
```
python benchmark.py -d path/to/ADBench/datasets/dir
```

#### Parameters
| Short | Long | Description | Type | Default |
|:-----:|:----:|:-----------:|:----:|:-------:|
| -d | --datasets_dir | Path to the directory containing the datasets | str | |
| -o | --output | Path to the output CSV file | str | `"results_benchmark.csv"` |
| -l | --log_file | Path to the log file | str | `"benchmark.log"` |
| -j | --n_jobs | Number of parallel jobs | int > 0 or `-1` | `-1` |
| -to | --timeout | Timeout in seconds | int > 0 | `7200` |
| -n | --n_repeats | Number of repeats per dataset | int > 0 | `5` |
| -s | --seed | Random seed for reproducibility | int | `42` |
| -ts | --test_size_ratio | Test set size ratio | float | `0.3` |
| -r | --ram_limit_gb | Maximum RAM per algo subprocess (GB) | float | `16` |
| -sc | --scaler | Scaling method | `"standard"` or `"minmax"` or `"auto"` | `"auto"` |

If the scaler is set to `"auto"`, the default value defined in the code is used, i.e. `"standard"` for the baselines and `"minmax"` for the Christoffel-based.

### Dataset properties
Jupyter notebook for categorizing datasets and generating tables/figures about datasets for the article. Outputs `datasets_properties.csv`.

In cell 2, replace ```ADBENCH_DIR = "path/to/ADBench/datasets/dir"``` by the path to the folder containing all the datasets from ADBench.

### Analyze benchmark results
Jupyternotebook containing the code to analyze benchmark results and generate tables/figures for the article. Requires `datasets_properties.csv` and `results_benchmark.csv`.

Contains utils functions to generate more stats and figures.

## Citation

If you use this code or reference our work in your research, please cite our article as:

```

```