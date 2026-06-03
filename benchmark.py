import argparse
import fcntl
import gc
import logging
import multiprocessing as mp
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
from cristal import *
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm

# ================================
# ARGPARSE
# ================================


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--datasets_dir", "-d", type=str, required=True, help="Path to the directory containing the datasets")
    parser.add_argument("--output", "-o", type=str, default="results_benchmark.csv", help="Path to the output CSV file")
    parser.add_argument("--log_file", "-l", type=str, default="benchmark.log", help="Path to the log file")

    parser.add_argument("--n_jobs", "-j", type=int, default=os.cpu_count(), help="Number of parallel jobs")
    parser.add_argument("--timeout", "-to", type=int, default=7200, help="Timeout in seconds")

    parser.add_argument("--n_repeats", "-n", type=int, default=5, help="Number of repeats per dataset")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--test_size_ratio", "-ts", type=float, default=0.3, help="Test set size ratio")

    parser.add_argument("--ram_limit_gb", "-r", type=float, default=16, help="Maximum RAM per algo subprocess (GB)")
    parser.add_argument("--scaler", "-sc", choices=["standard", "minmax", "auto"], default="auto", help="Scaling method")

    return parser.parse_args()


ARGS = parse_args()


# ================================
# LOGGING
# ================================

logging.basicConfig(
    filename=ARGS.log_file,
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s"))
logging.getLogger().addHandler(console_handler)
logger = logging.getLogger()


# ================================
# ALGORITHMS
# ================================

from pyod.models.cblof import CBLOF
from pyod.models.ecod import ECOD
from pyod.models.gmm import GMM
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.kde import KDE
from pyod.models.knn import KNN
from pyod.models.loda import LODA
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA

BASELINES = {
    "KNN": {"model": KNN, "scaler": "standard"},
    "CBLOF": {"model": lambda: CBLOF(random_state=42), "scaler": "standard"},
    "GMM": {"model": lambda: GMM(random_state=42), "scaler": "standard"},
    "KDE": {"model": KDE, "scaler": "standard"},
    "HBOS": {"model": HBOS, "scaler": "standard"},
    "ECOD": {"model": ECOD, "scaler": "standard"},
    "OCSVM": {"model": OCSVM, "scaler": "standard"},
    "LODA": {"model": LODA, "scaler": "standard"},
    "IForest": {"model": lambda: IForest(random_state=42), "scaler": "standard"},
    "PCA": {"model": lambda: PCA(random_state=42), "scaler": "standard"},
}
CHRISTOFFEL_BASED = {
    "DyCF_2": {"model": lambda: DyCF(n=2), "scaler": "minmax"},
    "DyCF_3": {"model": lambda: DyCF(n=3), "scaler": "minmax"},
    "DyCF_4": {"model": lambda: DyCF(n=4), "scaler": "minmax"},
    "DyCF_5": {"model": lambda: DyCF(n=5), "scaler": "minmax"},
    "DyCG": {"model": lambda: DyCG(n_list=range(2, 6)), "scaler": "minmax"},
    "KernelCF_rbf": {"model": lambda: KernelCF(n=2), "scaler": "minmax"},  # RBF kernel by default. n is not used with this kernel
    "KernelCF_lin_2": {"model": lambda: KernelCF(n=2, kernel="linear"), "scaler": "minmax"},
    "KernelCF_lin_3": {"model": lambda: KernelCF(n=3, kernel="linear"), "scaler": "minmax"},
    "KernelCF_lin_4": {"model": lambda: KernelCF(n=4, kernel="linear"), "scaler": "minmax"},
    "KernelCF_lin_5": {"model": lambda: KernelCF(n=5, kernel="linear"), "scaler": "minmax"},
    # "KernelCG": {"model": lambda: KernelCG(n_list=range(2, 6), kernel="linear"), "scaler": "minmax"},
    "UCF_2": {"model": lambda: UCF(n=2), "scaler": "minmax"},
    "UCF_3": {"model": lambda: UCF(n=3), "scaler": "minmax"},
    "UCF_4": {"model": lambda: UCF(n=4), "scaler": "minmax"},
    "UCF_5": {"model": lambda: UCF(n=5), "scaler": "minmax"},
    "UCF_6": {"model": lambda: UCF(n=6), "scaler": "minmax"},
    "UCF_7": {"model": lambda: UCF(n=7), "scaler": "minmax"},
    "UCF_8": {"model": lambda: UCF(n=8), "scaler": "minmax"},
    "UCG": {"model": lambda: UCG(n_list=range(2, 6)), "scaler": "minmax"},
}

ALGORITHMS = BASELINES | CHRISTOFFEL_BASED

# ================================
# SCALERS
# ================================

SCALERS = {"standard": StandardScaler, "minmax": lambda: MinMaxScaler(feature_range=(-1, 1))}


# ================================
# HELPERS
# ================================


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_data(path):
    data = np.load(path)
    return data["X"], data["y"]


def scale_data(X_train, X_test, scaler_type):
    scaler = SCALERS[scaler_type]()
    return scaler.fit_transform(X_train), scaler.transform(X_test)


def anomaly_split(X, y, seed):
    X_train, X_test, _, y_test = train_test_split(X, y, stratify=y, test_size=ARGS.test_size_ratio, random_state=seed)
    return X_train, X_test, y_test


def load_completed():
    if not os.path.exists(ARGS.output):
        return set()
    df = pd.read_csv(ARGS.output)
    return set((r.dataset, r.algorithm, r.repeat) for r in df.itertuples() if r.status == "ok" or r.status == "timeout")


def write_result(result_dict):
    """Write a single result row to CSV with a file lock to prevent race conditions."""
    row = pd.DataFrame([result_dict])
    header = not os.path.exists(ARGS.output)
    with open(ARGS.output, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        row.to_csv(f, header=header, index=False)
        fcntl.flock(f, fcntl.LOCK_UN)


# ================================
# ISOLATED ALGO SUBPROCESS
# ================================


def _algo_subprocess(result_queue, X_train, X_test, y_test, algo_name, scaler_override, ram_limit_gb):
    """
    Runs in a fully isolated subprocess via mp.Process.
    Any crash (OOM, segfault, RuntimeError) only kills this process,
    leaving the parent pool completely unaffected.
    """
    try:
        # # Enforce RAM limit at OS level for this subprocess
        # import resource

        # limit_bytes = int(ram_limit_gb * 1024**3)
        # resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))

        conf = ALGORITHMS[algo_name]

        scaler_type = conf["scaler"] if scaler_override == "auto" else scaler_override
        X_train_s, X_test_s = scale_data(X_train, X_test, scaler_type)

        t_total = time.perf_counter()

        model = conf["model"]()

        t0 = time.perf_counter()
        model.fit(X_train_s)
        train_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        scores = model.decision_function(X_test_s)
        inference_time = time.perf_counter() - t0

        total_time = time.perf_counter() - t_total

        auroc = roc_auc_score(y_test, scores)
        auprc = average_precision_score(y_test, scores)

        result_queue.put(("ok", train_time, inference_time, total_time, auroc, auprc))

    except Exception as exc:
        result_queue.put(("error", exc))


def run_algo_isolated(X_train, X_test, y_test, algo_name):
    """
    Spawns a fresh subprocess for each algo run.
    - Timeout is enforced by killing the process.
    - OOM or any crash is caught via exit code, never propagates up.
    """
    torch.cuda.empty_cache()
    gc.collect()

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    p = ctx.Process(
        target=_algo_subprocess,
        args=(result_queue, X_train, X_test, y_test, algo_name, ARGS.scaler, ARGS.ram_limit_gb),
    )
    p.start()
    p.join(timeout=ARGS.timeout)

    # Timed out: process still alive
    if p.is_alive():
        p.kill()
        p.join()
        raise TimeoutError(f"Algo timed out after {ARGS.timeout}s")

    # Process was killed by OS (OOM, segfault, etc.) -> non-zero exit code
    if p.exitcode != 0:
        raise MemoryError(f"Subprocess killed by OS (exit code {p.exitcode}) — likely OOM or segfault")

    # Process exited cleanly: retrieve result
    result = result_queue.get()
    if result[0] == "ok":
        _, train_time, inference_time, total_time, auroc, auprc = result
        return train_time, inference_time, total_time, auroc, auprc
    else:
        raise result[1] from result[1]  # re-raise exception from subprocess


# ================================
# DATASET WORKER
# ================================


def dataset_worker(dataset_path, completed):
    dataset = os.path.basename(dataset_path).replace(".npz", "")

    try:
        X, y = load_data(dataset_path)
    except Exception:
        logger.exception("Dataset '%s' load failed", dataset)
        return

    for repeat in range(1, 1 + ARGS.n_repeats):
        seed = ARGS.seed + repeat - 1
        set_seed(seed)

        try:
            X_train, X_test, y_test = anomaly_split(X, y, seed)
        except Exception:
            logger.exception("Dataset '%s' split failed (repeat %d)", dataset, repeat)
            continue

        for algo_name in ALGORITHMS.keys():
            if (dataset, algo_name, repeat) in completed:
                continue

            logger.info("Running '%s' on '%s' (repeat %d)", algo_name, dataset, repeat)

            try:
                train_time, inference_time, total_time, auroc, auprc = run_algo_isolated(X_train, X_test, y_test, algo_name)
                status = "ok"
                logger.info(
                    "'%s' on '%s' (repeat %d) done in %.2fs — AUROC %.4f, AUPRC %.4f",
                    algo_name,
                    dataset,
                    repeat,
                    total_time,
                    auroc,
                    auprc,
                )

            except TimeoutError as exc:
                logger.warning("Timeout — '%s' on '%s' (repeat %d): %s", algo_name, dataset, repeat, exc)
                train_time = inference_time = total_time = auroc = auprc = np.nan
                status = "timeout"

            except MemoryError as exc:
                logger.warning("OOM — '%s' on '%s' (repeat %d): %s", algo_name, dataset, repeat, exc)
                train_time = inference_time = total_time = auroc = auprc = np.nan
                status = "ram_kill"

            except Exception:
                logger.exception("Error — '%s' on '%s' (repeat %d)", algo_name, dataset, repeat)
                train_time = inference_time = total_time = auroc = auprc = np.nan
                status = "error"

            write_result(
                {
                    "dataset": dataset,
                    "algorithm": algo_name,
                    "repeat": repeat,
                    "train_time": train_time,
                    "inference_time": inference_time,
                    "total_time": total_time,
                    "auroc": auroc,
                    "auprc": auprc,
                    "status": status,
                }
            )


# ================================
# BENCHMARK
# ================================


def run_benchmark():
    datasets_paths = [os.path.join(ARGS.datasets_dir, f) for f in os.listdir(ARGS.datasets_dir) if f.endswith(".npz")]

    completed = load_completed()
    logger.info("%d runs already completed, skipping", len(completed))

    with ProcessPoolExecutor(max_workers=ARGS.n_jobs) as executor:
        futures = {executor.submit(dataset_worker, d, completed): d for d in datasets_paths}
        for future in tqdm(as_completed(futures), total=len(futures)):
            d = futures[future]
            try:
                future.result()
            except Exception:
                logger.exception("Unexpected error for dataset '%s'", d)


# ================================
# MAIN
# ================================

if __name__ == "__main__":
    logger.info("Benchmark start")
    run_benchmark()
    logger.info("Benchmark finished")
