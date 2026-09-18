#!/usr/bin/env python3
"""
Reproduce the MegNIST baseline hyperparameter search reported in the paper.

Paper configuration
-------------------
- Hidden sizes: 32, 64, 128, 256, 512, 1024, 2048
- Seeds: 42..51
- AdamW, learning rate 1e-3
- Batch size 32
- Maximum 200 epochs
- Early stopping on validation accuracy, patience 20
- Feature-wise z-scoring using training-set statistics only
- Representative seed: chosen after architecture selection as the run closest
  to the median validation accuracy; validation ties are broken by closeness to
  the median test accuracy, matching the released analysis.

The full search is intentionally a standalone script rather than a notebook
cell because it is slow. The paper-analysis notebook should load the released
results/checkpoint by default and point readers here for full retraining.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import pickle
import random
import shutil
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader, Dataset


PAPER_HIDDEN_SIZES = [32, 64, 128, 256, 512, 1024, 2048]
PAPER_SEEDS = list(range(42, 52))
HF_REPO = "pnpl/MegNIST"


def set_all_seeds(seed: int) -> None:
    """Match the seed handling used in the original training code."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class HDF5Dataset(Dataset):
    """Memory-efficient dataset matching the original paper implementation."""

    def __init__(self, h5_path: Path, mean: np.ndarray, std: np.ndarray):
        self.h5_path = str(h5_path)
        self.mean = mean
        self.std = std
        with h5py.File(self.h5_path, "r") as f:
            self.length = len(f["data"])
            self.data_shape = tuple(f["data"].shape[1:])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        with h5py.File(self.h5_path, "r") as f:
            x = f["data"][idx]
            y = f["labels"][idx]

        x = x.flatten()
        x = (x - self.mean) / self.std
        return torch.FloatTensor(x), torch.tensor(int(y), dtype=torch.long)


class SimpleMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def compute_dataset_statistics(
    h5_path: Path, batch_size: int = 1000
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sequential Welford statistics, deliberately matching the released
    training implementation.
    """
    with h5py.File(h5_path, "r") as f:
        n_samples = len(f["data"])
        sample_shape = f["data"][0].shape
        n_features = int(np.prod(sample_shape))

        mean = np.zeros(n_features)
        m2 = np.zeros(n_features)
        count = 0

        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch = f["data"][start_idx:end_idx]
            batch_flat = batch.reshape(batch.shape[0], -1)

            for x in batch_flat:
                count += 1
                delta = x - mean
                mean += delta / count
                delta2 = x - mean
                m2 += delta * delta2

    std = np.sqrt(m2 / count)
    std[std == 0] = 1.0
    return mean, std


def make_loaders(
    paths: dict[str, Path],
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    seed: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    # The original implementation reset the seed before constructing loaders.
    set_all_seeds(seed)

    train_dataset = HDF5Dataset(paths["train"], mean, std)
    val_dataset = HDF5Dataset(paths["val"], mean, std)
    test_dataset = HDF5Dataset(paths["test"], mean, std)
    input_size = int(np.prod(train_dataset.data_shape))

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        worker_init_fn=worker_init_fn,
        **common,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    test_loader = DataLoader(test_dataset, shuffle=False, **common)
    return train_loader, val_loader, test_loader, input_size


def accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            pred = model(inputs).argmax(dim=1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    return 100.0 * correct / total


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    max_epochs: int,
    learning_rate: float,
    patience: int,
) -> dict:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    train_accs = []
    val_accs = []
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    best_model_state = None

    for epoch in range(max_epochs):
        model.train()
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = outputs.argmax(dim=1)
            train_total += labels.size(0)
            train_correct += (pred == labels).sum().item()

        train_acc = 100.0 * train_correct / train_total
        val_acc = accuracy(model, val_loader, device)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                f"    epoch {epoch + 1:03d}: "
                f"train={train_acc:5.2f}% val={val_acc:5.2f}%"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(
                f"    early stop at epoch {epoch + 1}; "
                f"best val={best_val_acc:.2f}% at epoch {best_epoch + 1}"
            )
            break

    if best_model_state is None:
        raise RuntimeError("No best model state was recorded.")

    model.load_state_dict(best_model_state)
    return {
        "model": model,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
    }


def resolve_data(data_dir: Path, no_download: bool) -> dict[str, Path]:
    """
    Prefer local train.h5/val.h5/test.h5; otherwise download the released
    serialised splits from pnpl/MegNIST.
    """
    paths = {split: data_dir / f"{split}.h5" for split in ("train", "val", "test")}
    if all(path.exists() for path in paths.values()):
        return paths

    if no_download:
        missing = [str(p) for p in paths.values() if not p.exists()]
        raise FileNotFoundError("Missing data files: " + ", ".join(missing))

    print("Local serialised splits not found; downloading from Hugging Face...")
    return {
        split: Path(
            hf_hub_download(
                repo_id=HF_REPO,
                repo_type="dataset",
                filename=f"derivatives/serialised/{split}.h5",
            )
        )
        for split in ("train", "val", "test")
    }


def validate_paper_splits(paths: dict[str, Path]) -> None:
    expected_n = {"train": 10_000, "val": 1_000, "test": 1_000}
    for split, path in paths.items():
        with h5py.File(path, "r") as f:
            shape = tuple(f["data"].shape)
            labels = len(f["labels"])
        if shape != (expected_n[split], 306, 250) or labels != expected_n[split]:
            raise ValueError(
                f"Unexpected {split} split: data shape={shape}, labels={labels}. "
                "This script is intended for the released paper splits."
            )


def run_once(
    hidden_size: int,
    seed: int,
    paths: dict[str, Path],
    mean: np.ndarray,
    std: np.ndarray,
    args,
    device: torch.device,
) -> tuple[dict, int]:
    train_loader, val_loader, test_loader, input_size = make_loaders(
        paths=paths,
        mean=mean,
        std=std,
        batch_size=args.batch_size,
        seed=seed,
        num_workers=args.num_workers,
    )

    # Match the second seed reset immediately before model initialisation in
    # the original training code.
    set_all_seeds(seed)
    model = SimpleMLP(input_size, hidden_size, num_classes=10).to(device)

    t0 = time.time()
    trained = train_model(
        model,
        train_loader,
        val_loader,
        device,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
    )
    test_acc = accuracy(trained["model"], test_loader, device)
    elapsed = time.time() - t0

    result = {
        "hidden_size": hidden_size,
        "seed": seed,
        "val_acc": trained["best_val_acc"],
        "test_acc": test_acc,
        "best_epoch": trained["best_epoch"],
        "train_acc": trained["train_accs"][trained["best_epoch"]],
        "train_accs": trained["train_accs"],
        "val_accs": trained["val_accs"],
        "elapsed_seconds": elapsed,
    }

    del model, trained, train_loader, val_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result, input_size


def choose_representative(
    all_results: dict[int, list[dict]],
    seeds: list[int],
) -> tuple[int, int, int]:
    summary = []
    for hidden_size, results in all_results.items():
        vals = np.asarray([r["val_acc"] for r in results], dtype=float)
        summary.append((hidden_size, float(vals.mean())))

    best_hidden = max(summary, key=lambda item: item[1])[0]
    results = all_results[best_hidden]
    val_accs = np.asarray([r["val_acc"] for r in results], dtype=float)
    test_accs = np.asarray([r["test_acc"] for r in results], dtype=float)

    median_val = np.median(val_accs)
    val_distances = np.abs(val_accs - median_val)
    tied = np.flatnonzero(val_distances == val_distances.min())

    if len(tied) > 1:
        median_test = np.median(test_accs)
        test_distances = np.abs(test_accs[tied] - median_test)
        selected_idx = int(tied[np.argmin(test_distances)])
    else:
        selected_idx = int(tied[0])

    return best_hidden, seeds[selected_idx], selected_idx


def save_summary(
    all_results: dict[int, list[dict]],
    hidden_sizes: list[int],
    seeds: list[int],
    best_hidden: int,
    selected_seed: int,
    selected_idx: int,
    output_dir: Path,
) -> None:
    selected = all_results[best_hidden][selected_idx]
    released = {
        "architectures": hidden_sizes,
        "seeds": seeds,
        "val_accuracies": {
            h: [r["val_acc"] for r in all_results[h]] for h in hidden_sizes
        },
        "test_accuracies": {
            h: [r["test_acc"] for r in all_results[h]] for h in hidden_sizes
        },
        "best_architecture": best_hidden,
        "best_seed": selected_seed,
        "selected_val_acc": selected["val_acc"],
        "selected_test_acc": selected["test_acc"],
    }

    # Both names are written because earlier reproduction materials used
    # results_all_architectures.pkl, while the current analysis notebook uses
    # all_architectures.pkl in the paper repository.
    for name in ("all_architectures.pkl", "results_all_architectures.pkl"):
        with open(output_dir / name, "wb") as f:
            pickle.dump(released, f)

    detailed = {
        str(h): [
            {
                k: v
                for k, v in r.items()
                if k not in {"train_accs", "val_accs"}
            }
            for r in all_results[h]
        ]
        for h in hidden_sizes
    }
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(detailed, f, indent=2)


def retrain_representative_checkpoint(
    best_hidden: int,
    selected_seed: int,
    expected_result: dict,
    input_size: int,
    paths: dict[str, Path],
    mean: np.ndarray,
    std: np.ndarray,
    args,
    device: torch.device,
    output_dir: Path,
) -> None:
    """
    Retrain the selected run once so the full search does not need to retain
    ~12 GB of model states in memory/disk. This should reproduce the selected
    run under the same software/hardware configuration.
    """
    print(
        f"\nRetraining representative model for checkpoint: "
        f"H={best_hidden}, seed={selected_seed}"
    )
    train_loader, val_loader, test_loader, check_input_size = make_loaders(
        paths=paths,
        mean=mean,
        std=std,
        batch_size=args.batch_size,
        seed=selected_seed,
        num_workers=args.num_workers,
    )
    if check_input_size != input_size:
        raise RuntimeError("Input-size mismatch during representative retraining.")

    set_all_seeds(selected_seed)
    model = SimpleMLP(input_size, best_hidden, num_classes=10).to(device)
    trained = train_model(
        model,
        train_loader,
        val_loader,
        device,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
    )
    test_acc = accuracy(trained["model"], test_loader, device)

    if (
        not np.isclose(trained["best_val_acc"], expected_result["val_acc"])
        or not np.isclose(test_acc, expected_result["test_acc"])
    ):
        print(
            "WARNING: representative retraining did not exactly match the first "
            "pass. This can occur with nondeterministic GPU/library behaviour."
        )

    checkpoint = {
        "model_state_dict": {
            k: v.detach().cpu() for k, v in trained["model"].state_dict().items()
        },
        "architecture": {
            "input_size": input_size,
            "hidden_size": best_hidden,
            "output_size": 10,
        },
        "training_info": {
            "seed": selected_seed,
            "val_accuracy": trained["best_val_acc"],
            "test_accuracy": test_acc,
            "best_epoch": trained["best_epoch"],
        },
        "normalization_params": None,
    }
    model_path = output_dir / f"best_model_seed{selected_seed}.pth"
    torch.save(checkpoint, model_path)
    print(f"Saved representative checkpoint: {model_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Full 7-architecture x 10-seed MegNIST paper baseline search"
    )
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/training"))
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--device",
        default=None,
        help="e.g. cuda, cuda:0 or cpu; default chooses CUDA when available",
    )

    # Defaults below are the paper settings. The options are exposed mainly
    # to permit short smoke tests without editing the file.
    p.add_argument("--hidden-sizes", type=int, nargs="+", default=PAPER_HIDDEN_SIZES)
    p.add_argument("--seeds", type=int, nargs="+", default=PAPER_SEEDS)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument(
        "--skip-checkpoint",
        action="store_true",
        help="Do not retrain/save the final representative model checkpoint.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")
    print(f"Hidden sizes: {args.hidden_sizes}")
    print(f"Seeds: {args.seeds}")
    print(f"Total training runs: {len(args.hidden_sizes) * len(args.seeds)}")

    paths = resolve_data(args.data_dir, args.no_download)
    validate_paper_splits(paths)
    for split, path in paths.items():
        print(f"{split:>5}: {path}")

    stats_path = args.output_dir / "training_standardisation.npz"
    if stats_path.exists():
        cached = np.load(stats_path)
        mean = cached["mean"]
        std = cached["std"]
        print(f"Loaded training statistics: {stats_path}")
    else:
        print("Computing feature-wise training-set statistics...")
        mean, std = compute_dataset_statistics(paths["train"])
        np.savez_compressed(stats_path, mean=mean, std=std)
        print(f"Saved training statistics: {stats_path}")

    all_results: dict[int, list[dict]] = {}
    input_size = None
    overall_start = time.time()

    for hidden_size in args.hidden_sizes:
        all_results[hidden_size] = []
        print(f"\n{'=' * 72}\nHIDDEN SIZE {hidden_size}\n{'=' * 72}")

        for seed in args.seeds:
            print(f"\n  Seed {seed}")
            result, this_input_size = run_once(
                hidden_size, seed, paths, mean, std, args, device
            )
            input_size = this_input_size if input_size is None else input_size
            if this_input_size != input_size:
                raise RuntimeError("Input-size mismatch between runs.")
            all_results[hidden_size].append(result)
            print(
                f"  val={result['val_acc']:.2f}% "
                f"test={result['test_acc']:.2f}% "
                f"best_epoch={result['best_epoch'] + 1}"
            )

    best_hidden, selected_seed, selected_idx = choose_representative(
        all_results, args.seeds
    )
    selected = all_results[best_hidden][selected_idx]

    print("\n" + "=" * 72)
    print(f"Selected architecture: {best_hidden}")
    print(f"Representative seed:   {selected_seed}")
    print(f"Validation accuracy:    {selected['val_acc']:.2f}%")
    print(f"Test accuracy:          {selected['test_acc']:.2f}%")
    print(f"Total elapsed:          {(time.time() - overall_start) / 3600:.2f} h")
    print("=" * 72)

    save_summary(
        all_results,
        args.hidden_sizes,
        args.seeds,
        best_hidden,
        selected_seed,
        selected_idx,
        args.output_dir,
    )
    print(f"Saved results under: {args.output_dir}")

    if not args.skip_checkpoint:
        retrain_representative_checkpoint(
            best_hidden,
            selected_seed,
            selected,
            input_size,
            paths,
            mean,
            std,
            args,
            device,
            args.output_dir,
        )


if __name__ == "__main__":
    main()
