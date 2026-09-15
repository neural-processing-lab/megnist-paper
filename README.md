# MegNIST

Code accompanying **MegNIST: A Benchmark for Non-Invasive Inner Speech Decoding**.

MegNIST is a magnetoencephalography (MEG) dataset for benchmarking
non-invasive inner-speech decoding. The dataset contains 12,000 trials
from a single participant involving covert speech for the
digits 0–9.

## Data

The dataset is hosted on Hugging Face:

https://huggingface.co/datasets/pnpl/MegNIST

The Hugging Face release contains the raw BIDS MEG data and
machine-learning-ready training, validation and test splits.

## Code

- `preprocess.py` — preprocessing of the raw MEG recordings.
- `serialise.py` — creation of the released HDF5 train/validation/test splits.
- `MegNIST_analysis.ipynb` — reproduction of the baseline decoding and
  technical-validation analyses reported in the paper.

## Installation

```bash
pip install -r requirements.txt
