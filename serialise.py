"""
MEG Data Serialization for MegNIST Dataset

This script converts preprocessed BIDS-formatted MEG data to HDF5 format
optimized for machine learning training.

Data splits:
- Train: Runs 1-10 (sessions 1-3 all runs, session 4 run 1)
- Validation: Run 11 (session 4 run 2)
- Test: Run 12 (session 4 run 3)

Run from: ./derivatives/code/
"""

import mne
import numpy as np
import h5py
from pathlib import Path
import json
import time
from functools import wraps


# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths (relative to derivatives/code/)
BIDS_ROOT = Path('../..')
DERIVATIVES_ROOT = BIDS_ROOT / 'derivatives'
PREPROC_ROOT = DERIVATIVES_ROOT / 'preproc'
SERIALISED_ROOT = DERIVATIVES_ROOT / 'serialised'
NEO_ROOT = DERIVATIVES_ROOT / 'neo'

# Subject info
SUBJECT = '0'

# Data splits - in chronological order
TRAIN_RUNS = [
    ('1', '1'), ('1', '2'), ('1', '3'),  # Session 1
    ('2', '1'), ('2', '2'), ('2', '3'),  # Session 2
    ('3', '1'), ('3', '2'), ('3', '3'),  # Session 3
    ('4', '1'),                           # Session 4, run 1
]

VAL_RUNS = [
    ('4', '2'),  # Session 4, run 2 (11th run)
]

TEST_RUNS = [
    ('4', '3'),  # Session 4, run 3 (12th run)
]

# Event ID mapping
EVENT_ID = {
    'digit/zero': 10,
    'digit/one': 11,
    'digit/two': 12,
    'digit/three': 13,
    'digit/four': 14,
    'digit/five': 15,
    'digit/six': 16,
    'digit/seven': 17,
    'digit/eight': 18,
    'digit/nine': 19,
}

# Load sensor positions from JSON
SENSOR_XYZ_FILE = NEO_ROOT / 'sensor_xyz.json'
print(f"Loading sensor positions from: {SENSOR_XYZ_FILE}")
with open(SENSOR_XYZ_FILE, 'r') as f:
    SENSOR_XYZ = np.array(json.load(f), dtype=np.float32)
print(f"Loaded {len(SENSOR_XYZ)} sensor positions")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def timeit(func):
    """Decorator that times the execution of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {execution_time/60:.2f} minutes")
        return result
    return wrapper


def get_epoch_file(subject, session, run):
    """
    Get the path to the preprocessed epoch file.
    
    Parameters
    ----------
    subject : str
        Subject ID
    session : str
        Session ID
    run : str
        Run ID
    
    Returns
    -------
    Path
        Path to epoch file
    """
    epo_file = (PREPROC_ROOT / f'sub-{subject}' / f'ses-{session}' / 'meg' / 
                f'sub-{subject}_ses-{session}_task-MegNIST_run-{run}_proc-bads+headpos+sss+notch+bp+ds_epo.fif')
    return epo_file


def get_bad_channels_file(subject, session, run):
    """Get the path to the bad channels JSON file."""
    bad_ch_file = (PREPROC_ROOT / f'sub-{subject}' / f'ses-{session}' / 'meg' / 
                   f'sub-{subject}_ses-{session}_task-MegNIST_run-{run}_bad_channels.json')
    return bad_ch_file


def get_split_files(subject, split_runs):
    """
    Get list of epoch files for a given split.
    
    Parameters
    ----------
    subject : str
        Subject ID
    split_runs : list of tuples
        List of (session, run) tuples
    
    Returns
    -------
    list
        List of existing epoch file paths
    """
    files = []
    missing_files = []
    
    for session, run in split_runs:
        epo_file = get_epoch_file(subject, session, run)
        if epo_file.exists():
            files.append(epo_file)
        else:
            missing_files.append(epo_file)
    
    if missing_files:
        print(f"WARNING: {len(missing_files)} epoch files not found:")
        for f in missing_files:
            print(f"  - {f}")
    
    return files


def collect_bad_channels(subject, split_runs):
    """
    Collect bad channels information across all runs in a split.
    
    Parameters
    ----------
    subject : str
        Subject ID
    split_runs : list of tuples
        List of (session, run) tuples
    
    Returns
    -------
    dict
        Dictionary mapping run identifiers to bad channels info
    """
    bad_channels_by_run = {}
    
    for session, run in split_runs:
        bad_ch_file = get_bad_channels_file(subject, session, run)
        run_id = f"ses-{session}_run-{run}"
        
        if bad_ch_file.exists():
            with open(bad_ch_file, 'r') as f:
                bad_channels_by_run[run_id] = json.load(f)
        else:
            print(f"  WARNING: Bad channels file not found for {run_id}")
            bad_channels_by_run[run_id] = {"noisy": [], "flat": [], "all": []}
    
    return bad_channels_by_run


# ============================================================================
# SERIALIZATION FUNCTION
# ============================================================================

@timeit
def bids2serialised(epo_files, split_runs, save_file, subject, dtype=np.float32, overwrite=False):
    """
    Convert epoched MEG data to HDF5 format optimized for ML training.
    
    Parameters
    ----------
    epo_files : list
        List of epoch file paths
    split_runs : list of tuples
        List of (session, run) tuples for this split
    save_file : str or Path
        Output HDF5 file path
    subject : str
        Subject ID
    dtype : numpy.dtype, default np.float32
        Data type for saving (float32 saves 50% space vs float64)
    overwrite : bool, default False
        If True, overwrite existing files. If False, skip existing files.
    """
    
    save_file = Path(save_file)
    
    # Skip if file already exists (unless overwrite=True)
    if save_file.exists() and not overwrite:
        print(f"Skipping {save_file.name} - already exists (use overwrite=True to overwrite)")
        return
    elif save_file.exists() and overwrite:
        print(f"Overwriting existing file: {save_file.name}")
    
    print(f"\nProcessing {len(epo_files)} epoch files...")
    
    # Map event codes (10-19) to class labels (0-9)
    label_map = {code: code - 10 for code in range(10, 20)}
    
    X_list = []
    y_list = []
    
    for i, epo_file in enumerate(epo_files):
        print(f"  Loading file {i+1}/{len(epo_files)}: {epo_file.name}")
        
        epo = mne.read_epochs(epo_file, preload=True, verbose=False)
        
        # Check for bad channels (should be none after preprocessing)
        if len(epo.info['bads']) > 0:
            print(f"    Warning: Contains bad channels: {epo.info['bads']}")
        
        # Filter for digit events only (10-19)
        keep_idx = np.isin(epo.events[:, 2], list(range(10, 20)))
        n_before = len(epo)
        epo = epo[keep_idx]
        n_after = len(epo)
        
        if n_after < n_before:
            print(f"    Note: Filtered out {n_before - n_after} non-digit events (button presses)")
        
        if len(epo) == 0:
            print(f"    Warning: No valid digit events found")
            continue
        
        # Get data (shape: n_epochs x n_channels x n_times)
        data = epo.get_data()  # float64 by default
        
        # Get labels
        codes = epo.events[:, 2]
        labels = np.array([label_map[c] for c in codes], dtype=np.int32)
        
        X_list.append(data)
        y_list.append(labels)
        
        print(f"    Loaded {len(epo)} epochs")
    
    if not X_list:
        raise ValueError("No valid epoch data found in any files!")
    
    # Concatenate all data
    X = np.concatenate(X_list, axis=0)  # (total_epochs, 306, 250)
    y = np.concatenate(y_list, axis=0)  # (total_epochs,)
    
    print(f"\nFinal data shape: {X.shape}")
    print(f"Final labels shape: {y.shape}")
    print(f"Label distribution: {np.bincount(y)}")
    
    # Convert to specified dtype
    if dtype != X.dtype:
        print(f"Converting data from {X.dtype} to {dtype}")
        X = X.astype(dtype)
    
    # Get time points and convert to specified dtype
    times = epo.times.astype(dtype)
    
    # Get channel info
    meg_picks = mne.pick_types(epo.info, meg=True, eeg=False, eog=False)
    channel_names = np.array([epo.ch_names[idx] for idx in meg_picks], dtype='S10')
    channel_types = np.array([mne.io.pick.channel_type(epo.info, idx) for idx in meg_picks], dtype='S15')
    
    # Verify expected dimensions
    assert X.shape[1] == 306, f"Expected 306 channels, got {X.shape[1]}"
    assert X.shape[2] == 250, f"Expected 250 time points, got {X.shape[2]}"
    assert len(SENSOR_XYZ) == 306, f"Expected 306 sensor positions, got {len(SENSOR_XYZ)}"
    
    # Collect bad channels information
    print(f"\nCollecting bad channels information...")
    bad_channels_by_run = collect_bad_channels(subject, split_runs)
    
    # Create output directory if needed
    save_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving to {save_file}...")
    print(f"  Using dtype: {dtype}")
    
    # Save to HDF5
    with h5py.File(save_file, "w") as f:
        # Create main datasets
        f.create_dataset("data", data=X, dtype=dtype)
        f.create_dataset("labels", data=y, dtype="int32")
        f.create_dataset("times", data=times, dtype=dtype)
        f.create_dataset("sensor_xyz", data=SENSOR_XYZ, dtype=dtype)
        
        # Create channel info datasets
        f.create_dataset("channel_names", data=channel_names)
        f.create_dataset("channel_types", data=channel_types)
        
        # Save metadata as attributes
        f.attrs["sample_frequency"] = epo.info["sfreq"]
        f.attrs["highpass_cutoff"] = epo.info['highpass']
        f.attrs["lowpass_cutoff"] = epo.info['lowpass']
        f.attrs["n_channels"] = len(channel_names)
        f.attrs["n_epochs"] = len(X)
        f.attrs["n_classes"] = 10
        f.attrs["tmin"] = -0.05
        f.attrs["tmax"] = 0.95
        f.attrs["event_id"] = json.dumps(EVENT_ID)
        f.attrs["processing_steps"] = "bads+headpos+sss+notch+bp+ds"
        f.attrs["mne_version"] = mne.__version__
        f.attrs["dtype"] = str(dtype)
        f.attrs["bad_channels_by_run"] = json.dumps(bad_channels_by_run)
    
    print(f"Successfully saved {len(X)} epochs to {save_file.name}")
    
    # Print file size
    file_size_mb = save_file.stat().st_size / (1024 * 1024)
    print(f"File size: {file_size_mb:.1f} MB")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main(overwrite=False):
    """
    Serialize all data splits.
    
    Parameters
    ----------
    overwrite : bool, default False
        If True, overwrite existing HDF5 files
    """
    print("="*70)
    print("MEG DATA SERIALIZATION")
    print("="*70)
    print(f"\nBIDS root: {BIDS_ROOT.resolve()}")
    print(f"Preprocessed data: {PREPROC_ROOT.resolve()}")
    print(f"Output directory: {SERIALISED_ROOT.resolve()}")
    print(f"\nSubject: sub-{SUBJECT}")
    
    # Define output files
    train_save = SERIALISED_ROOT / 'train.h5'
    val_save = SERIALISED_ROOT / 'val.h5'
    test_save = SERIALISED_ROOT / 'test.h5'
    
    # Get files for each split
    print("\n" + "-"*70)
    print("TRAIN SPLIT (Runs 1-10)")
    print("-"*70)
    print(f"Sessions and runs: {TRAIN_RUNS}")
    train_files = get_split_files(SUBJECT, TRAIN_RUNS)
    print(f"Found {len(train_files)}/{len(TRAIN_RUNS)} epoch files")
    
    print("\n" + "-"*70)
    print("VALIDATION SPLIT (Run 11)")
    print("-"*70)
    print(f"Sessions and runs: {VAL_RUNS}")
    val_files = get_split_files(SUBJECT, VAL_RUNS)
    print(f"Found {len(val_files)}/{len(VAL_RUNS)} epoch files")
    
    print("\n" + "-"*70)
    print("TEST SPLIT (Run 12)")
    print("-"*70)
    print(f"Sessions and runs: {TEST_RUNS}")
    test_files = get_split_files(SUBJECT, TEST_RUNS)
    print(f"Found {len(test_files)}/{len(TEST_RUNS)} epoch files")
    
    # Check if we have all files
    total_expected = len(TRAIN_RUNS) + len(VAL_RUNS) + len(TEST_RUNS)
    total_found = len(train_files) + len(val_files) + len(test_files)
    
    if total_found < total_expected:
        print(f"\nWARNING: Found {total_found}/{total_expected} epoch files")
        response = input("Continue with available files? (y/n): ")
        if response.lower() != 'y':
            print("Aborting serialization.")
            return
    
    # Serialize each split
    print("\n" + "="*70)
    print("SERIALIZING DATA")
    print("="*70)
    
    if train_files:
        print("\n" + "-"*70)
        print("Serializing TRAIN split...")
        print("-"*70)
        bids2serialised(train_files, TRAIN_RUNS, train_save, SUBJECT, dtype=np.float32, overwrite=overwrite)
    
    if val_files:
        print("\n" + "-"*70)
        print("Serializing VALIDATION split...")
        print("-"*70)
        bids2serialised(val_files, VAL_RUNS, val_save, SUBJECT, dtype=np.float32, overwrite=overwrite)
    
    if test_files:
        print("\n" + "-"*70)
        print("Serializing TEST split...")
        print("-"*70)
        bids2serialised(test_files, TEST_RUNS, test_save, SUBJECT, dtype=np.float32, overwrite=overwrite)
    
    print("\n" + "="*70)
    print("SERIALIZATION COMPLETE")
    print("="*70)
    print(f"\nOutput files:")
    if train_save.exists():
        print(f"  Train: {train_save}")
    if val_save.exists():
        print(f"  Val:   {val_save}")
    if test_save.exists():
        print(f"  Test:  {test_save}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def inspect_h5_file(h5_file):
    """
    Inspect the contents of an HDF5 file.
    
    Parameters
    ----------
    h5_file : str or Path
        Path to HDF5 file
    """
    h5_file = Path(h5_file)
    
    if not h5_file.exists():
        print(f"File not found: {h5_file}")
        return
    
    print(f"\nInspecting: {h5_file.name}")
    print("-"*70)
    
    with h5py.File(h5_file, 'r') as f:
        print("\nDatasets:")
        for key in f.keys():
            dataset = f[key]
            print(f"  {key}: shape={dataset.shape}, dtype={dataset.dtype}")
        
        print("\nAttributes:")
        for key, value in f.attrs.items():
            if key == 'bad_channels_by_run':
                print(f"  {key}: <JSON string>")
            else:
                print(f"  {key}: {value}")
        
        # Show sample of data
        if 'data' in f:
            print(f"\nData sample (first epoch, first channel, first 5 timepoints):")
            print(f"  {f['data'][0, 0, :5]}")
        
        if 'labels' in f:
            print(f"\nLabel distribution:")
            labels = f['labels'][:]
            for i in range(10):
                count = np.sum(labels == i)
                print(f"  Class {i}: {count} samples")
        
        if 'channel_names' in f:
            print(f"\nFirst 5 channel names:")
            names = f['channel_names'][:5]
            print(f"  {[n.decode() for n in names]}")
        
        if 'sensor_xyz' in f:
            print(f"\nFirst sensor position:")
            print(f"  {f['sensor_xyz'][0]}")


if __name__ == '__main__':
    # Set MNE logging level
    mne.set_log_level('WARNING')
    
    # Serialize data
    main(overwrite=False)
    
    # Optionally inspect the output files
    # inspect_h5_file(SERIALISED_ROOT / 'train.h5')
    # inspect_h5_file(SERIALISED_ROOT / 'val.h5')
    # inspect_h5_file(SERIALISED_ROOT / 'test.h5')
