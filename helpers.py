"""
Utility functions for MegNIST analysis notebooks

This module provides reusable functions for:
- Memory-efficient data loading with streaming
- Model definition (SimpleMLP - no dropout)
- Storage (Google Drive and HuggingFace)
- Robust statistics computation
- Proper seed management
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import pickle
import random
import time


# ============================================================================
# MODEL DEFINITION (EXACT MATCH TO ORIGINAL)
# ============================================================================

class SimpleMLP(nn.Module):
    """
    Simple Multi-Layer Perceptron for MEG digit decoding
    
    Architecture:
        Input → Hidden (ReLU) → Output (10 classes)
    
    Note: NO dropout in this model
    """
    def __init__(self, input_size: int, hidden_size: int, num_classes: int = 10):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# ============================================================================
# SEED MANAGEMENT (FOR REPRODUCIBILITY)
# ============================================================================

def set_all_seeds(seed: int):
    """
    Set all random seeds for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    """
    Initialize worker seeds for DataLoader
    Ensures each worker has different random state
    
    Args:
        worker_id: Worker ID from DataLoader
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================================
# MEMORY-EFFICIENT STREAMING DATA LOADING
# ============================================================================

class HDF5Dataset(Dataset):
    """
    Memory-efficient dataset that reads from HDF5 file on-the-fly
    
    Instead of loading all data into RAM, this dataset:
    - Opens HDF5 file for each sample access
    - Reads only the requested sample
    - Closes the file immediately
    
    This allows training on datasets larger than available RAM
    """
    
    def __init__(self, h5_path: str, transform=None):
        """
        Args:
            h5_path: Path to HDF5 file
            transform: Optional transform (e.g., StandardizeTransform)
        """
        self.h5_path = h5_path
        self.transform = transform
        
        # Open file temporarily to get length and shape
        with h5py.File(h5_path, 'r') as f:
            self.length = len(f['data'])
            self.data_shape = f['data'].shape[1:]
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Open file, read single sample, close file
        with h5py.File(self.h5_path, 'r') as f:
            X = f['data'][idx]
            y = f['labels'][idx]
        
        # Flatten if needed
        X = X.flatten()
        
        # Apply transform (standardization)
        if self.transform is not None:
            X = self.transform(X)
        
        return torch.FloatTensor(X), torch.LongTensor([y]).squeeze()


class StandardizeTransform:
    """
    Transform that applies z-score normalization
    
    Args:
        mean: Mean for each feature
        std: Standard deviation for each feature
    """
    
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, x):
        return (x - self.mean) / self.std


# ============================================================================
# ROBUST STATISTICS COMPUTATION (WELFORD'S ALGORITHM)
# ============================================================================

def compute_dataset_statistics(h5_path: str, batch_size: int = 1000) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute mean and std for standardization without loading all data
    Uses Welford's online algorithm for numerical stability
    
    Welford's algorithm:
    - Computes running mean and variance in a single pass
    - Numerically stable (avoids catastrophic cancellation)
    - Memory efficient (processes data in batches)
    
    Args:
        h5_path: Path to HDF5 file
        batch_size: Number of samples to process at once
        
    Returns:
        mean: Mean for each feature
        std: Standard deviation for each feature  
        elapsed_time: Time taken for computation
    """
    start_time = time.time()
    
    with h5py.File(h5_path, 'r') as f:
        n_samples = len(f['data'])
        sample_shape = f['data'][0].shape
        n_features = np.prod(sample_shape)
        
        # Initialize Welford's algorithm variables
        mean = np.zeros(n_features)
        M2 = np.zeros(n_features)  # Sum of squared differences from mean
        count = 0
        
        # Process in batches for memory efficiency
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch = f['data'][start_idx:end_idx]
            batch_flat = batch.reshape(batch.shape[0], -1)
            
            # Welford's online algorithm update
            for x in batch_flat:
                count += 1
                delta = x - mean
                mean += delta / count
                delta2 = x - mean
                M2 += delta * delta2
        
        # Compute standard deviation
        std = np.sqrt(M2 / count)
        std[std == 0] = 1.0  # Avoid division by zero
    
    elapsed_time = time.time() - start_time
    return mean, std, elapsed_time

# ============================================================================
# DATA LOADING
# ============================================================================

def load_hdf5_data(filepath: str, return_metadata: bool = False):
    """
    Load MEG data from HDF5 file
    
    Args:
        filepath: Path to .h5 file
        return_metadata: If True, also return channel info and sampling params
        
    Returns:
        X: Data array (n_trials, n_channels, n_timepoints) or (n_trials, n_features)
        y: Labels array (n_trials,)
        metadata: Dict with additional info (if return_metadata=True)
    """
    with h5py.File(filepath, 'r') as f:
        X = f['data'][:]
        y = f['labels'][:]
        
        if return_metadata:
            metadata = {
                'channel_names': [name.decode('utf-8') if isinstance(name, bytes) else str(name) 
                                 for name in f['channel_names'][:]],
                'channel_types': [ctype.decode('utf-8') if isinstance(ctype, bytes) else str(ctype) 
                                 for ctype in f['channel_types'][:]],
                'sensor_xyz': f['sensor_xyz'][:],
                'sfreq': float(f.attrs['sample_frequency']),
                'tmin': float(f.attrs['tmin'])
            }
            return X, y, metadata
        
    return X, y


def prepare_data_streaming(batch_size: int = 32, seed: int = 42, num_workers: int = 2):
    """
    Prepare streaming data loaders that don't load everything into RAM
    
    This function:
    1. Computes statistics from training data using Welford's algorithm
    2. Creates StandardizeTransform with those statistics
    3. Creates HDF5Dataset instances for train/val/test
    4. Returns DataLoaders with proper worker initialization
    
    Args:
        batch_size: Batch size for training
        seed: Random seed for reproducibility
        num_workers: Number of worker processes for data loading
        
    Returns:
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        test_loader: DataLoader for test set
        input_size: Number of input features
        stats_time: Time taken to compute statistics
    """
    from torch.utils.data import DataLoader
    
    set_all_seeds(seed)
    
    # Compute statistics from training data (only once per seed)
    train_mean, train_std, stats_time = compute_dataset_statistics('train.h5')
    
    # Create transform
    transform = StandardizeTransform(train_mean, train_std)
    
    # Create datasets
    train_dataset = HDF5Dataset('train.h5', transform=transform)
    val_dataset = HDF5Dataset('val.h5', transform=transform)
    test_dataset = HDF5Dataset('test.h5', transform=transform)
    
    # Get input size
    input_size = np.prod(train_dataset.data_shape)
    
    # Create loaders with proper seed management
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        worker_init_fn=worker_init_fn  # Critical for reproducibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, test_loader, input_size, stats_time


def flatten_data(X: np.ndarray) -> np.ndarray:
    """
    Flatten 3D MEG data to 2D for neural network input
    
    Args:
        X: Shape (n_trials, n_channels, n_timepoints)
        
    Returns:
        X_flat: Shape (n_trials, n_features) where n_features = n_channels * n_timepoints
    """
    if len(X.shape) == 3:
        n_trials = X.shape[0]
        return X.reshape(n_trials, -1)
    return X


def normalize_data(X: np.ndarray, mean: Optional[np.ndarray] = None, 
                  std: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
    """
    Z-score normalization
    
    Args:
        X: Data to normalize
        mean: Pre-computed mean (if None, compute from X)
        std: Pre-computed std (if None, compute from X)
        
    Returns:
        X_normalized: Normalized data
        params: Dict with 'mean' and 'std' for later use
    """
    if mean is None:
        mean = X.mean(axis=0, keepdims=True)
    if std is None:
        std = X.std(axis=0, keepdims=True)
        std[std == 0] = 1.0  # Avoid division by zero
    
    X_normalized = (X - mean) / std
    params = {'mean': mean, 'std': std}
    
    return X_normalized, params


# ============================================================================
# MODEL TRAINING & EVALUATION
# ============================================================================

def train_model(model: nn.Module, train_loader, val_loader, 
                epochs: int = 50, lr: float = 0.001, device: str = 'cuda',
                early_stopping_patience: int = 10, verbose: bool = True):
    """
    Train neural network with early stopping
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Maximum number of epochs
        lr: Learning rate
        device: 'cuda' or 'cpu'
        early_stopping_patience: Stop if no improvement for N epochs
        verbose: Print progress
        
    Returns:
        history: Dict with training/validation loss and accuracy per epoch
        best_model_state: State dict of best model
    """
    import torch.optim as optim
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)  # AdamW not Adam!
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += y_batch.size(0)
            train_correct += predicted.eq(y_batch).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100.0 * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += y_batch.size(0)
                val_correct += predicted.eq(y_batch).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100.0 * val_correct / val_total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if verbose and (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping_patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break
    
    return history, best_model_state


def evaluate_model(model: nn.Module, data_loader, device: str = 'cuda') -> Tuple[float, float]:
    """
    Evaluate model on dataset
    
    Args:
        model: PyTorch model
        data_loader: Data loader
        device: 'cuda' or 'cpu'
        
    Returns:
        loss: Average loss
        accuracy: Accuracy percentage
    """
    model.eval()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()
    
    avg_loss = total_loss / len(data_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


# ============================================================================
# STORAGE HELPERS
# ============================================================================

def setup_google_drive(base_folder: str = "MegNIST_Paper") -> Optional[Dict[str, Path]]:
    """
    Robustly mount Google Drive and create folder structure
    
    Args:
        base_folder: Name of folder in My Drive root
        
    Returns:
        Dict with paths, or None if mounting fails
    """
    try:
        from google.colab import drive
        
        print("Mounting Google Drive...")
        drive.mount('/content/drive', force_remount=False)
        
        drive_root = Path('/content/drive/MyDrive')
        if not drive_root.exists():
            print("❌ Drive mounted but MyDrive not found")
            return None
        
        print("✓ Drive mounted successfully")
        
        base_path = drive_root / base_folder
        paths = {
            'base': base_path,
            'models': base_path / 'models',
            'results': base_path / 'results', 
            'figures': base_path / 'figures',
            'checkpoints': base_path / 'checkpoints',
            'data': base_path / 'data',
        }
        
        for name, path in paths.items():
            path.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ {name}: {path}")
        
        # Test write access
        test_file = paths['base'] / '.test_write'
        try:
            test_file.write_text('test')
            test_file.unlink()
            print("✓ Write access confirmed")
        except Exception as e:
            print(f"❌ Cannot write to Drive: {e}")
            return None
        
        return paths
        
    except Exception as e:
        print(f"❌ Drive mounting failed: {e}")
        print("\nTroubleshooting:")
        print("1. Runtime → Disconnect and delete runtime")
        print("2. Reconnect and try again")
        print("3. Try: drive.mount('/content/drive', force_remount=True)")
        return None


def save_checkpoint(obj, filename: str, drive_paths: Optional[Dict] = None, 
                   folder_type: str = 'models'):
    """
    Save checkpoint to Drive or local storage
    
    Args:
        obj: Object to save (model state dict, results dict, etc.)
        filename: Name of file
        drive_paths: Dict from setup_google_drive()
        folder_type: Which subfolder to save to
    """
    if drive_paths:
        save_path = drive_paths[folder_type] / filename
        location = "Google Drive"
    else:
        save_path = Path('/content') / filename
        location = "local storage (ephemeral!)"
    
    print(f"Saving to {location}: {filename}...", end='', flush=True)
    
    try:
        if filename.endswith('.pth'):
            torch.save(obj, save_path)
        elif filename.endswith('.pkl'):
            with open(save_path, 'wb') as f:
                pickle.dump(obj, f)
        elif filename.endswith('.npy'):
            np.save(save_path, obj)
        else:
            raise ValueError(f"Unsupported file extension: {filename}")
        
        if save_path.exists():
            size_mb = save_path.stat().st_size / (1024*1024)
            print(f" ✓ ({size_mb:.1f} MB)")
            return str(save_path)
        else:
            print(f" ❌ Failed!")
            return None
            
    except Exception as e:
        print(f" ❌ Error: {e}")
        return None


def load_checkpoint(filename: str, drive_paths: Optional[Dict] = None,
                   folder_type: str = 'models', hf_repo: Optional[str] = None):
    """
    Load checkpoint from Drive or HuggingFace
    
    Args:
        filename: Name of file
        drive_paths: Dict from setup_google_drive()
        folder_type: Which subfolder to load from
        hf_repo: HuggingFace repo ID (fallback if Drive fails)
        
    Returns:
        Loaded object
    """
    # Try Drive first
    if drive_paths:
        drive_path = drive_paths[folder_type] / filename
        if drive_path.exists():
            print(f"Loading from Drive: {filename}")
            if filename.endswith('.pth'):
                return torch.load(drive_path)
            elif filename.endswith('.pkl'):
                with open(drive_path, 'rb') as f:
                    return pickle.load(f)
            elif filename.endswith('.npy'):
                return np.load(drive_path)
    
    # Fallback to HuggingFace
    if hf_repo:
        try:
            from huggingface_hub import hf_hub_download
            print(f"Loading from HuggingFace: {filename}")
            hf_path = hf_hub_download(
                repo_id=hf_repo,
                filename=f'derivatives/{folder_type}/{filename}',
                repo_type='dataset'
            )
            if filename.endswith('.pth'):
                return torch.load(hf_path)
            elif filename.endswith('.pkl'):
                with open(hf_path, 'rb') as f:
                    return pickle.load(f)
            elif filename.endswith('.npy'):
                return np.load(hf_path)
        except Exception as e:
            print(f"❌ HuggingFace download failed: {e}")
    
    raise FileNotFoundError(f"Could not find {filename} in Drive or HuggingFace")
