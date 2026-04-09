"""Spiking Heidelberg Digits (SHD) tutorial using tonic."""

import torch
from torch.utils.data import DataLoader
import numpy as np

try:
    import tonic
    import tonic.transforms as transforms

    HAS_TONIC = True
except ImportError:
    HAS_TONIC = False


def get_task_defaults() -> dict:
    """Return default overrides for SHD."""
    return {
        "T": 100,
        "dt": 1.0,
        "batch_size": 64,
        "lr": 1e-3,
        "n_neuron": 256,
    }


class SHDDataset:
    """Wrapper for tonic SHD dataset with time binning."""

    def __init__(self, dataset, time_bins: int = 100, dt: float = 1.0):
        self.dataset = dataset
        self.time_bins = time_bins
        self.dt = dt

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        events, label = self.dataset[idx]

        # Convert events to spike tensor
        # events: (t, x, p) where t is time in us, x is channel (0-699), p is polarity
        spike_tensor = self._events_to_tensor(events)

        return spike_tensor, label

    def _events_to_tensor(self, events) -> torch.Tensor:
        """Convert events to binned spike tensor."""
        if len(events) == 0:
            return torch.zeros(self.time_bins, 700)

        # Extract times and channels
        times = events["t"].astype(np.float32) / 1e6  # Convert us to seconds
        channels = events["x"].astype(np.int64)

        # Normalize time to [0, time_bins)
        t_max = times.max()
        if t_max > 0:
            times = times / t_max * (self.time_bins - 1)

        # Bin spikes
        spike_tensor = torch.zeros(self.time_bins, 700)
        time_indices = torch.clamp(times.astype(np.int64), 0, self.time_bins - 1)

        for t, ch in zip(time_indices, channels):
            if 0 <= ch < 700:
                spike_tensor[t, ch] = 1.0

        return spike_tensor


def get_dataloaders(config) -> tuple:
    """
    Get SHD dataloaders.

    Returns:
        train_loader, test_loader, input_dim, output_dim, T
    """
    if not HAS_TONIC:
        raise ImportError(
            "tonic is required for SHD dataset. Install with: pip install tonic"
        )

    time_bins = getattr(config, "T", 100)
    dt = getattr(config, "dt", 1.0)
    batch_size = config.batch_size
    data_dir = config.data_dir

    # Load tonic datasets
    train_dataset_raw = tonic.datasets.SHD(
        save_to=data_dir,
        train=True,
    )
    test_dataset_raw = tonic.datasets.SHD(
        save_to=data_dir,
        train=False,
    )

    # Wrap with our preprocessing
    train_dataset = SHDDataset(train_dataset_raw, time_bins=time_bins, dt=dt)
    test_dataset = SHDDataset(test_dataset_raw, time_bins=time_bins, dt=dt)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, test_loader, 700, 20, time_bins
