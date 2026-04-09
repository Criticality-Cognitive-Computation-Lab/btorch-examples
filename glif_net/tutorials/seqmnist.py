"""Sequential MNIST tutorial with threshold-based encoding (paper scheme)."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np


def get_task_defaults() -> dict:
    """Return default overrides for sequential MNIST (paper scheme)."""
    return {
        "n_neuron": 220,
        "n_adapt": 100,  # 100 neurons with SFA, 120 without
        "asc_amp": -1.8,  # mV (adaptation strength)
        "tau_adapt": 700.0,  # ms
        "tau_mem": 20.0,
        "tau_ref": 5.0,
        "v_threshold": -10.0,  # mV (baseline threshold)
        "T": 784,
        "dt": 1.0,
        "batch_size": 256,
        "lr": 0.01,
        "lr_decay_every": 2500,
        "lr_decay_factor": 0.8,
        "rate_weight": 0.1,
        "voltage_weight": 0.0,  # Not mentioned in paper
        "readout_tau": 20.0,
        "epochs": 100,
    }


class ThresholdEncodedMNIST(Dataset):
    """
    Sequential MNIST with threshold-based input encoding.

    Input encoding: 80 input neurons, each with a threshold for gray value.
    Neuron i fires when gray value crosses threshold[i] from previous to current pixel.

    Paper scheme: 80 input neurons with thresholds evenly spaced in [0, 1].
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        n_input_neurons: int = 80,
        download: bool = True,
    ):
        # Load MNIST
        transform = transforms.Compose(
            [
                transforms.ToTensor(),  # Converts to [0, 1]
            ]
        )

        self.mnist = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=transform,
        )

        self.n_input_neurons = n_input_neurons
        self.T = 784  # 28x28 pixels presented sequentially

        # Create thresholds evenly spaced in [0, 1]
        # Paper uses 80 input neurons with thresholds evenly spaced
        self.thresholds = torch.linspace(0, 1, n_input_neurons)

    def __len__(self) -> int:
        return len(self.mnist)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        Returns:
            spikes: (T, n_input_neurons) binary spike tensor
            label: class label
        """
        image, label = self.mnist[idx]

        # Flatten to sequence: (1, 28, 28) -> (784,)
        pixels = image.view(-1)  # (784,)

        # Threshold encoding: for each pixel, which thresholds are crossed?
        # Neuron i fires when pixel value crosses threshold[i] from previous pixel
        spikes = self._encode_sequence(pixels)

        return spikes, label

    def _encode_sequence(self, pixels: torch.Tensor) -> torch.Tensor:
        """
        Encode pixel sequence to spike trains.

        Args:
            pixels: (784,) pixel values in [0, 1]

        Returns:
            spikes: (784, n_input_neurons) binary spikes
        """
        T = len(pixels)
        spikes = torch.zeros(T, self.n_input_neurons)

        prev_pixel = 0.0
        for t, pixel in enumerate(pixels):
            # Fire if pixel crosses threshold from below
            # Neuron i fires if: prev_pixel < threshold[i] <= pixel
            # OR if pixel <= threshold[i] < prev_pixel (for decreasing)
            crossed_up = (self.thresholds > prev_pixel) & (self.thresholds <= pixel)
            crossed_down = (self.thresholds <= prev_pixel) & (self.thresholds > pixel)
            spikes[t] = (crossed_up | crossed_down).float()
            prev_pixel = pixel

        return spikes


def get_dataloaders(config) -> tuple:
    """
    Get Sequential MNIST dataloaders with threshold encoding.

    Returns:
        train_loader, test_loader, input_dim, output_dim, T
    """
    # Default to 80 input neurons per paper
    n_input_neurons = getattr(config, "n_input_neurons", 80)
    batch_size = config.batch_size
    data_dir = config.data_dir

    # Create datasets
    train_dataset = ThresholdEncodedMNIST(
        root=data_dir,
        train=True,
        n_input_neurons=n_input_neurons,
        download=True,
    )

    test_dataset = ThresholdEncodedMNIST(
        root=data_dir,
        train=False,
        n_input_neurons=n_input_neurons,
        download=True,
    )

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

    return train_loader, test_loader, n_input_neurons, 10, 784
