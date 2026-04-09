"""Poisson MNIST tutorial with rate-based encoding."""

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


def get_task_defaults() -> dict:
    """Return default overrides for Poisson MNIST."""
    return {
        "T": 100,
        "dt": 1.0,
        "batch_size": 64,
        "lr": 1e-3,
        "poisson_rate": 100.0,  # Max firing rate in Hz
        "n_neuron": 256,
    }


class PoissonMNIST(Dataset):
    """
    MNIST with Poisson rate encoding.

    Each pixel is converted to a Poisson spike train with rate
    proportional to pixel intensity.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        time_bins: int = 100,
        max_rate: float = 100.0,  # Hz
        dt: float = 1.0,  # ms
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

        self.time_bins = time_bins
        self.max_rate = max_rate
        self.dt = dt

    def __len__(self) -> int:
        return len(self.mnist)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        Returns:
            spikes: (T, 784) spike tensor
            label: class label
        """
        image, label = self.mnist[idx]

        # Flatten image: (1, 28, 28) -> (784,)
        pixels = image.view(-1)  # (784,)

        # Generate Poisson spikes
        spikes = self._poisson_encode(pixels)

        return spikes, label

    def _poisson_encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """
        Encode pixel intensities to Poisson spike trains.

        Args:
            pixels: (784,) pixel values in [0, 1]

        Returns:
            spikes: (time_bins, 784) binary spikes
        """
        # Firing probability per timestep
        # rate = pixel_value * max_rate (Hz)
        # prob = rate * dt / 1000 (convert ms to s)
        rates = pixels * self.max_rate  # Hz
        probs = rates * self.dt / 1000.0  # Probability per timestep

        # Generate spikes
        spikes = torch.rand(self.time_bins, 784) < probs.unsqueeze(0)

        return spikes.float()


def get_dataloaders(config) -> tuple:
    """
    Get Poisson MNIST dataloaders.

    Returns:
        train_loader, test_loader, input_dim, output_dim, T
    """
    time_bins = getattr(config, "T", 100)
    dt = getattr(config, "dt", 1.0)
    batch_size = config.batch_size
    data_dir = config.data_dir
    max_rate = getattr(config, "poisson_rate", 100.0)

    # Create datasets
    train_dataset = PoissonMNIST(
        root=data_dir,
        train=True,
        time_bins=time_bins,
        max_rate=max_rate,
        dt=dt,
        download=True,
    )

    test_dataset = PoissonMNIST(
        root=data_dir,
        train=False,
        time_bins=time_bins,
        max_rate=max_rate,
        dt=dt,
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

    return train_loader, test_loader, 784, 10, time_bins
