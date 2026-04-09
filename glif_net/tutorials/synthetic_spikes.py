"""Fast synthetic spike pattern classification task for smoke testing."""

import torch
from torch.utils.data import Dataset, DataLoader

from btorch.datasets.noise import poisson_noise, ou_noise


def get_task_defaults() -> dict:
    """Return default overrides for synthetic spike task."""
    return {
        "T": 50,
        "dt": 1.0,
        "batch_size": 128,
        "lr": 1e-3,
        "n_neuron": 64,
        "epochs": 20,
        "n_input_neurons": 20,
    }


class SyntheticSpikeDataset(Dataset):
    """
    Synthetic spike pattern classification using btorch noise generators.

    Generates random spike trains where each class has a distinct temporal
    pattern. No dataset download required - perfect for quick testing.

    Classes:
        0: Low uniform firing rate (poisson_noise, rate=5 Hz)
        1: High uniform firing rate (poisson_noise, rate=25 Hz)
        2: Early burst (high rate in first half, low in second)
        3: Late burst (low rate in first half, high in second)
    """

    def __init__(
        self,
        n_samples: int = 1000,
        time_bins: int = 50,
        n_input_neurons: int = 20,
        seed: int = 42,
    ):
        self.n_samples = n_samples
        self.time_bins = time_bins
        self.n_input_neurons = n_input_neurons
        self.base_seed = seed

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        Returns:
            spikes: (T, n_input_neurons) binary spike tensor
            label: class label (0-3)
        """
        g = torch.Generator().manual_seed(self.base_seed + idx)
        label = torch.randint(0, 4, (1,), generator=g).item()

        if label == 0:
            # Low uniform rate using btorch poisson_noise
            counts = poisson_noise(
                self.n_input_neurons,
                rate=5.0,
                T=self.time_bins,
                dt=1.0,
                generator=g,
            )
        elif label == 1:
            # High uniform rate using btorch poisson_noise
            counts = poisson_noise(
                self.n_input_neurons,
                rate=25.0,
                T=self.time_bins,
                dt=1.0,
                generator=g,
            )
        elif label == 2:
            # Early burst: high rate first half, low second half
            rate_profile = torch.zeros(self.time_bins, 1)
            rate_profile[: self.time_bins // 2] = 30.0
            rate_profile[self.time_bins // 2 :] = 2.0
            counts = poisson_noise(
                self.n_input_neurons,
                rate=rate_profile,
                T=self.time_bins,
                dt=1.0,
                generator=g,
            )
        else:
            # Late burst: low rate first half, high second half
            rate_profile = torch.zeros(self.time_bins, 1)
            rate_profile[: self.time_bins // 2] = 2.0
            rate_profile[self.time_bins // 2 :] = 30.0
            counts = poisson_noise(
                self.n_input_neurons,
                rate=rate_profile,
                T=self.time_bins,
                dt=1.0,
                generator=g,
            )

        # Convert counts to binary spikes (clip at 1)
        spikes = (counts > 0).float()

        return spikes, label


def get_dataloaders(config) -> tuple:
    """
    Get synthetic spike classification dataloaders.

    Returns:
        train_loader, test_loader, input_dim, output_dim, T
    """
    time_bins = getattr(config, "T", 50)
    batch_size = config.batch_size
    n_input_neurons = getattr(config, "n_input_neurons", 20)

    train_dataset = SyntheticSpikeDataset(
        n_samples=2000,
        time_bins=time_bins,
        n_input_neurons=n_input_neurons,
        seed=42,
    )

    test_dataset = SyntheticSpikeDataset(
        n_samples=500,
        time_bins=time_bins,
        n_input_neurons=n_input_neurons,
        seed=12345,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, test_loader, n_input_neurons, 4, time_bins
