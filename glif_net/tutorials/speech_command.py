"""Google Speech Commands tutorial with MFCC encoding."""

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader
import numpy as np


def get_task_defaults() -> dict:
    """Return default overrides for Speech Commands."""
    return {
        "T": 160,
        "dt": 1.0,
        "batch_size": 64,
        "lr": 1e-3,
        "n_neuron": 256,
        "n_mfcc": 20,
    }


class SpeechCommandsDataset:
    """Wrapper for SpeechCommands with MFCC and threshold encoding."""

    def __init__(
        self,
        dataset,
        n_mfcc: int = 20,
        time_bins: int = 160,
        n_input_neurons: int = 80,
    ):
        self.dataset = dataset
        self.n_mfcc = n_mfcc
        self.time_bins = time_bins
        self.n_input_neurons = n_input_neurons

        # MFCC transform
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=16000,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": 400,
                "hop_length": 160,
                "n_mels": 40,
            },
        )

        # Label mapping
        self.label_map = {
            "yes": 0,
            "no": 1,
            "up": 2,
            "down": 3,
            "left": 4,
            "right": 5,
            "on": 6,
            "off": 7,
            "stop": 8,
            "go": 9,
            "zero": 10,
            "one": 11,
            "two": 12,
            "three": 13,
            "four": 14,
            "five": 15,
            "six": 16,
            "seven": 17,
            "eight": 18,
            "nine": 19,
            "bed": 20,
            "bird": 21,
            "cat": 22,
            "dog": 23,
            "happy": 24,
            "house": 25,
            "marvin": 26,
            "sheila": 27,
            "tree": 28,
            "wow": 29,
            "_silence_": 30,
            "_unknown_": 31,
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        waveform, sample_rate, label, speaker_id, utterance_number = self.dataset[idx]

        # Compute MFCC
        mfcc = self.mfcc_transform(waveform)  # (1, n_mfcc, time)
        mfcc = mfcc.squeeze(0)  # (n_mfcc, time)

        # Normalize
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)

        # Convert to spike trains using threshold encoding
        spikes = self._mfcc_to_spikes(mfcc)

        # Map label
        label_idx = self.label_map.get(label, self.label_map["_unknown_"])

        return spikes, label_idx

    def _mfcc_to_spikes(self, mfcc: torch.Tensor) -> torch.Tensor:
        """
        Convert MFCC to spike trains.

        Args:
            mfcc: (n_mfcc, time) MFCC features

        Returns:
            spikes: (time_bins, n_input_neurons) spike tensor
        """
        n_mfcc, time_steps = mfcc.shape

        # Reshape to time_bins if needed
        if time_steps != self.time_bins:
            # Interpolate to fixed time bins
            mfcc = (
                nn.functional.interpolate(
                    mfcc.unsqueeze(0).unsqueeze(0),
                    size=(n_mfcc, self.time_bins),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .squeeze(0)
            )

        # Threshold encoding: divide MFCC coefficients into groups
        # and apply threshold crossing detection
        n_groups = self.n_input_neurons // self.n_mfcc
        thresholds = torch.linspace(-3, 3, n_groups)

        spikes = torch.zeros(self.time_bins, self.n_input_neurons)

        for t in range(self.time_bins):
            neuron_idx = 0
            for mf in range(self.n_mfcc):
                value = mfcc[mf, t].item()
                # Activate neurons based on which thresholds are crossed
                for th in thresholds:
                    if value > th:
                        if neuron_idx < self.n_input_neurons:
                            spikes[t, neuron_idx] = 1.0
                            neuron_idx += 1

        return spikes


def get_dataloaders(config) -> tuple:
    """
    Get Speech Commands dataloaders.

    Returns:
        train_loader, test_loader, input_dim, output_dim, T
    """
    time_bins = getattr(config, "T", 160)
    batch_size = config.batch_size
    data_dir = config.data_dir
    n_mfcc = getattr(config, "n_mfcc", 20)
    n_input_neurons = getattr(config, "n_input_neurons", 80)

    # Load datasets
    train_dataset_raw = torchaudio.datasets.SPEECHCOMMANDS(
        root=data_dir,
        url="speech_commands_v0.02",
        folder_in_archive="SpeechCommands",
        download=True,
        subset="training",
    )

    test_dataset_raw = torchaudio.datasets.SPEECHCOMMANDS(
        root=data_dir,
        url="speech_commands_v0.02",
        folder_in_archive="SpeechCommands",
        download=True,
        subset="testing",
    )

    # Wrap with preprocessing
    train_dataset = SpeechCommandsDataset(
        train_dataset_raw,
        n_mfcc=n_mfcc,
        time_bins=time_bins,
        n_input_neurons=n_input_neurons,
    )
    test_dataset = SpeechCommandsDataset(
        test_dataset_raw,
        n_mfcc=n_mfcc,
        time_bins=time_bins,
        n_input_neurons=n_input_neurons,
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

    # 35 classes: 30 words + silence + unknown
    return train_loader, test_loader, n_input_neurons, 35, time_bins
