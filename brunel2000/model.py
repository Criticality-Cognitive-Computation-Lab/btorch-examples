"""Shared Brunel RSNN module wrapper."""

import torch.nn as nn

from btorch.models import functional
from btorch.models.init import uniform_v_
from btorch.models.rnn import RecurrentNN


class ModelRSNN(nn.Module):
    def __init__(self, neuron, synapse):
        super().__init__()
        self.neuron = neuron
        self.synapse = synapse
        self.rnn = RecurrentNN(
            neuron=neuron,
            synapse=synapse,
            step_mode="m",
            update_state_names=("neuron.v", "synapse.psc"),
        )

    def reset_state(self, batch_size: int = 1):
        functional.reset_net(self.rnn, batch_size=batch_size)
        uniform_v_(self.neuron, set_reset_value=True)
