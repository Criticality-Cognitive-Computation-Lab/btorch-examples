import torch
import torch.nn as nn
from btorch.models import environ, functional, glif, rnn, synapse
from btorch.models.linear import DenseConn


class MinimalRSNN(nn.Module):
    def __init__(self, num_input, num_hidden, num_output, device="cpu"):
        super().__init__()

        # 1. Input projection
        self.fc_in = nn.Linear(num_input, num_hidden, bias=False, device=device)

        # 2. Spiking neuron
        neuron_module = glif.GLIF3(
            n_neuron=num_hidden,
            v_threshold=-45.0,
            v_reset=-60.0,
            c_m=2.0,
            tau=20.0,
            tau_ref=2.0,
            k=[0.1, 0.2],
            asc_amps=[1.0, -2.0],
            step_mode="s",  # single-step definition
            backend="torch",
            device=device,
        )

        # 3. Recurrent connection
        conn = DenseConn(num_hidden, num_hidden, bias=None, device=device)

        # 4. Synapse
        psc_module = synapse.AlphaPSC(
            n_neuron=num_hidden,
            tau_syn=5.0,
            linear=conn,
            step_mode="s",
        )

        # 5. Recurrent wrapper (multi-step)
        self.brain = rnn.RecurrentNN(
            neuron=neuron_module,
            synapse=psc_module,
            step_mode="m",
            update_state_names=("neuron.v", "synapse.psc"),
        )

        # 6. Output readout
        self.fc_out = nn.Linear(num_hidden, num_output, bias=False, device=device)

    def forward(self, x):
        x = self.fc_in(x)  # (T, Batch, num_input) -> (T, Batch, N)
        spike, states = self.brain(x)  # spike: (T, Batch, N)
        rate = spike.mean(dim=0)  # (Batch, N)
        out = self.fc_out(rate)  # (Batch, num_output)
        return out


# Keep dtype/device consistent across all params, buffers, states, and inputs
torch.set_default_dtype(torch.float32)
DEVICE = torch.device("cpu")
DTYPE = torch.float32

model = MinimalRSNN(num_input=20, num_hidden=64, num_output=5, device=str(DEVICE))
model = model.to(device=DEVICE, dtype=DTYPE)

# Initialize/reset states AFTER dtype/device are finalized
functional.init_net_state(model, batch_size=4, device=str(DEVICE))
functional.reset_net(model, batch_size=4)

# Some btorch/spikingjelly memory states are recreated from Python floats during
# reset and may become float64. Force all floating buffers back to desired dtype.
def _cast_all_floating_buffers_(module: nn.Module, dtype: torch.dtype, device: torch.device):
    for buffer_name, buffer in list(module.named_buffers()):
        if buffer is not None and torch.is_floating_point(buffer):
            casted = buffer.to(device=device, dtype=dtype)
            # buffer_name can be nested like "brain.synapse.psc"
            parts = buffer_name.split(".")
            target = module
            for p in parts[:-1]:
                target = getattr(target, p)
            setattr(target, parts[-1], casted)

_cast_all_floating_buffers_(model, DTYPE, DEVICE)

environ.set(dt=1.0)
inputs = torch.rand((100, 4, 20), device=DEVICE, dtype=DTYPE)  # (T, Batch, input_dim)

# Debug key dtypes involved in recurrent synapse path
print("inputs:", inputs.dtype)
print("fc_in.weight:", model.fc_in.weight.dtype)
print("synapse.linear.weight:", model.brain.synapse.linear.weight.dtype)
print("synapse.psc:", model.brain.synapse.psc.dtype)

with environ.context(dt=1.0):
    out = model(inputs)  # (Batch, num_output)

print("out:", out.shape, out.dtype)






####
inputs = torch.rand((100, 4, 64), device=DEVICE, dtype=DTYPE)  # (T, Batch, input_dim)
with environ.context(dt=1.0):
    spike, states = model.brain(inputs)

print(states["neuron.v"].shape)     # (T, Batch, N)
print(states["synapse.psc"].shape)  # (T, Batch, N)

from btorch.utils.dict_utils import unflatten_dict
nested = unflatten_dict(states, dot=True)
nested["neuron"]["v"][:, 0, :]   # voltage of batch 0