from __future__ import annotations

import pandas as pd
import torch

from btorch.models.synapse import HeterSynapsePSC as MainHeterSynapsePSC


class HeterSynapsePSC(MainHeterSynapsePSC):
    """Compatibility wrapper for btorch main-branch HeterSynapsePSC.

    Adds optional `receptor_is_exc` support (used in mice branch) and keeps
    `psc_e` / `psc_i` states available without changing main simulation logic.
    """

    def __init__(
        self,
        n_neuron: int,
        n_receptor: int,
        receptor_type_index: pd.DataFrame,
        linear: torch.nn.Module,
        base_psc,
        step_mode="s",
        backend="torch",
        receptor_is_exc: torch.Tensor | None = None,
        **kwargs,
    ):
        super().__init__(
            n_neuron=n_neuron,
            n_receptor=n_receptor,
            receptor_type_index=receptor_type_index,
            linear=linear,
            base_psc=base_psc,
            step_mode=step_mode,
            backend=backend,
            **kwargs,
        )

        if receptor_is_exc is None:
            receptor_is_exc = torch.zeros(n_receptor, dtype=torch.bool)
            if {
                "pre_receptor_type",
                "post_receptor_type",
            }.issubset(self.receptor_type_index.columns):
                receptor_is_exc = torch.as_tensor(
                    self.receptor_type_index.sort_values("receptor_index")[
                        "pre_receptor_type"
                    ].astype(str)
                    == "E",
                    dtype=torch.bool,
                )
            else:
                receptor_is_exc[: (n_receptor // 2)] = True

        self.register_buffer("receptor_is_exc", receptor_is_exc.bool(), persistent=False)
        self.register_memory("psc_e", 0.0, self.n_neuron)
        self.register_memory("psc_i", 0.0, self.n_neuron)

    def single_step_forward(self, z: torch.Tensor):
        psc = super().single_step_forward(z)

        psc_all = self.base_psc.psc.view(*self.base_psc.psc.shape[:-1], *self.n_neuron, self.n_receptor)

        if self.receptor_is_exc.any():
            self.psc_e = psc_all[..., self.receptor_is_exc].sum(-1)
            self.psc_i = psc_all[..., ~self.receptor_is_exc].sum(-1)
        else:
            self.psc_e = torch.zeros_like(psc)
            self.psc_i = psc.clone()

        return psc
