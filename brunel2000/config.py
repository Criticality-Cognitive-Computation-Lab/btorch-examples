"""OmegaConf dataclass configuration for Brunel (2000) RSNN simulation."""

from dataclasses import dataclass, field
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from brunel2000.model import ModelRSNN


@dataclass
class ModelConfig:
    def build_model(self, dt_ms: float, device: str = "cpu") -> "ModelRSNN":
        raise NotImplementedError


@dataclass
class ModelAConfig(ModelConfig):
    """Model A: identical E/I neurons with fixed delay."""

    n_neurons: int = 12500
    n_e_ratio: float = 0.8
    c_e: int = 1000
    c_ext: int = 1000

    j: float = 0.1
    g: float = 5.0
    d_ms: float = 1.5

    tau_ms: float = 20.0
    tau_ref_ms: float = 2.0
    theta: float = 20.0
    v_reset: float = 0.0
    tau_syn_ms: float = 5.0
    nu_ext_hz: float = 20.0

    def build_model(self, dt_ms: float, device: str = "cpu") -> "ModelRSNN":
        from btorch.models import environ
        from btorch.models.linear import SparseConn
        from btorch.models.neurons import LIF
        from btorch.models.synapse import DelayedPSC, ExponentialPSC

        from brunel2000.connection import build_model_a_conn
        from brunel2000.model import ModelRSNN

        n_e = int(self.n_neurons * self.n_e_ratio)
        n_i = self.n_neurons - n_e
        c_i = int(self.c_e * (1.0 - self.n_e_ratio) / self.n_e_ratio)
        latency_steps = max(int(round(self.d_ms / dt_ms)), 1)

        with environ.context(dt=dt_ms):
            conn = build_model_a_conn(
                n_e=n_e,
                n_i=n_i,
                c_e=self.c_e,
                c_i=c_i,
                j=self.j,
                g=self.g,
                dt_ms=dt_ms,
                seed=42,
            )
            linear = SparseConn(conn, enforce_dale=False)
            neuron = LIF(
                n_neuron=self.n_neurons,
                tau=self.tau_ms,
                tau_ref=self.tau_ref_ms,
                v_threshold=self.theta,
                v_reset=self.v_reset,
                c_m=1.0,
            )
            base_psc = ExponentialPSC(
                n_neuron=self.n_neurons,
                tau_syn=self.tau_syn_ms,
                linear=linear,
            )
            psc = DelayedPSC(
                base_psc,
                max_delay_steps=latency_steps,
                use_circular_buffer=True,
            )

        model = ModelRSNN(neuron, psc)
        model.to(device)
        return model


@dataclass
class ModelBConfig(ModelConfig):
    """Model B: heterogeneous E/I parameters and delays."""

    n_neurons: int = 12500
    n_e_ratio: float = 0.8
    c_e: int = 1000
    c_ext: int = 1000

    j_e: float = 0.1
    j_i: float = 0.2
    g_e: float = 5.0
    g_i: float = 5.0

    tau_e_ms: float = 20.0
    tau_i_ms: float = 10.0
    tau_ref_ms: float = 2.0
    theta: float = 20.0
    v_reset: float = 0.0

    d_ee_ms: float = 4.0
    d_ei_ms: float = 4.0
    d_ie_ms: float = 4.0
    d_ii_ms: float = 4.0

    tau_syn_e_ms: float = 5.0
    tau_syn_i_ms: float = 5.0

    nu_e_ext_hz: float = 15.0
    nu_i_ext_hz: float = 15.0

    def build_model(self, dt_ms: float, device: str = "cpu") -> "ModelRSNN":
        import torch

        from btorch.models import environ
        from btorch.models.linear import SparseConn
        from btorch.models.neurons import LIF
        from btorch.models.synapse import ExponentialPSC, HeterSynapsePSC

        from brunel2000.connection import build_model_b_conn
        from brunel2000.model import ModelRSNN

        n_e = int(self.n_neurons * self.n_e_ratio)
        n_i = self.n_neurons - n_e
        c_i = int(self.c_e * (1.0 - self.n_e_ratio) / self.n_e_ratio)

        with environ.context(dt=dt_ms):
            conn_d, receptor_idx, _, n_delay_bins = build_model_b_conn(
                n_e=n_e,
                n_i=n_i,
                c_e=self.c_e,
                c_i=c_i,
                j_e=self.j_e,
                j_i=self.j_i,
                g_e=self.g_e,
                g_i=self.g_i,
                d_ee_ms=self.d_ee_ms,
                d_ei_ms=self.d_ei_ms,
                d_ie_ms=self.d_ie_ms,
                d_ii_ms=self.d_ii_ms,
                dt_ms=dt_ms,
                seed=42,
            )
            linear = SparseConn(conn_d, enforce_dale=False)
            n_receptor = len(receptor_idx)

            tau = torch.full((self.n_neurons,), self.tau_e_ms)
            tau[n_e:] = self.tau_i_ms
            v_threshold = torch.full((self.n_neurons,), self.theta)
            v_reset = torch.full((self.n_neurons,), self.v_reset)
            tau_ref = torch.full((self.n_neurons,), self.tau_ref_ms)
            c_m = torch.ones(self.n_neurons)

            neuron = LIF(
                n_neuron=self.n_neurons,
                tau=tau,
                tau_ref=tau_ref,
                v_threshold=v_threshold,
                v_reset=v_reset,
                c_m=c_m,
            )

            tau_syn_2d = torch.full((self.n_neurons, n_receptor), self.tau_syn_e_ms)
            post_types = receptor_idx["post_receptor_type"].values
            i_cols_i = [i for i, post_type in enumerate(post_types) if post_type == "I"]
            if i_cols_i:
                tau_syn_2d[:, i_cols_i] = self.tau_syn_i_ms
            tau_syn = tau_syn_2d.T.flatten()

            synapse = HeterSynapsePSC(
                n_neuron=self.n_neurons,
                n_receptor=n_receptor,
                receptor_type_index=receptor_idx,
                linear=linear,
                base_psc=ExponentialPSC,
                tau_syn=tau_syn,
                max_delay_steps=n_delay_bins,
                use_circular_buffer=True,
            )

        model = ModelRSNN(neuron, synapse)
        model.to(device)
        return model


@dataclass
class SimConfig:
    """Simulation parameters."""

    duration_ms: float = 500.0
    dt_ms: float = 0.1
    seed: int = 42
    regime: Literal["sr", "ai", "si_fast", "si_slow"] = "ai"
    device: str = "cpu"


@dataclass
class BrunelConfig:
    """Top-level configuration with model union."""

    model: ModelConfig = field(default_factory=ModelAConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    case_name: str | None = None

    @classmethod
    def default_from_case(cls, name: str) -> "BrunelConfig":
        """Return a config preset by name."""
        if not hasattr(cls, "_CASE_MAP") or cls._CASE_MAP is None:
            cls._CASE_MAP = {
                # Model A cases
                "model_a_sr": {
                    "model": ModelAConfig(g=3.0, nu_ext_hz=20.0),
                    "sim": SimConfig(regime="sr", dt_ms=0.1),
                },
                "model_a_ai": {
                    "model": ModelAConfig(g=5.0, nu_ext_hz=20.0),
                    "sim": SimConfig(regime="ai", dt_ms=0.1),
                },
                "model_a_si_fast": {
                    "model": ModelAConfig(g=6.0, nu_ext_hz=40.0),
                    "sim": SimConfig(regime="si_fast", dt_ms=0.1),
                },
                "model_a_si_slow": {
                    "model": ModelAConfig(g=4.5, nu_ext_hz=9.0),
                    "sim": SimConfig(regime="si_slow", dt_ms=0.1),
                },
                # Model B cases
                "model_b_sr": {
                    "model": ModelBConfig(
                        j_i=0.1, g_e=3.0, g_i=3.0, nu_e_ext_hz=20.0, nu_i_ext_hz=20.0
                    ),
                    "sim": SimConfig(regime="sr", dt_ms=0.1),
                },
                "model_b_ai": {
                    "model": ModelBConfig(
                        j_i=0.2, g_e=5.0, g_i=5.0, nu_e_ext_hz=15.0, nu_i_ext_hz=15.0
                    ),
                    "sim": SimConfig(regime="ai", dt_ms=0.1),
                },
                "model_b_si_fast": {
                    "model": ModelBConfig(
                        j_i=0.4, g_e=6.0, g_i=6.0, nu_e_ext_hz=30.0, nu_i_ext_hz=30.0
                    ),
                    "sim": SimConfig(regime="si_fast", dt_ms=0.1),
                },
                "model_b_si_slow": {
                    "model": ModelBConfig(
                        j_i=0.15, g_e=4.5, g_i=4.5, nu_e_ext_hz=10.0, nu_i_ext_hz=10.0
                    ),
                    "sim": SimConfig(regime="si_slow", dt_ms=0.1),
                },
            }

        if name not in cls._CASE_MAP:
            raise KeyError(
                f"Unknown case: {name}. Available: {list(cls._CASE_MAP.keys())}"
            )
        case = cls._CASE_MAP[name]
        cfg = cls(model=case["model"], sim=case["sim"], case_name=name)
        return cfg
