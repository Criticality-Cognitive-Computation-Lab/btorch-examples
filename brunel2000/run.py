"""CLI entry point for Brunel (2000) RSNN simulation."""

import sys
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf

from btorch.models import environ, functional

from brunel2000.config import BrunelConfig
from brunel2000.simulate import run_simulation
from brunel2000.analysis import analyze, plot_results


def load_config() -> BrunelConfig:
    defaults = OmegaConf.structured(BrunelConfig())
    cli_cfg = OmegaConf.from_cli()

    if "case_name" in cli_cfg and cli_cfg.case_name is not None:
        case_name = cli_cfg.case_name
        base = OmegaConf.structured(BrunelConfig.default_from_case(case_name))
        # Remove case_name from cli_cfg so it doesn't conflict when re-merging
        cli_without_case = OmegaConf.create(
            {k: v for k, v in cli_cfg.items() if k != "case_name"}
        )
        cfg = OmegaConf.unsafe_merge(base, cli_without_case)
    else:
        cfg = OmegaConf.unsafe_merge(defaults, cli_cfg)

    return OmegaConf.to_object(cfg)


def main():
    config = load_config()

    model_type = config.model._type_  # type: ignore
    regime = config.sim.regime
    device = torch.device(config.sim.device if torch.cuda.is_available() else "cpu")
    dt = config.sim.dt_ms

    print(f"Model: {model_type}")
    print(f"Regime: {regime}")
    print(f"Device: {device}")
    print(f"dt: {dt} ms")
    print(f"Duration: {config.sim.duration_ms} ms")

    torch.manual_seed(config.sim.seed)

    with environ.context(dt=dt):
        model = config.model.build_model(dt_ms=dt, device=str(device))

    functional.init_net_state(model.rnn, batch_size=1, device=str(device))
    model.reset_state(batch_size=1)

    result = run_simulation(model, config, seed=config.sim.seed)

    stats = analyze(result, model_type)
    print("\n--- Results ---")
    print(f"Mean rate (overall): {stats['mean_rate_overall_hz']:.2f} Hz")
    print(f"Mean rate (E):       {stats['mean_rate_e_hz']:.2f} Hz")
    print(f"Mean rate (I):       {stats['mean_rate_i_hz']:.2f} Hz")
    print(f"CV of ISI:           {stats['cv_isi']:.3f}")
    if model_type == "ModelBConfig":
        print(f"E/I phase lag:       {stats['ei_phase_lag_ms']:.2f} ms")
        print(f"E/I phase lag:       {stats['ei_phase_lag_deg']:.2f} deg")
    print(f"Dominant freq:       {stats['dominant_freq_hz']:.2f} Hz")

    outdir = Path("outputs") / f"{regime}_{model_type}"
    plot_results(result, model_type, regime, outdir)

    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "config.yaml", "w") as f:
        yaml.safe_dump(OmegaConf.to_container(OmegaConf.structured(config)), f)

    print(f"\nOutputs saved to {outdir}")


if __name__ == "__main__":
    main()
