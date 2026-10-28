"""ASE calculators for the universal potentials compared here.

Imports are lazy: each package is only needed for the models you run.
"""

from __future__ import annotations

from typing import Callable

from ase.calculators.calculator import Calculator


def _mace_mp0(device: str) -> Calculator:
    from mace.calculators import mace_mp

    return mace_mp(model="medium", default_dtype="float64", device=device)


def _chgnet(device: str) -> Calculator:
    from chgnet.model.dynamics import CHGNetCalculator

    return CHGNetCalculator(use_device=device)


def _mattersim(device: str) -> Calculator:
    from mattersim.forcefield import MatterSimCalculator

    return MatterSimCalculator(load_path="MatterSim-v1.0.0-1M.pth", device=device)


def _orb(device: str) -> Calculator:
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.inference.calculator import ORBCalculator

    model, adapter = pretrained.orb_v3_conservative_inf_omat(
        device=device, precision="float32-highest", compile=False
    )
    return ORBCalculator(model, adapter, device=device)


REGISTRY: dict[str, Callable[[str], Calculator]] = {
    "mace-mp-0": _mace_mp0,
    "chgnet": _chgnet,
    "mattersim-v1": _mattersim,
    "orb-v3": _orb,
}


def get_calculator(name: str, device: str = "cpu") -> Calculator:
    try:
        return REGISTRY[name](device)
    except KeyError:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(REGISTRY)}") from None
