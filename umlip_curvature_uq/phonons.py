"""Model force constants on exactly the same displaced supercells as the DFT reference."""

from __future__ import annotations

import numpy as np
import phonopy
from ase import Atoms
from ase.calculators.calculator import Calculator
from phonopy.structure.atoms import PhonopyAtoms


def to_ase(cell: PhonopyAtoms) -> Atoms:
    return Atoms(
        symbols=cell.symbols,
        cell=cell.cell,
        scaled_positions=cell.scaled_positions,
        pbc=True,
    )


def _without_forces(dataset: dict) -> dict:
    """Displacement dataset (type 1 or type 2) with the DFT forces removed."""
    if "first_atoms" in dataset:
        return {
            "natom": dataset["natom"],
            "first_atoms": [
                {"number": d["number"], "displacement": d["displacement"]}
                for d in dataset["first_atoms"]
            ],
        }
    return {"displacements": np.array(dataset["displacements"])}


def _forces(atoms: Atoms, calc: Calculator) -> np.ndarray:
    atoms = atoms.copy()
    atoms.calc = calc
    return np.asarray(atoms.get_forces(), dtype=float)


def model_phonopy(reference: phonopy.Phonopy, calc: Calculator) -> phonopy.Phonopy:
    """Copy of the reference setup with forces from `calc`.

    The structure is the DFT-relaxed one and is not relaxed again with the
    model, so curvature is compared at the same geometry. Residual model forces
    on the undisplaced supercell are subtracted from every displaced set.
    """
    ph = phonopy.Phonopy(
        reference.unitcell,
        supercell_matrix=reference.supercell_matrix,
        primitive_matrix=reference.primitive_matrix,
        log_level=0,
    )
    ph.dataset = _without_forces(reference.dataset)
    f0 = _forces(to_ase(ph.supercell), calc)
    ph.forces = np.array([_forces(to_ase(s), calc) - f0 for s in ph.supercells_with_displacements])
    ph.produce_force_constants()
    return ph


def mesh_frequencies(ph: phonopy.Phonopy, mesh=(6, 6, 6)) -> np.ndarray:
    """Frequencies in THz on a Gamma-centred mesh, shape (n_q, n_bands).

    Imaginary modes are returned as negative numbers (phonopy convention).
    """
    ph.run_mesh(list(mesh), is_gamma_center=True, with_eigenvectors=False)
    return np.asarray(ph.mesh.frequencies, dtype=float)
