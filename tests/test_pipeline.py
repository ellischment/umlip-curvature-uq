import numpy as np
import phonopy
import pytest
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.emt import EMT
from phonopy.structure.atoms import PhonopyAtoms

from umlip_curvature_uq import metrics, phonons


class ScaledEMT(Calculator):
    """EMT with forces multiplied by a constant: a uniformly softened potential."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.inner = EMT()

    def calculate(self, atoms=None, properties=("forces",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        a = atoms.copy()
        a.calc = self.inner
        self.results = {"energy": self.scale * a.get_potential_energy(), "forces": self.scale * a.get_forces()}


@pytest.fixture(scope="module")
def reference():
    cu = bulk("Cu", "fcc", a=3.6, cubic=True)
    unit = PhonopyAtoms(symbols=cu.get_chemical_symbols(), cell=cu.cell[:], scaled_positions=cu.get_scaled_positions())
    ph = phonopy.Phonopy(unit, supercell_matrix=np.eye(3) * 2, log_level=0)
    ph.generate_displacements(distance=0.01)
    calc = EMT()
    f0 = phonons._forces(phonons.to_ase(ph.supercell), calc)
    ph.forces = [phonons._forces(phonons.to_ase(s), calc) - f0 for s in ph.supercells_with_displacements]
    ph.produce_force_constants()
    return ph


def test_same_potential_reproduces_reference(reference):
    ref = phonons.mesh_frequencies(reference, (4, 4, 4))
    pred = phonons.mesh_frequencies(phonons.model_phonopy(reference, EMT()), (4, 4, 4))
    np.testing.assert_allclose(pred, ref, atol=1e-6)
    assert metrics.slope_through_origin(pred, ref) == pytest.approx(1.0, abs=1e-6)


def test_scaled_forces_give_sqrt_scaled_frequencies(reference):
    ref = phonons.mesh_frequencies(reference, (4, 4, 4))
    pred = phonons.mesh_frequencies(phonons.model_phonopy(reference, ScaledEMT(0.81)), (4, 4, 4))
    b = metrics.model_bias(pred, ref)
    assert b.slope == pytest.approx(0.9, abs=1e-6)
    assert b.mean_rel_error == pytest.approx(-0.1, abs=1e-6)
    assert b.n_imaginary == 0


def test_common_mode_bias_is_invisible_to_spread(reference):
    ref = phonons.mesh_frequencies(reference, (4, 4, 4))
    soft = [phonons.mesh_frequencies(phonons.model_phonopy(reference, ScaledEMT(0.81)), (4, 4, 4))] * 3
    s = metrics.ensemble_stats(soft, ref)
    assert s.spread == pytest.approx(0.0, abs=1e-9)
    assert s.bias_sq == pytest.approx(0.01, abs=1e-6)
    assert metrics.prediction_spread(soft) == pytest.approx(0.0, abs=1e-9)
