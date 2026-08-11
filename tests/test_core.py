import numpy as np

from simplemartini.core import coarse_grain_charges


def test_coarse_grain_charges_uses_atom_indices():
    charges = coarse_grain_charges(
        beads=[[4, 1], [7]],
        charges_heavy=[0.2, -0.1, 0.9],
        heavy_atom_indices=[1, 4, 7],
    )

    np.testing.assert_allclose(charges, [0.1, 0.9])
