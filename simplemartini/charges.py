from __future__ import annotations

import numpy as np
from rdkit import Chem
from openff.toolkit import Molecule
from openff.units import unit

ASHGC_MODEL = "openff-gnn-am1bcc-1.0.0.pt"

def ashgc_charges_in_rdkit_order(rdmol: Chem.Mol) -> tuple[Chem.Mol, np.ndarray]:
    rdmol = Chem.Mol(rdmol)
    rdmol = Chem.AddHs(rdmol)

    offmol = Molecule.from_rdkit(
        rdmol,
        hydrogens_are_explicit=True,
        allow_undefined_stereo=True,
    )

    offmol.assign_partial_charges(
        partial_charge_method=ASHGC_MODEL,
        normalize_partial_charges=True,
    )

    off_charges = np.asarray(
        offmol.partial_charges.m_as(unit.elementary_charge),
        dtype=float,
    )

    n_atoms = rdmol.GetNumAtoms()
    if off_charges.shape != (n_atoms,):
        raise RuntimeError(
            f"OpenFF returned {off_charges.size} charges for {n_atoms} RDKit atoms."
        )

    # Molecule.from_rdkit preserves RDKit atom ordering.
    charges_rdkit_order = off_charges
    formal_charge = Chem.GetFormalCharge(rdmol)

    if not np.isclose(
        charges_rdkit_order.sum(),
        formal_charge,
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise RuntimeError(
            f"Charges sum to {charges_rdkit_order.sum():.10f}, "
            f"expected {formal_charge:+d}."
        )

    for rd_atom, charge in zip(rdmol.GetAtoms(), charges_rdkit_order, strict=True):
        rd_atom.SetDoubleProp("_PartialCharge", float(charge))

    return rdmol, charges_rdkit_order

def fold_hydrogen_charges_to_heavy_atoms(mol: Chem.Mol, charges) -> tuple[np.ndarray, list[int]]:

    """Return heavy-atom charges with directly bonded H charges summed in."""

    charges = np.asarray(charges, dtype=float)

    if charges.shape[0] != mol.GetNumAtoms():
        raise ValueError("Number of charges must match number of atoms in mol.")

    heavy_charges = []
    heavy_atom_indices = []

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue

        q = charges[atom.GetIdx()]
        for nbr in atom.GetNeighbors():
            if nbr.GetAtomicNum() == 1:
                q += charges[nbr.GetIdx()]
        heavy_atom_indices.append(atom.GetIdx())
        heavy_charges.append(q)

    return np.asarray(heavy_charges), heavy_atom_indices

def check_elements(mol):
    allowed_elements = ['C', 'O', 'H', 'N', 'S', 'F', 'Br', 'Cl', 'I', 'P']
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in allowed_elements:
            return False
    return True

def get_formal_charges(mol):
    charges = []
    for atom in mol.GetAtoms():
        charges.append(atom.GetFormalCharge())
    charges = np.array(charges)
    return charges

def get_heavy_atom_charges(mol_input):
    if check_elements(mol_input):
        mol, charges = ashgc_charges_in_rdkit_order(mol_input)
    else:
        mol = Chem.Mol(mol_input)
        charges = get_formal_charges(mol)
    charges_heavy, heavy_atom_indices = fold_hydrogen_charges_to_heavy_atoms(mol,charges)
    return mol, charges_heavy, heavy_atom_indices
