from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D


def _bead_sigma(bead_type):
    if bead_type.startswith('T'):
        return 0.34
    if bead_type.startswith('S'):
        return 0.41
    return 0.47


def _bead_color(bead_type, charge):
    if 'Q' in bead_type:
        return 'blue' if charge > 0 else 'red'
    if 'C' in bead_type:
        return 'gray'
    if 'N' in bead_type:
        return 'forestgreen'
    if 'P' in bead_type:
        return 'purple'
    if 'X' in bead_type:
        return 'brown'
    return 'black'


def draw_mapping_overlay(mol, beads, bead_types, bead_charges, output_path):
    """Draw an RDKit structure with its coarse-grained beads overlaid."""

    if not (len(beads) == len(bead_types) == len(bead_charges)):
        raise ValueError('Beads, bead types, and bead charges must have equal lengths.')

    mol_2d = Chem.RemoveHs(Chem.Mol(mol))
    rdMolDraw2D.PrepareMolForDrawing(mol_2d)
    Chem.rdDepictor.Compute2DCoords(mol_2d)

    width, height = 600, 400
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.DrawMolecule(mol_2d)
    drawer.FinishDrawing()

    atom_positions = np.asarray([
        [drawer.GetDrawCoords(idx).x, drawer.GetDrawCoords(idx).y]
        for idx in range(mol_2d.GetNumAtoms())
    ])
    bead_positions = np.asarray([
        atom_positions[bead].mean(axis=0)
        for bead in beads
    ])

    bond_lengths = [
        np.linalg.norm(atom_positions[bond.GetBeginAtomIdx()] - atom_positions[bond.GetEndAtomIdx()])
        for bond in mol_2d.GetBonds()
    ]
    nm_to_pixels = (np.mean(bond_lengths) / 1.5 * 10.0 * 0.9) if bond_lengths else 50.0
    bead_radii = np.asarray([_bead_sigma(bead_type) * nm_to_pixels / 2.0 for bead_type in bead_types])
    padding = float(bead_radii.max() * 1.3) if len(bead_radii) else 0.0

    namespace = 'http://www.w3.org/2000/svg'
    ET.register_namespace('', namespace)
    root = ET.fromstring(drawer.GetDrawingText())
    root.set('width', str(width + 2 * padding))
    root.set('height', str(height + 2 * padding))
    root.set('viewBox', f'{-padding} {-padding} {width + 2 * padding} {height + 2 * padding}')

    font_size = 16 if len(beads) > 8 else 20
    for idx, (center, radius, bead_type, charge) in enumerate(
        zip(bead_positions, bead_radii, bead_types, bead_charges)
    ):
        color = _bead_color(bead_type, charge)
        cx, cy = center

        circle = ET.SubElement(root, f'{{{namespace}}}circle')
        circle.set('cx', f'{cx:.2f}')
        circle.set('cy', f'{cy:.2f}')
        circle.set('r', f'{radius:.2f}')
        circle.set('fill', color)
        circle.set('fill-opacity', '0.3')
        circle.set('stroke', color)
        circle.set('stroke-width', '2')
        circle.set('stroke-dasharray', '6,3')

        label = ET.SubElement(root, f'{{{namespace}}}text')
        label.set('x', f'{cx:.2f}')
        label.set('y', f'{cy - radius * 1.15:.2f}')
        label.set('text-anchor', 'middle')
        label.set('dominant-baseline', 'middle')
        label.set('font-size', str(font_size))
        label.set('font-weight', 'bold')
        label.set('font-family', 'IBM Plex Sans, sans-serif')
        label.set('fill', color)
        label.text = f'[{idx + 1}] {bead_type}'

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(ET.tostring(root, encoding='unicode'))
