import ast
import os
import re
import shutil
from pathlib import Path

import warnings

import numpy as np
from rdkit import Chem

from cgparam.core import CGParam

from .visualization import draw_mapping_overlay

warnings.filterwarnings(
    "ignore",
    message=r"'xdrlib' is deprecated and slated for removal in Python 3\.13",
    category=DeprecationWarning,
    module=r"MDAnalysis\.topology\.TPRParser",
)

import MDAnalysis as mda  # noqa: E402

def load_itp(path,name):
    if not os.path.isfile(f'{path}/{name}.itp'):
        print(f'{path}/{name}.itp not found.')
        # continue
    with open(f'{path}/{name}.itp','r') as f_in:
        return f_in.readlines()

def parse_input(lines,name):
    # Parse itp input lines
    section = None
    # atoms = []
    lines_moleculetype = []
    lines_atoms = []

    bonds = []
    dihedrals = []
    lines_angles = []
    # lines_dihedrals = []
    vsites = []
    # exclusions = []

    for line in lines:
        # check for new header
        if line[0] == ';' or len(line) == 1:
            continue
        # if len(re.findall('\[',line)) > 0:
        if line[0] == '[':
            start = re.search(r'\[',line).span()[0]
            end = re.search(r'\]',line).span()[0]
            section = line[start+1:end].replace(' ','')

            continue
        if re.match(r'#ifdef',line):
            section = 'ifdef'
        if section == 'moleculetype':
            line = re.sub('MOL',name,line)
            lines_moleculetype.append(line)
        elif section == 'atoms':
            lines_atoms.append(line)
        elif section == 'angles':
            lines_angles.append(line)
        elif re.match(r'#endif',line):
            continue
        # elif section == 'dihedrals':
        #     lines_dihedrals.append(line)
        else:
            spl = line.split()
            if len(spl) > 0:
                spl = [s.replace(' ','') for s in spl]
                if section == 'bonds':
                    bond = read_bond(spl)
                    bonds.append(bond)
                elif section == 'constraints': # convert to bonds
                    constr = read_constraint(spl)
                    bonds.append(constr)
                elif section == 'dihedrals':
                    dihedral = read_dihedral(spl)
                    dihedrals.append(dihedral)
                elif section == 'virtual_sitesn':
                    vsite = read_vsite(spl)
                    vsites.append(vsite)
    return lines_moleculetype, lines_atoms, lines_angles, dihedrals, bonds, vsites

def read_vsite(spl):
    vsite = [int(spl[0])]
    if int(spl[1]) == 3:
        for jdx in range(2,len(spl),2):
            vsite.append([int(spl[jdx]), float(spl[jdx+1])])
    return vsite 

def read_bond(spl):

    idxs = [int(spl[0]),int(spl[1])]
    length = float(spl[3])
    k = float(spl[4])
    return [idxs, length, k]

def read_constraint(spl,k=20000.):

    idxs = [int(spl[0]),int(spl[1])]
    length = float(spl[3])
    return [idxs, length, k]

def read_dihedral(spl):
    if spl[4] == '2':
        dihedral = [int(x) for x in spl[:4]]
        dihedral.append(float(spl[5]))
        return dihedral
    else:
        raise

def analyse_dihedrals(dihedrals):

    flagged_idxs = []
    for idx, d0 in enumerate(dihedrals[:-1]):
        d0 = np.array(d0[:-1]) # without angle
        for jdx, d1 in enumerate(dihedrals[idx+1:],start=idx+1):
            d1 = np.array(d1[:-1]) # without angle
            inters = np.intersect1d(d0,d1)
            if len(inters) >= 3:
                if jdx not in flagged_idxs:
                    flagged_idxs.append(jdx)

    flagged_dihedrals = []
    kept_dihedrals = []
    for jdx, dihedral in enumerate(dihedrals):
        if jdx in flagged_idxs:
            flagged_dihedrals.append(dihedral)
        else:
            kept_dihedrals.append(dihedral)
    return flagged_dihedrals, kept_dihedrals

def repl_dihedral(dihedral,u,bonds,k=20000.):
    for idx, di in enumerate(dihedral[:-2]):
        for dj in dihedral[idx+1:-1]:
            found = False # already present in bonds
            for bond in bonds:
                bond_idxs = bond[0]
                if (di in bond_idxs) and (dj in bond_idxs):
                    found = True
                    break
            if not found:
                xi = u.atoms[di-1].position / 10.
                xj = u.atoms[dj-1].position / 10.
                dist = np.linalg.norm(xj - xi)
                dij = [di,dj]
                newbond = [[min(dij),max(dij)], dist, k]
                bonds.append(newbond)
    return bonds

# def calc_ABC(a,b,c):
#     alpha = np.arccos((b**2+c**2-a**2) / (2.*b*c))
#     print(f'alpha: {alpha}')
#     A = np.array([0.,0.])
#     B = np.array([c,0.])
#     C = np.array([b*np.cos(alpha),b*np.sin(alpha)])
#     return(A,B,C)

def add_vsite_bonds(vsite,u,bonds,k=20000.):
    xs = []
    weights = []
    other_indices = []
    vsite_idx = vsite[0] # 1-based
    for idx, w in vsite[1:]:
        x = u.atoms[idx-1].position / 10.
        xs.append(x)
        weights.append(w)
        other_indices.append(idx) # 1-based
    xs = np.array(xs)
    weights = np.array(weights)

    x_vsite = np.average(xs,axis=0,weights=weights)

    for idx, x in zip(other_indices,xs):
        dist = np.linalg.norm(x - x_vsite)
        if idx < vsite_idx:
            a, b = idx, vsite_idx
        else:
            a, b = vsite_idx, idx
        bonds.append([[a, b], dist, k])
    return bonds

def make_bondlines(bonds):
    lines_bonds = []
    for bond in bonds:
        line = f'    {bond[0][0]}  {bond[0][1]}    1     {bond[1]:.3f}    {bond[2]:.1f}\n'
        lines_bonds.append(line)
    return lines_bonds

def make_atomlines(u):
    lines_atoms = []
    for idx, at in enumerate(u.atoms):
        line = f'{idx+1:>5d}{at.type:>5s}    1{at.resname:>5s}{at.name:>5s}{idx+1:>5d}     {at.charge:.3f}   {at.mass:.3f}\n'
        lines_atoms.append(line)
    return lines_atoms

# def add_dihedral(dihedrals,vsite,angle=0.,k=100.):
#     other_indices = []
    
#     for idx, w in vsite[1:]:
#         other_indices.append(idx)
#     for perm in itertools.combinations(other_indices,3):
#         dihedral = [vsite[0]] # 1-based
#         for idx in perm:
#             dihedral.append(idx)
#         dihedral.append(angle)
#         dihedral.append(k)
#         dihedrals.append(dihedral)
#     return dihedrals

    # for idx, w in vsite[1:4]: # only use vsite + up to 3 beads for improper
    #     dihedral.append(idx)
    # dihedral.append(angle)
    # dihedral.append(k)
    # # print(dihedral)
    # return dihedral

def make_dihedrallines(dihedrals,k_dihedral=500.):
    lines_dihedrals = []
    for dih in dihedrals:
        line = f'    {dih[0]}  {dih[1]}  {dih[2]}  {dih[3]}    2    {dih[4]:.3f}    {k_dihedral:.1f}\n'
        lines_dihedrals.append(line)
    return lines_dihedrals

def write_itp(fname,lines_moleculetype, lines_atoms, lines_bonds, lines_angles, lines_dihedrals):
    with open(fname,'w') as f:
        write_section(f,'moleculetype',lines_moleculetype)
        write_section(f,'atoms',lines_atoms)
        write_section(f,'bonds',lines_bonds)
        write_section(f,'angles',lines_angles)
        write_section(f,'dihedrals',lines_dihedrals)

def write_section(f,header,lines):
    f.write(f'[{header}]\n')
    for line in lines:
        f.write(line)
    f.write('\n')

# def assign_ashgc_charges(u,charges_ashgc):
    # for 

def simplify(name,path_in,path_out,qs_cg = [],masses_cg = []):

    u = mda.Universe(f'{path_in}/{name}.itp',f'{path_in}/{name}.gro')

    if len(qs_cg) > 0:
        u.atoms.charges = qs_cg
    if len(masses_cg) > 0:
        u.atoms.masses = masses_cg

    lines = load_itp(path_in,name)
    lines_moleculetype, lines_atoms, lines_angles, dihedrals, bonds, vsites = parse_input(lines,name)

    for vsite in vsites:
        bonds = add_vsite_bonds(vsite,u,bonds)
        # dihedrals = add_dihedral(dihedrals,vsite)
        # dihedrals.append(dihedral)

    flagged_dihedrals, kept_dihedrals = analyse_dihedrals(dihedrals)
    # print(f'Replace: {flagged_dihedrals}')
    # print(f'Keep: {kept_dihedrals}')

    for dihedral in flagged_dihedrals:
        bonds = repl_dihedral(dihedral,u,bonds)

    lines_atoms = make_atomlines(u)
    lines_bonds = make_bondlines(bonds)

    lines_dihedrals = make_dihedrallines(kept_dihedrals) # lines_dihedrals

    os.makedirs(path_out,exist_ok=True)

    fname = f'{path_out}/{name}.itp'
    write_itp(fname,lines_moleculetype, lines_atoms, lines_bonds, lines_angles, lines_dihedrals)

    if Path(path_in) != Path(path_out):
        shutil.copy2(Path(path_in) / f'{name}.gro', path_out)

def coarse_grain_charges(beads, charges_heavy, heavy_atom_indices=None):
    if heavy_atom_indices is None:
        heavy_atom_indices = range(len(charges_heavy))
    charge_by_atom = dict(zip(heavy_atom_indices, charges_heavy))

    qs_cg = []
    for bead in beads:
        try:
            qs_cg.append(sum(charge_by_atom[at_idx] for at_idx in bead))
        except KeyError as error:
            raise ValueError(
                f'Bead mapping contains atom {error.args[0]}, which has no heavy-atom charge.'
            ) from error
    return np.array(qs_cg)

def coarse_grain_masses(beads,mol_h):
    masses_cg = []
    for bead in beads:
        mass = 0.
        for at_idx in bead:
            atom = mol_h.GetAtomWithIdx(at_idx)
            mass += atom.GetMass()
            for neighbor in atom.GetNeighbors():
                if neighbor.GetAtomicNum() == 1:
                    mass += neighbor.GetMass()
        masses_cg.append(mass)
    return np.array(masses_cg)


def _run_martini_mapper(name, mol, path_mapping, run_xtb, nthreads):
    """Run Martini Mapper and return bead indices in the input molecule's order."""

    from martini_mapper.main import run_mapping
    from martini_mapper.outputs import fix_beadtypes, group_beads_by_type

    if any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms()):
        raise ValueError('Martini Mapper integration expects a molecule with implicit hydrogens.')

    # Martini Mapper reparses SMILES, so translate its atom indices back to the
    # input molecule. RDKit records this permutation whenever it writes SMILES.
    smiles = Chem.MolToSmiles(mol, canonical=False)
    smiles_atom_order = ast.literal_eval(mol.GetProp('_smilesAtomOutputOrder'))

    final, _ = run_mapping(
        name,
        smiles,
        run_xtb=run_xtb,
        write_files=True,
        out_dir=Path(path_mapping),
        dihedrals=False,
        nthreads=nthreads,
    )
    beads_smiles_order, raw_bead_types = group_beads_by_type(final)
    beads = [
        [smiles_atom_order[atom_idx] for atom_idx in bead]
        for bead in beads_smiles_order
    ]

    mapped_atoms = sorted(atom_idx for bead in beads for atom_idx in bead)
    expected_atoms = list(range(mol.GetNumAtoms()))
    if mapped_atoms != expected_atoms:
        raise RuntimeError('Martini Mapper did not map every input heavy atom exactly once.')

    return beads, fix_beadtypes(raw_bead_types), Chem.AddHs(Chem.Mol(mol))

def run_simplemartini(
        name,
        mol,
        path_out = None, #  'output',
        calc_charges = True,
        mapping_backend = 'cgparam',
        path_mapping = None,
        martini_mapper_run_xtb = True,
        nthreads = 1,
        draw_overlay = True,
    ):

    if path_mapping is None:
        path_mapping = f'{mapping_backend}_tmp'
    if path_out is None:
        path_out = f'{mapping_backend}_out'

    print(Chem.MolToSmiles(mol))

    if mapping_backend == 'cgparam':
        print('Using cgparam backend')
        cgp = CGParam()
        cgp.run_pipeline(name, mol, path_out=path_mapping)
        beads = cgp.beads
        bead_types = cgp.bead_types
        mol_h = cgp.mol_h
        default_charges = np.asarray(cgp.charges)
    elif mapping_backend == 'martini_mapper':
        print('Using martini_mapper backend')
        beads, bead_types, mol_h = _run_martini_mapper(
            name,
            mol,
            path_mapping,
            run_xtb=martini_mapper_run_xtb,
            nthreads=nthreads,
        )
        default_charges = np.zeros(len(beads))
    else:
        raise ValueError(
            "mapping_backend must be either 'cgparam' or 'martini_mapper'."
        )

    masses_cg = coarse_grain_masses(beads, mol_h)

    if calc_charges:
        from .charges import get_heavy_atom_charges

        mol_with_charges, charges_heavy, heavy_atom_indices = get_heavy_atom_charges(mol)
        qs_cg = coarse_grain_charges(beads, charges_heavy, heavy_atom_indices)
    else:
        mol_with_charges = Chem.Mol(mol)
        qs_cg = np.array([])

    simplify(name, path_mapping, path_out, qs_cg=qs_cg, masses_cg=masses_cg)

    if draw_overlay:
        overlay_charges = qs_cg if len(qs_cg) else default_charges
        draw_mapping_overlay(
            mol_with_charges,
            beads,
            bead_types,
            overlay_charges,
            Path(path_out) / f'{name}_overlay.svg',
        )
