import os
import re
import tempfile

import MDAnalysis as mda
import numpy as np

from cgparam import CGParam

def load_itp(path,name):
    if not os.path.isfile(f'{path}/{name}.itp'):
        print(f'{path}/{name}.itp not found.')
        # continue
    with open(f'{path}/{name}.itp','r') as f_in:
        return f_in.readlines()

def parse_input(lines,name,qtype):
    # Parse itp input lines
    section = None
    # atoms = []
    lines_moleculetype = []
    lines_atoms = []

    bonds = []
    constraints = []
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
            start = re.search('\[',line).span()[0]
            end = re.search('\]',line).span()[0]
            section = line[start+1:end].replace(' ','')
            # print(start, end)
            # print(section)
            continue
        if re.match(r'#ifdef',line):
            section = 'ifdef'
        if section == 'moleculetype':
            line = re.sub('MOL',name,line)
            lines_moleculetype.append(line)
        elif section == 'atoms':
            line = re.sub('Qx',qtype,line)
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
    # print(spl)
    idxs = [int(spl[0]),int(spl[1])]
    length = float(spl[3])
    k = float(spl[4])
    return [idxs, length, k]

def read_constraint(spl,k=20000.):
    # print(spl)
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
    print(dihedrals)
    flagged_idxs = []
    for idx, d0 in enumerate(dihedrals[:-1]):
        print(d0)
        d0 = np.array(d0[:-1]) # without angle
        for jdx, d1 in enumerate(dihedrals[idx+1:],start=idx+1):
            print(d1)
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
        # print(di)
        for dj in dihedral[idx+1:-1]:
            # print(di, dj)
            found = False # already present in bonds
            for bond in bonds:
                bond_idxs = bond[0]
                # print(bond_idxs)
                if (di in bond_idxs) and (dj in bond_idxs):
                    found = True
                    # print('found')
                    break
            if not found:
                xi = u.atoms[di-1].position / 10.
                xj = u.atoms[dj-1].position / 10.
                dist = np.linalg.norm(xj - xi)
                dij = [di,dj]
                newbond = [[min(dij),max(dij)], dist, k]
                print(f'Adding bond {newbond}')
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
    # print(vsite)
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

def repartition_masses(vsite,u,scale=1.):
    vsite_idx = vsite[0] # 1-based
    mass_vsite = 0.
    n = len(vsite)
    # print(u.atoms.masses)
    for idx, w in vsite[1:]: # 1-based
        m = u.atoms[idx-1].mass # 0-based
        m_partition = m / n * scale
        mass_vsite += m_partition
        u.atoms[idx-1].mass = m - m_partition
    u.atoms[vsite_idx-1].mass = mass_vsite
    # print(u.atoms.masses)
    return u

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

def simplify(name,path_in,path_out,qtype):
    u = mda.Universe(f'{path_in}/{name}.itp',f'{path_in}/{name}.gro')
    lines = load_itp(path_in,name)
    lines_moleculetype, lines_atoms, lines_angles, dihedrals, bonds, vsites = parse_input(lines,name,qtype)

    for vsite in vsites:
        bonds = add_vsite_bonds(vsite,u,bonds)
        u = repartition_masses(vsite,u)
        # dihedrals = add_dihedral(dihedrals,vsite)
        # dihedrals.append(dihedral)

    flagged_dihedrals, kept_dihedrals = analyse_dihedrals(dihedrals)
    print(f'Replace: {flagged_dihedrals}')
    print(f'Keep: {kept_dihedrals}')

    for dihedral in flagged_dihedrals:
        bonds = repl_dihedral(dihedral,u,bonds)

    lines_atoms = make_atomlines(u)
    lines_bonds = make_bondlines(bonds)

    lines_dihedrals = make_dihedrallines(kept_dihedrals) # lines_dihedrals

    os.makedirs(path_out,exist_ok=True)

    fname = f'{path_out}/{name}.itp'
    write_itp(fname,lines_moleculetype, lines_atoms, lines_bonds, lines_angles, lines_dihedrals)

    if path_in != path_out:
        os.system(f'cp {path_in}/{name}.gro {path_out}/')

def run_simplemartini(name, mol, qtype = 'Qx', path_out = 'output'):
    with tempfile.TemporaryDirectory() as tmpdir:
        cgp = CGParam()
        cgp.run_pipeline(name, mol, path_out = tmpdir) # mol_martini = ...
    simplify(name,tmpdir,path_out,qtype) # read in mol_martini, return an object