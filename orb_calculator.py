
import numpy as np
import ase
from ase import Atoms
#from ase.io import read, write
import ase.units as units
E = units.Ha / (units.kcal/units.mol)

import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_calculator():
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.calculator import ORBCalculator

    device = 'cpu'
    model = pretrained.orb_v3_conservative_omol(
        device=device,
        precision="float64",   # or "float32-high(est)"
    )
    calculator = ORBCalculator(model, device=device)

    # test run to complete loading
    from ase import build
    atoms = build.molecule('CH4')
    atoms.calc = calculator
    atoms.info = {'charge': 0, 'spin': 1}
    el = atoms.get_potential_energy() / units.Ha

    return calculator

def mol_to_atoms(mol, conf=0):
    symbols = []
    for a in mol.GetAtoms():
        sym = a.GetSymbol()
        symbols.append('H' if (sym == '*') or (sym == 'R') else sym)

    positions = mol.GetConformer(conf).GetPositions()
    atoms = Atoms(symbols, positions)
    return atoms

def get_single_point_energies(mols, calculator, conf=0, charge=0, spin=1):
    results = []
    for mol in mols:
        if mol is None:
            continue
        atoms = mol_to_atoms(mol, conf)
        atoms.info = {'charge': charge, 'spin': spin}
        atoms.calc = calculator

        el = atoms.get_potential_energy() / units.Ha
        results.append((el, atoms))
        atoms.calc = None

    return results


def get_rotational_energies(mol, bond_idx, calculator, conf=0, charge=0, spin=1):
    from scipy.signal import find_peaks
    from tools import rotate_bond

    if mol is None:
        return

    i, j = bond_idx
    angles = np.arange(360)
    results = np.zeros(360)
    idx = []
    def calculate(deg):
        if results[deg] != 0:
            return
        
        mol2 = rotate_bond(mol, i, j, deg)
        atoms = mol_to_atoms(mol2, conf)
        atoms.info = {'charge': charge, 'spin': spin}
        atoms.calc = calculator

        el = atoms.get_potential_energy() / units.Ha
        results[deg] = el
    
        atoms.calc = None

    start = 0
    end = 360
    step = 30
    n = 1
    print('Scanning:')

    while n < 8:
        print('Round', n)
        if n > 2:
            idx = results<0
            y = results[idx]
            peaks,_ = find_peaks(y)
            valleys,_ = find_peaks(-y)
            x = np.append(peaks, valleys)
            x = x[y[x]-y[0] < 0.035]
            #print(angles[idx][x], (y[x]-y[0])*627)
            for p in angles[idx][x]:
                if p < 105 or p > 255:
                    continue
                start = p-2*step 
                end = p+2*step+1
                #print(start, end, step)
                for deg in range(start, end, step):
                    calculate(deg)
        else:
            for deg in range(start, end, step):
                calculate(deg)
            start=105
            end=270

        step = int(np.ceil(step/2))
        n+=1
    
    idx = results<0    
    results = results[idx]
    angles = angles[idx]
    peaks,_ = find_peaks(results)
    valleys,_ = find_peaks(-results)
    return results, angles, np.append(peaks, valleys)

"""def get_optimized_energies(mols, calculator, conf=0, charge=0, spin=1):
    from ase.optimize import BFGS
    results = []
    for mol in mols:
        if mol is None:
            continue
        atoms = mol_to_atoms(mol, conf)
        atoms.info = {'charge': charge, 'spin': spin}
        atoms.calc = calculator

        opt = BFGS(atoms)
        opt.run(0.01)

        el = atoms.get_potential_energy() / units.Ha
        results.append((el, atoms))
        atoms.calc = None

    return results"""