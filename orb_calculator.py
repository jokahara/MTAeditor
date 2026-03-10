
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
        if mol == None:
            continue
        atoms = mol_to_atoms(mol, conf)
        atoms.info = {'charge': charge, 'spin': spin}
        atoms.calc = calculator

        el = atoms.get_potential_energy() / units.Ha
        results.append((el, atoms))

    return results
