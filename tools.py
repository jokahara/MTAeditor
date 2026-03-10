from rdkit import Chem
from ase import Atoms
import numpy as np

from filter import ClusterFilter

def load_cluster_filter(file_in, mol_file='parameters.txt') -> ClusterFilter:
    cf = ClusterFilter(file_in, mol_file=mol_file)
    
    return cf

def generate_mol(cluster_filter: ClusterFilter, ct: str) -> Chem.Mol:
    from rdkit.Chem.AllChem import ETKDGv3, EmbedMolecule
    from topologger import generate_rdkit_cluster

    cluster_info = cluster_filter.cluster_info(ct)
    mol = generate_rdkit_cluster(cluster_info.smiles)
    if mol.GetNumConformers() == 0:
        params = ETKDGv3()
        EmbedMolecule(mol, params)
        
    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        atom.SetIntProp("atomNote", atom.GetIdx())
            
    return mol

def add_hydrogen_bonds(mol, donors, acceptors):
    from topologger import add_Hbonds
    mol = add_Hbonds(mol, donors, acceptors, addId=True)
    
    return mol

HC_DIST = 1.093 # H-C bond distance

def get_bonds(mol):
    bonds = {(b.GetBeginAtom().GetIntProp('atomNote'), b.GetEndAtom().GetIntProp('atomNote')) for b in mol.GetBonds()
            if b.GetBeginAtom().HasProp('atomNote') and b.GetEndAtom().HasProp('atomNote')}
    return bonds

def get_props(mol):
    return [a.GetIntProp('atomNote') for a in mol.GetAtoms() 
            if a.HasProp('atomNote')]

def get_idx(mol: Chem.Mol, values):
    props = get_props(mol)
    if isinstance(values, int):
        return props.index(values)
    
    return [props.index(a) for a in values]

def cut_molecule(mol, cut_at) -> Chem.Mol:
    if isinstance(cut_at[0], int):
        cut_at = [cut_at]

    mol = Chem.RWMol(mol)
    cuts = [get_idx(mol, c) for c in cut_at]
    for a, b in cuts:
        Chem.rdMolTransforms.SetBondLength(mol.GetConformer(0), a, b, HC_DIST)

    frags = Chem.FragmentOnBonds(mol, [mol.GetBondBetweenAtoms(a,b).GetIdx() for a,b in cuts])
    frags = Chem.GetMolFrags(frags, asMols=True)

    new_mol = frags[0] if cut_at[0][0] in get_props(frags[0]) else frags[1]
    return new_mol

def rot_ar_x(radi):
    return  np.array([[1, 0, 0, 0],
                    [0, np.cos(radi), -np.sin(radi), 0],
                    [0, np.sin(radi), np.cos(radi), 0],
                    [0, 0, 0, 1]], dtype=np.double)

def rot_ar_y(radi):
    return  np.array([[np.cos(radi), 0, np.sin(radi), 0],
                    [0, 1, 0, 0],
                    [-np.sin(radi), 0, np.cos(radi), 0],
                    [0, 0, 0, 1]], dtype=np.double)

def rot_ar_z(radi):
    return  np.array([[np.cos(radi), -np.sin(radi), 0, 0],
                    [np.sin(radi), np.cos(radi), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]], dtype=np.double)

tforms = {0: rot_ar_x, 1: rot_ar_y, 2: rot_ar_z}

def rotate_mol(mol, radii, axis):
    radii = np.pi*radii/180
    Chem.rdMolTransforms.TransformConformer(mol.GetConformer(0), tforms[axis](radii))
