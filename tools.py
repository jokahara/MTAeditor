from rdkit import Chem
import numpy as np

from filter import ClusterFilter
HC_DIST = 1.093 # H-C bond distance

def load_cluster_filter(file_in, mol_file='parameters.txt') -> ClusterFilter:
    cf = ClusterFilter(file_in, mol_file=mol_file)
    cf.set_Hbond('H', 'O', (1.4, 3.0), (100, 180))
    cf.set_Hbond('H', 'N', (1.4, 3.0), (100, 180))
    cf.set_Hbond('N', 'O', (2.4, 3.5), (70, 110))
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

def add_hydrogen_bonds(mol, donors, acceptors, parent: Chem.Mol=None):
    from topologger import add_Hbonds
    if parent is None:
        mol = add_Hbonds(mol, donors, acceptors, addId=True)
    else:
        from rdkit.Chem import RWMol, SanitizeMol, BondType
        mol = RWMol(mol) # make sure to copy the cluster

        idx = get_props(mol)
        # Copy hydrogen bond labels
        for d, a in zip(donors, acceptors):
            if d not in idx or a not in idx:
                continue
            
            label = parent.GetBondBetweenAtoms(int(d),int(a)).GetProp('bondNote')
            i1, i2 = get_idx(idx, [d,a])
            mol.AddBond(i1,i2, BondType.HYDROGEN)
            mol.GetBondBetweenAtoms(i1, i2).SetProp('bondNote', label+'*')

        SanitizeMol(mol)

    return mol


def get_bonds(mol):
    bonds = {(b.GetBeginAtom().GetIntProp('atomNote'), b.GetEndAtom().GetIntProp('atomNote')) for b in mol.GetBonds()
            if b.GetBeginAtom().HasProp('atomNote') and b.GetEndAtom().HasProp('atomNote')}
    return bonds

def get_props(mol):
    return [a.GetIntProp('atomNote') if a.HasProp('atomNote') else -1 for a in mol.GetAtoms()]

def get_idx(props, values):
    if isinstance(values, int):
        return props.index(values)
    
    return [props.index(a) for a in values]

def cut_molecule(mol, cut_at) -> Chem.Mol:
    try:
        if isinstance(cut_at[0], int):
            cut_at = [cut_at]

        mol = Chem.RWMol(mol)
        props = get_props(mol)
        cuts = [get_idx(props, c) for c in cut_at]
        for a, b in cuts:
            # set bond length to that of a mean H-C bond
            Chem.rdMolTransforms.SetBondLength(mol.GetConformer(0), a, b, HC_DIST)

        frags = Chem.FragmentOnBonds(mol, [mol.GetBondBetweenAtoms(a,b).GetIdx() for a,b in cuts])
        frags = Chem.GetMolFrags(frags, asMols=True)

        new_mol = frags[0] if cut_at[0][0] in get_props(frags[0]) else frags[1]
        return new_mol
    except: 
        return None

def rotate_bond(mol, j, k, l=None, deg=180):
    from rdkit.Chem import rdMolTransforms
    
    mol = Chem.RWMol(mol)
    bond = mol.GetBondBetweenAtoms(j, k)
    if bond.GetBondType() != Chem.BondType.SINGLE and bond.GetBondType() != Chem.BondType.DOUBLE:
        return
    if l is not None:
        bond2 = mol.GetBondBetweenAtoms(k, l)
        if bond2.GetBondType() != Chem.BondType.SINGLE and bond2.GetBondType() != Chem.BondType.DOUBLE:
            return
        
        #deg0 = rdMolTransforms.GetAngleDeg(mol.GetConformer(0), j, k, l)
        rdMolTransforms.SetAngleDeg(mol.GetConformer(0), j, k, l, deg)
        return mol 
    
    a = bond.GetBeginAtom()
    b = bond.GetEndAtom()
    if a.GetAtomicNum() == 1 or b.GetAtomicNum() == 1:
        return
    
    for bond in a.GetBonds():
        i = bond.GetBeginAtomIdx()
        if i != j and i != k:
            break
        i = bond.GetEndAtomIdx()
        if i != j and i != k:
            break
    for bond in b.GetBonds():
        l = bond.GetBeginAtomIdx()
        if l != j and l != k:
            break
        l = bond.GetEndAtomIdx()
        if l != j and l != k:
            break
    
    deg0 = rdMolTransforms.GetDihedralDeg(mol.GetConformer(0), i,j,k,l)
    rdMolTransforms.SetDihedralDeg(mol.GetConformer(0), i,j,k,l, deg0 + deg)

    return mol

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

def write_json(file, df):
    from ase import Atoms
    import json

    class Encoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.int64):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return list(obj)
            if isinstance(obj, Atoms):
                return obj.todict()
            return super().default(obj)
    
    molecules = np.unique(df['molecule'])
    data = {}
    for m in molecules:
        subset = df[df['molecule']==m]
        data[m] = {conf: {} for conf in subset['conformer']}
        for conf in subset['conformer']:
            bonds = subset[subset['conformer']==conf]
            for bond in np.unique(bonds['H-bond']):
                b = bonds[bonds['H-bond']==bond]
                data[m][conf][bond]= {
                    'energy': list(b['energy'].values), 'Hbond_pairs': b['Hbond_pairs'].values[0], 
                    'length': list(b['length'].values), 'angle': list(b['angle'].values), 
                    'fragments': list(b['fragments'].values)
                }

    with open(file, "w") as f:
        json.dump(data, f, cls=Encoder)


def find_rings(s):
    rings = {}
    for i, c in enumerate(s):
        if c.isnumeric() and s[i-1] != '#':
            if c in rings:
                rings[c].append(i)
            else: 
                rings[c] = [i]
    return rings

def find_brackets(s):
    stack = []
    brackets = []
    for i, c in enumerate(s):
        if c=="(":
            stack.append([c, i])
        elif c==")":
            _, idx = stack.pop()
            if i+1 == len(s):
                break
            brackets.append([idx, i])

    brackets = {k: v for k, v in brackets}
    return brackets

# recursively replace occurences of X with dummies
def add_dummies(patt: str, order=None):
    import re
    n = patt.count('X')
    X = ['C(-*)(-*)(-*)', 'C(-*)(=*)', 'c(:c)(:c)']
    size = [3, 2, 2]
    stripped = re.sub('[^a-zA-Z*]', '', patt)
    
    if order is None:
        order = list(range(len(stripped)-n))

    count = len(order)
    patterns = []; orders = []
    for x, s in zip(X, size):
        p = patt.replace('C-X', x, 1)
        start = stripped.find('X')
        ord = order.copy()
        for i in range(count, count+s):
            ord.insert(start, i)
            start+=1
            
        if n > 1:
            p, o = add_dummies(p, ord)
            patterns += p; orders += o
        else:
            patterns.append(p)
            orders.append(ord)

    return patterns, orders

def generate_mol_pattern(patt: str):
    patt = patt.replace('--', '~')
    # recursively replace occurences of X with dummies

    bonds="-=#:~"
    chars = list(patt)
    to_insert = {}

    rings = find_rings(chars)
    for s, e in rings.values():
        if chars[e-1] in bonds:
            to_insert[s] = chars[e-1]
        elif chars[s-1] in bonds:
            to_insert[e] = chars[s-1]
        elif chars[s-1].islower() and chars[e-1].islower():
            to_insert[e] = ':'
            to_insert[s] = ':'
        else:
            to_insert[e] = '-'
            to_insert[s] = '-'

    brackets = find_brackets(chars)
    i = 0
    while i < len(chars)-1:
        i+=1; e=i
        c1 = chars[i-1]
        c2 = chars[e]
        if not (c1 in '(*' or c1.isalpha()):
            continue
        while not (c2.isalpha() or c2 in '*)['):
            if c2 in ')[.':
                break
            if c2 == '(':
                e = brackets[e]
                c2 = chars[e+1]
                
            e+=1
            if e > len(chars)-1:
                break
            c2 = chars[e]
        if chars[e-1] in "-=#:~":
            continue

        if c1.islower() and c2.islower():
            to_insert[e]=':'
        elif c2.isalpha() or c2 in '*[':
            to_insert[e]='-'

    for k in sorted(to_insert.keys(), reverse=True):
        chars.insert(k, to_insert[k])
    smarts = ''.join(chars)

    if 'X' in smarts:
        smarts = smarts.replace('X-C', 'C-X')
        patterns, orders = add_dummies(smarts)
        orders = [sorted(range(len(ord)), key = lambda i: ord[i]) for ord in orders]
    else:
        patterns = [smarts]; orders = [None]

    mols = []
    for smarts, order in zip(patterns, orders):
        try:
            mol = Chem.MolFromSmarts(smarts)
            Chem.SanitizeMol(mol)
            if order is not None:
                mol = Chem.RenumberAtoms(mol, order)
            for atom in mol.GetAtoms():
                atom.SetIntProp('atomNote', atom.GetIdx())
            mols.append(mol)
        except:
            print('Failed with:', patt, '->', smarts)
    return mols

if __name__ == '__main__':
    from time import time
    mol = Chem.MolFromSmiles('O'+'CO'*100)
    mol = Chem.AddHs(mol)
    
    for a in mol.GetAtoms():
        a.SetIntProp('atomNote', a.GetIdx())
        
    t0 = time()
    props = get_props(mol)
    for i in range(10):
        for a in mol.GetAtoms():
            idx = get_idx(mol, range(400), props)

    print(time()-t0)