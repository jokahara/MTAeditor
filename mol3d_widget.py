#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import py3Dmol
from rdkit import Chem

from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtWebEngineWidgets import QWebEngineView



class Mol3DWindow(QMainWindow):
    """A window that contains a single py3Dmol view."""
    def __init__(self):
        super().__init__()
        
        self.browser = QWebEngineView()
        self.setCentralWidget(self.browser)
        self.setFixedSize(660,500)


    def set_molecule(self, mol):
        view = py3Dmol.view(
            data=Chem.MolToMolBlock(mol),  # Convert the RDKit molecule for py3Dmol
            style={"stick": {}}
        )
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            view.addLabel(idx,
                {'position': {'x': mol.GetConformer(0).GetAtomPosition(idx).x,
                            'y': mol.GetConformer(0).GetAtomPosition(idx).y,
                            'z': mol.GetConformer(0).GetAtomPosition(idx).z},
                'backgroundColor': 'white',
                'fontColor': 'yellow',
                'fontSize': 18,
                'showBackground': False})
            
        view.setBackgroundColor('black')
        view.zoomTo()
        
        html = view._make_html()
        self.browser.setHtml(html)

        return 
    

if __name__ == "__main__":
    from rdkit.Chem import rdDepictor, rdDetermineBonds
    mol = Chem.MolFromXYZFile('1TolGECF-1_14.xyz')
    rdDetermineBonds.DetermineBonds(mol, allowChargedFragments=False)

    app = QApplication(sys.argv)
    window = Mol3DWindow()
    window.show()

    sys.exit(app.exec())
