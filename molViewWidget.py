#!/usr/bin/python
# Import required modules
from __future__ import print_function
from PySide6 import QtCore, QtGui, QtWidgets, QtSvgWidgets
import sys
import copy

# from types import *
import logging

import numpy as np
from collections import defaultdict
from matplotlib.colors import hex2color

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry.rdGeometry import Point2D

# The Viewer Class
class MolWidget(QtSvgWidgets.QSvgWidget):

    fileDropped = QtCore.Signal(list)

    def __init__(self, mol=None, parent=None, moldrawoptions: rdMolDraw2D.MolDrawOptions = None):
        # Also init the super class
        super(MolWidget, self).__init__(parent)
        self.setAcceptDrops(True)

        # logging
        logging.basicConfig()
        self.logger = logging.getLogger()
        self.loglevel = logging.WARNING

        # This sets the window to delete itself when its closed, so it doesn't keep lingering in the background
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        # Private Properties
        self._mol = None  # The molecule
        self._drawmol = None  # Molecule for drawing
        self.drawer = None  # drawing object for producing SVG
        self._selectedAtoms = defaultdict(list)  # List of selected atoms
        self._darkmode = False
        self._flatten = False

        # Color options
        self.color_list = [hex2color("#8cd2f9"), hex2color("#df59b0"), hex2color("#9dc857")]

        # Draw options
        self._removeHs = True
        if moldrawoptions is None:
            self._moldrawoptions = rdMolDraw2D.MolDraw2DSVG(300, 300).drawOptions()
            self._moldrawoptions.prepareMolsBeforeDrawing = True
            self._moldrawoptions.addStereoAnnotation = True
            self._moldrawoptions.unspecifiedStereoIsUnknown = False
            self._moldrawoptions.fixedBondLength = 25
        else:
            self._moldrawoptions = moldrawoptions

        # Bind signales to slots for automatic actions
        self.molChanged.connect(self.sanitize_draw)
        self.selectionChanged.connect(self.draw)
        self.drawSettingsChanged.connect(self.draw)
        self.sanitizeSignal.connect(self.changeSanitizeStatus)

        # Initialize class with the mol passed
        self.mol = mol
        
    def resizeEvent(self, event: QtGui.QResizeEvent):
        #super().resizeEvent(event)
        w = event.size().width() 
        h = event.size().height()
        if w < h:
            h = w
        else:
            w = h
        self.resize(w, h)

        return 
    
    # Drag and drop events
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(QtCore.Qt.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(QtCore.Qt.CopyAction)
            event.accept()
            links = []
            for url in event.mimeData().urls():
                links.append(str(url.toLocalFile()))
            self.fileDropped.emit(links)
        else:
            event.ignore()

    ##Properties and their wrappers
    @property
    def loglevel(self):
        return self.logger.level

    @loglevel.setter
    def loglevel(self, loglvl):
        self.logger.setLevel(loglvl)

    @property
    def darkmode(self):
        return self._darkmode

    @darkmode.setter
    def darkmode(self, value: bool):
        self._darkmode = bool(value)
        self.draw()

    @property
    def moldrawoptions(self):
        """Returns the current drawing options.
        If settings aremanipulated directly, a drawSettingsChanged signal is not emitted,
        consider using setDrawOption instead."""
        return self._moldrawoptions

    @moldrawoptions.setter
    def moldrawoptions(self, value):
        self._moldrawoptions = value
        self.drawSettingsChanged.emit()

    def getDrawOption(self, attribute):
        return getattr(self._moldrawoptions, attribute)

    def setDrawOption(self, attribute, value):
        setattr(self._moldrawoptions, attribute, value)
        self.drawSettingsChanged.emit()

    # Getter and setter for mol
    molChanged = QtCore.Signal(name="molChanged")

    @property
    def mol(self):
        return self._mol

    @mol.setter
    def mol(self, mol):
        if mol is None:
            mol = Chem.MolFromSmiles("")
        if mol != self._mol:
            assert isinstance(mol, Chem.Mol)
            if self._mol is not None:
                self._prevmol = copy.deepcopy(self._mol)  # Chem.Mol(self._mol.ToBinary())  # Copy

            # Fix pseudo atoms
            atom: Chem.Atom
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    if not atom.HasProp("dummyLabel") or atom.GetProp("dummyLabel") == "*":
                        atom.SetProp("dummyLabel", "R")
                    #else:
                    #    print(atom.GetPropsAsDict())

            self._mol = mol
            self.molChanged.emit()

    def setMol(self, mol):
        self.mol = mol

    # Handling of selections
    selectionChanged = QtCore.Signal(name="selectionChanged")


    def getProp(self, mol, i):
        if i is None:
            return
        a = mol.GetAtomWithIdx(i)
        return a.GetIntProp('atomNote') if a.HasProp('atomNote') else ''
    
    def getProps(self, mol):
        return [a.GetIntProp('atomNote') for a in mol.GetAtoms() 
                if a.HasProp('atomNote')]

    def getIdx(self, mol: Chem.Mol, values):
        props = self.getProps(mol)
        if isinstance(values, (int, np.int64)):
            return props.index(values)
    
        return [props.index(a) for a in values if a in props]
    
    def selectAtom(self, atomidx, cut=0):
        if cut > 0:
            if self.mol.GetAtomWithIdx(atomidx).GetSymbol() == 'H':
                # TODO: implement a whole monomer removal
                return
            
            if len(self._selectedAtoms[cut]) > 0 and self._selectedAtoms[cut][-1][-1] is None:
                prev = self._selectedAtoms[cut][-1][0]
                b = self.mol.GetBondBetweenAtoms(prev, atomidx)
                if b is None:
                    self._selectedAtoms[cut].pop()
                    return
                self._selectedAtoms[cut][-1][-1] = self._selectedAtoms[cut][-1][0]
                self._selectedAtoms[cut][-1][0] = atomidx
                self.selectionChanged.emit()
            else:
                self._selectedAtoms[cut].append([atomidx, None])
                self.selectionChanged.emit()
        elif atomidx not in self._selectedAtoms[cut]:    
            self._selectedAtoms[0].append(atomidx)
            self.selectionChanged.emit()

    def unselectAtom(self, atomidx, cut=0):
        if atomidx in self._selectedAtoms[0]:  
            self._selectedAtoms[0].remove(atomidx)
            if cut > 0:
                self.selectAtom(atomidx, cut)
            self.selectionChanged.emit()
            return
        
        if cut == 0 or cut == 1:
            for pair in self._selectedAtoms[1]:
                if atomidx in pair:
                    self._selectedAtoms[1].remove(pair)
                    self.selectionChanged.emit()
                    return
        if cut == 0 or cut == 2:
            for pair in self._selectedAtoms[2]:
                if atomidx in pair:
                    self._selectedAtoms[2].remove(pair)
                    self.selectionChanged.emit()
                    return
        
        if cut > 0:
            self.selectAtom(atomidx, cut)

    def clearAtomSelection(self):
        if sum(len(x) for x in self._selectedAtoms.values()) > 0:
            self._selectedAtoms.clear()
            self.selectionChanged.emit()

    def GetSelectedCuts(self):
        cut1 = [[self.getProp(self.mol, a), self.getProp(self.mol, b)] for a,b in self._selectedAtoms[1]]
        cut2 = [[self.getProp(self.mol, a), self.getProp(self.mol, b)] for a,b in self._selectedAtoms[2]]
        return cut1, cut2

    @property
    def selectedAtoms(self):
        if self._drawmol == None:
            return []
        
        selected = self._selectedAtoms[0].copy()
        for s in self._selectedAtoms[1]: 
            selected += s
        for s in self._selectedAtoms[2]: 
            selected += s
        return selected


    @property
    def colors(self):
        if len(self.selectedAtoms) == 0:
            return {}, {}
        
        try:
            selectedAtoms = {0: [self.getProp(self.mol, a) for a in self._selectedAtoms[0]],
                            1: [[self.getProp(self.mol, a), self.getProp(self.mol, b)] for a,b in self._selectedAtoms[1]],
                            2: [[self.getProp(self.mol, a), self.getProp(self.mol, b)] for a,b in self._selectedAtoms[2]]}
            atom_cols = defaultdict(list)
            bond_cols = defaultdict(list)
            atom_cols.update({self.getIdx(self._drawmol, a): [self.color_list[0]] for a in selectedAtoms[0]})
            for bond in self._drawmol.GetBonds():
                a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                a, b = self.getProp(self._drawmol, a), self.getProp(self._drawmol, b)
                if (a in selectedAtoms[0]) and (b in selectedAtoms[0]):
                    bond_cols[bond.GetIdx()].append(self.color_list[0])

            for i in [1,2]:
                for a, b in selectedAtoms[i]: 
                    idx1 = self.getIdx(self._drawmol, a)
                    atom_cols[idx1].append(self.color_list[i])
                    if b is not None:
                        idx2 = self.getIdx(self._drawmol, b)
                        atom_cols[idx2].append(self.color_list[i])
                        bond = self._drawmol.GetBondBetweenAtoms(idx1, idx2)
                        if bond is not None:
                            bond_cols[bond.GetIdx()].append(self.color_list[i])

        except:
            atom_cols = {}
            bond_cols = {}
        

        return dict(atom_cols), dict(bond_cols)

    @property
    def radii(self):
        try:
            atom_radii = {self.getIdx(self._drawmol, self.getProp(self.mol, a)): 0.4 for a in self._selectedAtoms[0]}
            for i in [1, 2]:
                for a, b in self._selectedAtoms[i]: 
                    if b is None:
                        atom_radii[self.getIdx(self._drawmol, self.getProp(self.mol, a))] = 0.25
                    else:
                        atom_radii[self.getIdx(self._drawmol, self.getProp(self.mol, a))] = 0.4
                        atom_radii[self.getIdx(self._drawmol, self.getProp(self.mol, b))] = 0.25
        except:
            atom_radii = {}

        return atom_radii

    drawSettingsChanged = QtCore.Signal(name="drawSettingsChanged")

    # Actions and functions
    @QtCore.Slot()
    def draw(self):
        #self.logger.debug("Updating SVG")
        svg = self.getMolSvg()
        self.load(QtCore.QByteArray(svg.encode("utf-8")))

    @QtCore.Slot()
    def sanitize_draw(self):
        # self.computeNewCoords()
        self.sanitizeDrawMol()
        self.draw()

    @QtCore.Slot()
    def changeSanitizeStatus(self, value):
        #self.logger.debug(f"changeBorder called with value {value}")
        if value.upper() == "SANITIZABLE":
            self.molecule_sanitizable = True
        else:
            self.molecule_sanitizable = False

    def computeNewCoords(self, mol, ignoreExisting=False, canonOrient=False):
        """Computes new coordinates for the molecule taking into account all
        existing positions (feeding these to the rdkit coordinate generation as
        prev_coords).
        """
        # This code is buggy when you are not using the CoordGen coordinate
        # generation system, so we enable it here
        rdDepictor.SetPreferCoordGen(True)
        prev_coords = {}
        if mol.GetNumConformers() == 0:
            self.logger.debug("No Conformers found, computing all 2D coords")
        elif ignoreExisting:
            self.logger.debug("Ignoring existing conformers, computing all 2D coords")
        else:
            assert mol.GetNumConformers() == 1
            self.logger.debug(f"{self._mol.GetNumConformers()} conformer(s) found, computing 2D coords not in found conformer")
            conf = mol.GetConformer(0)
            for a in mol.GetAtoms():
                pos3d = conf.GetAtomPosition(a.GetIdx())
                if (pos3d.x, pos3d.y) == (0, 0):
                    continue
                prev_coords[a.GetIdx()] = Point2D(pos3d.x, pos3d.y)
        #self.logger.debug("Coordmap %s" % prev_coords)
        self.logger.debug("canonOrient %s" % canonOrient)
        rdDepictor.Compute2DCoords(mol, coordMap=prev_coords, canonOrient=canonOrient)

    def canon_coords_and_draw(self):
        self.logger.debug("Recalculating coordinates")
        self._drawmol = copy.deepcopy(self._mol)  # Chem.Mol(self._mol.ToBinary())
        self.computeNewCoords(canonOrient=True, ignoreExisting=True)
        self.draw()

    def updateStereo(self):
        #self.logger.debug("Updating stereo info")
        for atom in self.mol.GetAtoms():
            if atom.HasProp("_CIPCode"):
                atom.ClearProp("_CIPCode")
        for bond in self.mol.GetBonds():
            if bond.HasProp("_CIPCode"):
                bond.ClearProp("_CIPCode")
        Chem.rdmolops.SetDoubleBondNeighborDirections(self.mol)
        self.mol.UpdatePropertyCache(strict=False)
        Chem.rdCIPLabeler.AssignCIPLabels(self.mol)

    sanitizeSignal = QtCore.Signal(str, name="sanitizeSignal")

    @QtCore.Slot()
    def sanitizeDrawMol(self, kekulize=False, drawkekulize=False):

        self.updateStereo()
        self._drawmol_test = copy.deepcopy(self._mol) 
        self._drawmol = copy.deepcopy(self._mol)
        
        try:
            Chem.SanitizeMol(self._drawmol_test)
            self.sanitizeSignal.emit("Sanitizable")
        except Exception as e:
            self.sanitizeSignal.emit("UNSANITIZABLE")
            self.logger.warning("Unsanitizable")

        if self._flatten:
            self.computeNewCoords(self._drawmol)
            
        self._drawmol = rdMolDraw2D.PrepareMolForDrawing(self._drawmol, kekulize=False)

    finishedDrawing = QtCore.Signal(name="finishedDrawing")

    def getMolSvg(self):
        self.drawer = rdMolDraw2D.MolDraw2DSVG(300, 300)
        
        if self._drawmol is not None:
            opts = self.drawer.drawOptions()
            opts.dummiesAreAttachments = True
            if self._darkmode:
                rdMolDraw2D.SetDarkMode(opts)
            if (not self.molecule_sanitizable) and self.unsanitizable_background_colour:
                opts.setBackgroundColour(self.unsanitizable_background_colour)
                
            if self._removeHs:
                self._drawmol = Chem.RemoveHs(self._drawmol)

            Chem.AssignStereochemistryFrom3D(self._drawmol)   
            
            atom_colors, bond_colors = self.colors
            self.drawer.DrawMoleculeWithHighlights(
                self._drawmol, "", atom_colors, bond_colors, self.radii, {}
            )
                
        self.drawer.FinishDrawing()
        self.finishedDrawing.emit()  # Signal that drawer has finished
        svg = self.drawer.GetDrawingText().replace("svg:", "")
        
        return svg


if __name__ == "__main__":
    #    model = SDmodel()
    #    model.loadSDfile('dhfr_3d.sd')
    mol = Chem.MolFromSmiles("CCN(C)c1ccccc1S")
    # rdDepictor.Compute2DCoords(mol)
    myApp = QtWidgets.QApplication(sys.argv)
    molview = MolWidget(mol)
    molview.selectAtom(1)
    molview.selectedAtoms = [1, 2, 3]
    molview.show()
    myApp.exec()
