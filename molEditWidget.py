#!/usr/bin/python
# Import required modules
from PySide6 import QtCore, QtGui, QtWidgets

from PySide6.QtCore import Qt
from PySide6.QtGui import QMouseEvent, QPainter, QPen

import sys
import logging
from warnings import warn
import copy

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Geometry.rdGeometry import Point2D, Point3D

from molViewWidget import MolWidget

# The Molblock editor class
class MolEditWidget(MolWidget):
    def __init__(self, mol=None, parent=None):
        # Also init the super class
        super(MolEditWidget, self).__init__(parent)
        # This sets the window to delete itself when its closed, so it doesn't keep querying the model
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.is_dragging = False  # If a drag event is being performed

        self.sanitize_on_cleanup = True
        self.kekulize_on_cleanup = True

        # Properties
        self._prevmol = None  # For undo
        self.coordlist = None  # SVG coords of the current mols atoms

        # Standard atom, bond and ring types
        #self.symboltoint = symboltoint
        self.bondtypes = Chem.rdchem.BondType.names  # A dictionary with all available rdkit bondtypes

        # Default actions
        self._action = "Select"

        # Points to calculate the SVG to coord scaling
        self.points = [Point2D(0, 0), Point2D(1, 1)]

        # Bind signals to slots
        self.finishedDrawing.connect(self.update_coordlist)  # When drawing finished, update coordlist of SVG atoms.

        # Init with a mol if passed at construction
        # if mol != None:
        self.mol = mol

    # Getters and Setters for properties
    actionChanged = QtCore.Signal(name="actionChanged")

    @property
    def action(self):
        return self._action

    @action.setter  # TODO make it more explicit what actions are available here.
    def action(self, actionname):
        if actionname != self.action:
            self._action = actionname
            self.actionChanged.emit()

    def setAction(self, actionname):
        self.action = actionname

    def rotateMol(self, actionname):
        self.molChanged.emit()

    def setAtom(self, atomtype):
        self.logger.debug("Setting atomtype selection to %s" % atomtype)
        if atomtype in self.symboltoint.keys():
            self.logger.debug("Atomtype found in keys")
            # self.atomtype = self.symboltoint[atomtype]
            self._chementitytype = "atom"
            self._chementity = self.symboltoint[atomtype]
        elif isinstance(atomtype, int):
            if atomtype in self.symboltoint.values():
                self._chementitytype = "atom"
                self._chementity = atomtype
            else:
                self.logger.error(f"Atom number {atomtype} not known.")
        else:
            self.logger.error("Atomtype must be string or integer, not %s" % type(atomtype))

    # Function to translate from SVG coords to atom coords using scaling calculated from atomcoords (0,0) and (1,1)
    # Returns rdkit Point2D
    def SVG_to_coord(self, x_svg, y_svg):
        if self.drawer is not None:
            scale0 = self.drawer.GetDrawCoords(self.points[0])
            scale1 = self.drawer.GetDrawCoords(self.points[1])

            ax = scale1.x - scale0.x
            bx = scale0.x

            ay = scale1.y - scale0.y
            by = scale0.y

            return Point2D((x_svg - bx) / ax, (y_svg - by) / ay)
        else:
            return Point2D(0.0, 0.0)

    def update_coordlist(self):
        if self.mol is not None:
            self.coordlist = np.array([list(self.drawer.GetDrawCoords(i)) for i in range(self._drawmol.GetNumAtoms())])
            #self.logger.debug("Current coordlist:\n%s" % self.coordlist)
        else:
            self.coordlist = None

    def get_nearest_atom(self, x_svg, y_svg):
        if self.mol is not None and self.mol.GetNumAtoms() > 0:
            atomsvgcoords = np.array([x_svg, y_svg])
            # find distance, https://codereview.stackexchange.com/questions/28207/finding-the-closest-point-to-a-list-of-points
            deltas = self.coordlist - atomsvgcoords
            dist_2 = np.einsum("ij,ij->i", deltas, deltas)
            min_idx = np.argmin(dist_2)
            return min_idx, dist_2[min_idx] ** 0.5
        else:
            return None, 1e10  # Return ridicilous long distance so that its not chosen

    def get_nearest_bond(self, x_svg, y_svg):
        if self.mol is not None and len(self.mol.GetBonds()) > 0:
            bondlist = []
            for bond in self._drawmol.GetBonds():
                bi = bond.GetBeginAtomIdx()
                ei = bond.GetEndAtomIdx()
                avgcoords = np.mean(self.coordlist[[bi, ei]], axis=0)
                bondlist.append(avgcoords)

            bondlist = np.array(bondlist)
            # if not bondlist:  # If there's no bond
            #     return None, 1e10
            atomsvgcoords = np.array([x_svg, y_svg])
            deltas = bondlist - atomsvgcoords
            dist_2 = np.einsum("ij,ij->i", deltas, deltas)
            min_idx = np.argmin(dist_2)
            return min_idx, dist_2[min_idx] ** 0.5
        else:
            return None, 1e10  # Return ridicilous long distance so that its not chosen

    # Function that translates coodinates from an event into a molobject
    def get_molobject(self, event):
        # Recalculate to SVG coords
        viewbox = self.renderer().viewBox()
        size = self.size()

        x = event.pos().x()
        y = event.pos().y()
        # Rescale, divide by the size of the widget, multiply by the size of the viewbox + offset.
        x_svg = float(x) / size.width() * viewbox.width() + viewbox.left()
        y_svg = float(y) / size.height() * viewbox.height() + viewbox.top()
        self.logger.debug("SVG_coords:\t%s\t%s" % (x_svg, y_svg))
        # Identify Nearest atomindex
        atom_idx, atom_dist = self.get_nearest_atom(x_svg, y_svg)
        bond_idx, bond_dist = self.get_nearest_bond(x_svg, y_svg)
        self.logger.debug("Distances to atom %0.2F, bond %0.2F" % (atom_dist, bond_dist))
        # If not below a given threshold, then it was not clicked
        #if min([atom_dist, bond_dist]) < 20.0:
        if atom_dist < 20.0:
            atom_idx = self.getProp(self._drawmol, int(atom_idx))
            return self.mol.GetAtomWithIdx(atom_idx)
        else:
            # Translate SVG to Coords
            return self.SVG_to_coord(x_svg, y_svg)

    def mousePressEvent(self, event):
        if event.button() is QtCore.Qt.LeftButton:
            # For visual feedback on the dragging event
            self.press_pos = event.position()
            self.current_pos = event.position()
            self.is_dragging = True
            
            # For chemistry
            self.start_molobject = self.get_molobject(event)
            self.event_handler(self.start_molobject, None)  # Click events has None as second object

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.is_dragging:
            self.current_pos = event.position()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() is QtCore.Qt.LeftButton:
            end_mol_object = self.get_molobject(event)
            if self.is_same_object(self.start_molobject, end_mol_object):
                return
            else:
                self.event_handler(
                    self.start_molobject, end_mol_object
                )  # Drag events has different objects as start and end
            self.start_molobject = None

            self.is_dragging = False
            self.update()  # Final repaint to clear the line

    def paintEvent(self, event):
        super().paintEvent(event)  # Render the SVG (Molecule)

        # Paint a line from where the canvas was clicked to the current position.
        if self.is_dragging:
            painter = QPainter(self)
            pen = QPen(Qt.gray, 4, Qt.SolidLine)
            painter.setPen(pen)
            painter.drawLine(self.press_pos, self.current_pos)

    def is_same_object(self, object1, object2):
        if isinstance(object1, Chem.rdchem.Atom) and isinstance(object2, Chem.rdchem.Atom):
            return object1.GetIdx() == object2.GetIdx()
        if isinstance(object1, Chem.rdchem.Bond) and isinstance(object2, Chem.rdchem.Bond):
            return object1.GetIdx() == object2.GetIdx()
        if isinstance(object1, Point2D) and isinstance(object2, Point2D):
            distance = (object1 - object2).Length()
            self.logger.debug(f"Dragged distance on Canvas {distance}")
            if distance < 0.1:
                return True
        return False

    # def clicked_handler(self, clicked):
    #     try:
    #         self.event_handler(clicked, None)
    #     except Exception as e:
    #         self.logger.error(f"Error in clicked_handler: {e}")

    # def drag_handler(self, object1, object2):
    #     try:
    #         self.event_handler(object1, object2)
    #     except Exception as e:
    #         self.logger.error(f"Error in drag_handler: {e}")

    def event_handler(self, object1, object2):
        # Matches which objects are clicked/dragged and what chemical type and action is selected
        # With click events, the second object is None
        # Canvas clicks and drags are Point2D objects
        match (object1, object2, self.action):

            case (Chem.rdchem.Atom(), None, "Select"):
                self.select_atom_add(object1)
            case (Chem.rdchem.Atom(), None, "Cut1"):
                self.select_atom_add(object1, cut=1)
            case (Chem.rdchem.Atom(), None, "Cut2"):
                self.select_atom_add(object1, cut=2)
            case (Chem.rdchem.Atom(), None, "Increase Charge"):
                self.increase_charge(object1)
            case (Chem.rdchem.Atom(), None, "Decrease Charge"):
                self.decrease_charge(object1)
            #case (Chem.rdchem.Atom(), None, "Number Atom"):
            #    self.number_atom(object1)
            #case (Chem.rdchem.Atom(), None, "RStoggle"):
            #    self.toogleRS(object1)
            #case (Chem.rdchem.Bond(), None, "EZtoggle"):
            #    self.toogleEZ(object1)

            # Canvas click events
            case (Point2D(), None, "Select"):
                self.clearAtomSelection()

            # Drag events
            # Atom to Atom
            case (Chem.rdchem.Atom(), Chem.rdchem.Atom(), "Select"):
                self.select_atom_add(object2)
            case (Chem.rdchem.Atom(), Chem.rdchem.Atom(), "Cut1"):
                self.select_atom_add(object2, cut=1)
            case (Chem.rdchem.Atom(), Chem.rdchem.Atom(), "Cut2"):
                self.select_atom_add(object2, cut=2)

            # Default case for undefined actions
            case _:
                self.logger.warning(
                    f"Undefined action for combination: "
                    f"{(type(object1), type(object2), self.action)}"
                )

    def getNewAtom(self, chemEntity):
        newatom = Chem.rdchem.Atom(chemEntity)
        if newatom.GetAtomicNum() == 0:
            newatom.SetProp("dummyLabel", "R")
        return newatom
    
    def select_atom_add(self, atom, cut=0):
        selidx = atom.GetIdx()
        if selidx in self.selectedAtoms:
            self.unselectAtom(selidx, cut)
        else:
            self.selectAtom(selidx, cut)

    def replace_atom(self, atom):
        rwmol = Chem.rdchem.RWMol(self.mol)
        newatom = self.getNewAtom(self.chemEntity)
        rwmol.ReplaceAtom(atom.GetIdx(), newatom)
        self.mol = rwmol


    def toogleRS(self, atom):
        self.backupMol()
        # atom = self._mol.GetAtomWithIdx(atom.GetIdx())
        stereotype = atom.GetChiralTag()
        self.logger.debug("Current stereotype of clicked atom %s" % stereotype)
        stereotypes = [
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
            # Chem.rdchem.ChiralType.CHI_OTHER, this one doesn't show a wiggly bond
            Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        ]
        newidx = np.argmax(np.array(stereotypes) == stereotype) + 1
        atom.SetChiralTag(stereotypes[newidx])
        self.logger.debug("New stereotype set to %s" % atom.GetChiralTag())
        # rdDepictor.Compute2DCoords(self._mol)
        # self._mol.ClearComputedProps()
        self._mol.UpdatePropertyCache(strict=False)
        rdDepictor.Compute2DCoords(self._mol)
        self.molChanged.emit()

    def assert_stereo_atoms(self, bond):
        if len(bond.GetStereoAtoms()) == 0:
            # get atoms and idx's of bond
            bondatoms = [bond.GetBeginAtom(), bond.GetEndAtom()]
            bondidx = [atom.GetIdx() for atom in bondatoms]

            # Figure out the atom idx's of the neigbor atoms, that are NOT the other end of the bond
            stereoatoms = []
            for bondatom in bondatoms:
                neighboridxs = [atom.GetIdx() for atom in bondatom.GetNeighbors()]
                neighboridx = [idx for idx in neighboridxs if idx not in bondidx][0]
                stereoatoms.append(neighboridx)
            # Set the bondstereoatoms
            bond.SetStereoAtoms(stereoatoms[0], stereoatoms[1])
            self.logger.debug(f"Setting StereoAtoms to {stereoatoms}")
        else:
            pass

    def assign_stereo_atoms(self, mol: Chem.Mol):
        self.logger.debug("Identifying stereo atoms")
        mol_copy = copy.deepcopy(mol)
        Chem.SanitizeMol(mol_copy, sanitizeOps=Chem.rdmolops.SanitizeFlags.SANITIZE_SYMMRINGS)
        Chem.rdmolops.FindPotentialStereoBonds(mol_copy, cleanIt=True)
        for i, bond in enumerate(mol_copy.GetBonds()):
            stereoatoms = list(
                set(bond.GetStereoAtoms())
            )  # Is FindPotentialStereoBonds are run successively, the list is simply expanded.
            if stereoatoms:
                try:
                    mol.GetBondWithIdx(i).SetStereoAtoms(stereoatoms[0], stereoatoms[1])
                except RuntimeError:
                    mol.GetBondWithIdx(i).SetStereoAtoms(
                        stereoatoms[1], stereoatoms[0]
                    )  # Not sure why this can get the wrong way. Seem to now work correctly for Absisic Acid

    def toogleEZ(self, bond: Chem.Bond):
        self.backupMol()

        stereotype = bond.GetStereo()  # TODO, when editing the molecule, we could change the CIP rules?
        # so stereo assignment need to be updated on other edits as well?
        self.logger.debug("Current stereotype of clicked atom %s" % stereotype)
        self.logger.debug(f"StereoAtoms are {list(bond.GetStereoAtoms())}")
        self.logger.debug(f"Bond properties are {bond.GetPropsAsDict(includePrivate=True, includeComputed=True)}")

        self.assign_stereo_atoms(self._mol)  # TODO, make something that ONLY works on a single bond?

        stereocycler = {
            Chem.rdchem.BondStereo.STEREONONE: Chem.rdchem.BondStereo.STEREOTRANS,
            Chem.rdchem.BondStereo.STEREOE: Chem.rdchem.BondStereo.STEREOCIS,
            Chem.rdchem.BondStereo.STEREOTRANS: Chem.rdchem.BondStereo.STEREOCIS,
            Chem.rdchem.BondStereo.STEREOZ: Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOCIS: Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOANY: Chem.rdchem.BondStereo.STEREONONE,
        }

        newstereotype = stereocycler[stereotype]
        bond.SetStereo(newstereotype)

        self.logger.debug("New stereotype set to %s" % bond.GetStereo())
        self.logger.debug(f"StereoAtoms are {list(bond.GetStereoAtoms())}")
        self.logger.debug(f"Bond properties are {bond.GetPropsAsDict(includePrivate=True, includeComputed=True)}")

        self.logger.debug(f"StereoAtoms are {list(bond.GetStereoAtoms())}")
        self.logger.debug(f"Bond properties are {bond.GetPropsAsDict(includePrivate=True, includeComputed=True)}")

        self.molChanged.emit()

    def increase_charge(self, atom):
        self.backupMol()
        atom.SetFormalCharge(atom.GetFormalCharge() + 1)
        self.molChanged.emit()

    def decrease_charge(self, atom):
        self.backupMol()
        atom.SetFormalCharge(atom.GetFormalCharge() - 1)
        self.molChanged.emit()

    def number_atom(self, atom: Chem.Atom):
        atomMapNumber = atom.GetIntProp("molAtomMapNumber") if atom.HasProp("molAtomMapNumber") else 0
        (atomMapNumber, ok) = QtWidgets.QInputDialog.getInt(
            self, "Number Atom", "Atom number", value=atomMapNumber, minValue=0
        )

        if not ok:
            return

        self.backupMol()
        if atomMapNumber == 0:
            atom.ClearProp("molAtomMapNumber")
        else:
            atom.SetProp("molAtomMapNumber", str(atomMapNumber))
        self.molChanged.emit()

    def undo(self):
        self.mol = self._prevmol

    def backupMol(self):
        self._prevmol = copy.deepcopy(self.mol)

    def cleanup_mol(self):
        mol = copy.deepcopy(self.mol)
        if self.sanitize_on_cleanup:
            Chem.SanitizeMol(mol)
        if self.kekulize_on_cleanup:
            Chem.Kekulize(mol)
        # if Chem.MolToCXSmiles(self.mol) != Chem.MolToCXSmiles(mol):
        self.mol = mol


if __name__ == "__main__":
    mol = Chem.MolFromSmiles("CCN(C)C1CCCCC1S")
    rdDepictor.Compute2DCoords(mol)
    myApp = QtWidgets.QApplication(sys.argv)
    molblockview = MolWidget(mol)
    molblockview.show()
    myApp.exec()
