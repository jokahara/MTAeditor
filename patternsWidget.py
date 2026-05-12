import os, sys
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from rdkit import Chem

from PySide6 import QtCore, QtGui, QtWidgets, QtSvgWidgets
from PySide6.QtCore import Qt, QRunnable, QThreadPool, QTimer, Slot, Signal, QObject
from PySide6.QtGui import QIcon, QColor, QColorConstants, QBrush
from PySide6.QtWidgets import (QLayout, QCheckBox, QComboBox, QFileDialog, QMessageBox,
                               QCommandLinkButton, QDateTimeEdit, QDial,
                               QDialog, QDialogButtonBox, QFileSystemModel,
                               QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QListView, QMenu, QPlainTextEdit,
                               QProgressBar, QPushButton, QSizePolicy,
                               QScrollBar, QSizePolicy, QSlider, QSpinBox, QHeaderView,
                               QStyleFactory, QTableWidget, QTableWidgetItem, QTabWidget,
                               QTextBrowser, QTextEdit, QToolBox, QToolButton,
                               QTreeView, QTreeWidget, QTreeWidgetItem, QVBoxLayout, 
                               QWidget, QFormLayout, QAbstractItemView)

from molViewWidget import MolWidget
from tools import generate_mol_pattern

def group_widgets(parent, widgets):
    group = QWidget(parent)
    row_layout = QHBoxLayout(group)
    for widget in widgets:
        row_layout.addWidget(widget)
        
    row_layout.setContentsMargins(0, 0, 0, 0)
    #row_layout.setSpacing(0)
    return group

class DropLineEdit(QLineEdit):
    fileDropped = QtCore.Signal(list)

    def __init__(self, filetype='.pkl', **kwarks):
        super(DropLineEdit, self).__init__(**kwarks)
        self.setReadOnly(True)
        self.setAcceptDrops(True)
        self.filetype=filetype

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(Qt.CopyAction)

            for url in event.mimeData().urls():
                url = str(url.toLocalFile())
                if url.endswith(self.filetype):
                    event.accept()
                    return

        event.ignore()
        
    def dropEvent(self, event):
        
        if event.mimeData().hasUrls:
            event.setDropAction(Qt.CopyAction)
            event.accept()
            links = []
            for url in event.mimeData().urls():
                url = str(url.toLocalFile())
                if url.endswith(self.filetype):
                    links.append(url)
            self.fileDropped.emit(links)
            self.setText(' '.join(links))
        else:
            event.ignore()
        

class PatternsWidget(QWidget):

    def __init__(self, /, parent, editor: MolWidget=None):
        super().__init__(parent)

        self.editor = editor
        self.dataBox = parent

        self.count = 0
        self.df = pd.DataFrame(data={'SMILES':[], 'cut1':[], 'cut2':[], 'correction': [],
                                     'visible': [], 'matched': [], 'Mol': []}, dtype=object)

        layout = QGridLayout(self)
        layout.setColumnStretch(1,1)

        self.fileDrop = DropLineEdit(filetype=('.csv', '.pkl'), placeholderText='Patterns file')
        self.fileDrop.fileDropped.connect(self.load_patterns)
        self.saveButton = QPushButton('Save')  
        self.saveButton.clicked.connect(self.save_patterns)   
        self.loadButton = QPushButton('Load')  
        self.loadButton.clicked.connect(self.load_patterns) 
        self.saveButton.setFixedWidth(50)
        self.loadButton.setFixedWidth(50)
        layout.addWidget(group_widgets(self, [self.fileDrop, self.loadButton, self.saveButton]), 0, 0, 1, 3)   
        

        self.findButton = QPushButton('Find matches')
        self.findButton.setCheckable(True)
        self.findButton.clicked.connect(self.match_all_patterns)

        self.calcButton1 = QPushButton('Calculate Selected')
        self.calcButton1.clicked.connect(self.calculate)
        self.calcButton1.setEnabled(False)

        self.calcButton2 = QPushButton('Calculate Visible')
        self.calcButton2.clicked.connect(self.calculate_all)
        self.calcButton2.setEnabled(False)
        layout.addWidget(group_widgets(self, [self.findButton, self.calcButton1, self.calcButton2]), 1, 0, 1, 3)

        self.table = QTableWidget(self, columnCount=5)
        self.table.setHorizontalHeaderLabels(['Descriptor', 'Cut 1', 'Cut 2' , 'Shift', ''])
        self.table.cellChanged.connect(self.edit_pattern)
        self.current_match = -1
        self.table.cellClicked.connect(lambda row, col: self.draw_pattern(row, col, clicked=True))
        self.table.currentCellChanged.connect(self.draw_pattern)
        layout.addWidget(self.table, 2, 0, 1, 3)

        self.table.setColumnWidth(0,240)
        self.table.setColumnWidth(4,10)
        
        self.addButton = QPushButton('Add pattern')
        self.addButton.clicked.connect(self.add_selection)
        self.delButton = QPushButton('Delete pattern')
        self.delButton.clicked.connect(self.delete_pattern)
        layout.addWidget(group_widgets(self, [self.addButton, self.delButton]), 3, 0, 1, 3)   
        
        self.canvas = MolWidget()
        self.canvas.setMinimumSize(300,200)
        layout.addWidget(self.canvas, 4, 0, 1, 3)

        self.load_patterns()
        

    def add_pattern(self, patt, cut1, cut2, correction=0.0, visible=True):
        if patt in self.df:
            return
        
        self.table.cellChanged.disconnect()
        self.count += 1
            
        n = len(self.df)
        mol = generate_mol_pattern(patt)
        self.df.loc[self.count] = [patt, cut1, cut2, correction, visible, False, mol]
        self.table.setRowCount(n+1)
        
        self.table.setVerticalHeaderItem(n, QTableWidgetItem(str(n)))
        self.table.setItem(n, 0, QTableWidgetItem(patt))
        self.table.setItem(n, 1, QTableWidgetItem(str(cut1)[1:-1]))
        self.table.setItem(n, 2, QTableWidgetItem(str(cut2)[1:-1]))
        self.table.setItem(n, 3, QTableWidgetItem("{0:0.2f}".format(correction)))

        hide_button = QPushButton('')
        hide_button.setCheckable(True)
        hide_button.setIcon(QIcon.fromTheme("icons8-hide".format()))
        hide_button.clicked.connect(self.hide_pattern)
        self.table.setCellWidget(n, 4, hide_button)
        if not visible:
            hide_button.setChecked(True)
            color = QColorConstants.Gray
            for col in range(4):
                self.table.item(n, col).setForeground(color)

        self.table.cellChanged.connect(self.edit_pattern)
        self.match_all_patterns()
        return
    
    def add_selection(self):
        mol = self.editor.mol
        if mol is None or mol.GetNumAtoms() == 0:
            return
        
        selected = self.editor.selectedAtoms
        cut1 = self.editor._selectedAtoms[1]
        cut2 = self.editor._selectedAtoms[2]

        mol1 = Chem.CopyMolSubset(mol, selected)
        smiles = Chem.MolToSmiles(mol1, isomericSmiles=False)
        print('Selected:', smiles)
        frag = generate_mol_pattern(smiles)[0]
        props = {atom.GetIntProp('atomNote'): atom.GetIdx() for atom in mol1.GetAtoms()}
        # map cuts to new indexes
        cut1 = [[props[a], props[b]] for a,b in cut1]
        cut2 = [[props[a], props[b]] for a,b in cut2]
        #print(props)
        # map from copied subset to new fragment
        match = mol1.GetSubstructMatch(frag)
        match = {j: i for i, j in enumerate(match)}
        
        cut1 = [[match[a], match[b]] for a,b in cut1]
        cut2 = [[match[a], match[b]] for a,b in cut2]
        
        self.add_pattern(smiles.replace('~','--'), cut1, cut2)
        self.draw_pattern(self.table.rowCount()-1, 0)
        return


    def edit_pattern(self, row, col, draw=True):
        
        idx = self.df.index[row]
        text = self.table.item(row, col).text()
        if col == 0:
            self.df.loc[idx, 'SMILES'] = text
            mol = generate_mol_pattern(text)
            self.df.at[idx, 'Mol'] = mol
        elif col == 3:
            self.df.loc[idx, 'correction'] = float(text.replace(',', '.'))
            draw = False
        else:
            cuts = [x.split(']')[0] for x in text.split('[') if ']' in x]
            cuts = [x.split(',') for x in cuts]
            if col == 1:
                self.df.at[idx, 'cut1'] = [[int(a), int(b)] for a,b in cuts]
            elif col == 2:
                self.df.at[idx, 'cut2'] = [[int(a), int(b)] for a,b in cuts]

        if draw:
            self.draw_pattern(row, col)
            if col > 0:
                self.canvas.molChanged.emit()
        return 
    
    def hide_pattern(self):
        self.table.blockSignals(True)
        
        row = self.table.currentRow()
        i = self.df.index[row]
        self.df.loc[i,'visible'] = not self.df.loc[i,'visible']
        color = QColorConstants.Black if self.df.loc[i, 'visible'] else QColorConstants.Gray
        for col in range(4):
            self.table.item(row, col).setForeground(color)
            
        self.table.blockSignals(True)
        return

    def delete_pattern(self):
        row = self.table.currentRow()
        def msgApp(title, msg):
                userInfo = QMessageBox.question(self, title, msg)
                if userInfo == QMessageBox.Yes:
                    return "Y"
                if userInfo == QMessageBox.No:
                    return "N"
                
        response = msgApp("Confirmation", f"This will delete pattern {row}. Do you want to Continue?")
        if response == "Y":
            idx = self.df.index[row]
            self.df = self.df.drop(index=idx)
            self.table.removeRow(row)
            self.table.setVerticalHeaderLabels([str(i) for i in range(len(self.df))])
        else:
            self.editor.logger.debug("Abort")

        return

    def save_patterns(self):
        
        file, _ = QFileDialog.getSaveFileName(self, filter="CSV Files (*.csv *.csv);; Any File (*.*)")

        df = self.df[['SMILES', 'cut1', 'cut2', 'correction', 'visible']]
        df.index = [i for i in range(len(df))]
        df.to_csv(file, sep=';')
        
        return
    
    def load_patterns(self, file='default_patterns.csv'):
        if not os.path.isfile(file):
            file, _  = QFileDialog.getOpenFileName(self, caption="Open file", filter="CSV Files (*.csv *.csv)")
        
        self.fileDrop.setText(file)

        df = pd.read_csv(file, sep=';', index_col=0)
        df = df.sort_values(by='SMILES')
        self.count = 0
        self.df = pd.DataFrame(data={'SMILES':[], 'cut1':[], 'cut2':[], 'correction': [],
                                     'visible': [], 'matched': [], 'Mol': []}, dtype=object)
        for i in df.index:
            self.add_pattern(*df.loc[i].values)
        
        for i in range(self.table.rowCount()):
            self.edit_pattern(i, 1, False)
            self.edit_pattern(i, 2, False)

        self.canvas.clearAtomSelection()
        self.canvas.mol = None

        self.table.resizeColumnsToContents()
        self.table.setColumnWidth(4,10)
        return

    def reset(self):
        self.findButton.setChecked(False)
        self.match_all_patterns()

    def calculate(self, selection=False):

        if selection is False:
            rows = np.unique([item.row() for item in self.table.selectedIndexes()])
            selection = self.df.index[rows]
            
        cuts = []
        patterns = []
        corrections = []
        for idx in selection:
        
            matches = self.df.loc[idx, 'matched']
            if matches is None:
                continue

            for match in matches:
                cut1 = [ [match[a], match[b]] for a, b in self.df.loc[idx, 'cut1']]
                cut2 = [ [match[a], match[b]] for a, b in self.df.loc[idx, 'cut2']]
                if (cut1, cut2) in cuts or (cut2, cut1) in cuts:
                    continue
                cuts.append((cut1, cut2))
                patterns.append(self.df.loc[idx, 'SMILES'])
                corrections.append(self.df.loc[idx, 'correction'])

        self.dataBox.mode = 'MTA*' if '*' in self.dataBox.mode else 'MTA'
        self.dataBox.calculate_with_patterns(cuts, patterns, corrections)
        return

    def calculate_all(self):
        selection = []
        for idx in self.df.index:
            matches = self.df.loc[idx, 'matched']
            if matches is None:
                continue

            if self.df.loc[idx, 'visible']:
                selection.append(idx)
            
        self.calculate(selection)
        return


    def match_pattern(self, mol: Chem.Mol, patt: Chem.Mol):
        atom_idx = [a.GetIdx() for a in patt.GetAtoms()]
        matches = []
        for match in mol.GetSubstructMatches(patt):
            #print(Chem.MolToSmiles(patt))
            #print(match)
            #print(atom_idx, [atom.GetIntProp('atomNote') for atom in patt.GetAtoms()])

            import pickle
            with open('mol1.pkl', 'wb') as f:
                pickle.dump((mol, patt), f)

            passed = True
            for i, a in zip(atom_idx, match):
                for j, b in zip(atom_idx, match):
                    if i==j:
                        continue
                    bond1 = patt.GetBondBetweenAtoms(i,j)
                    bond2 = mol.GetBondBetweenAtoms(a,b)
                    if bond2 is None:
                        continue
                    
                    # make sure there are no additional covalent bonds 
                    if (bond2.GetBondType() > 0) and (bond2.GetBondType() < 14):
                        if (bond1 is None) or bond2.GetBondType() != bond1.GetBondType():
                            passed = False
                            break
                if not passed:
                    break
            
            if passed:
                matches.append(match)
            
        return matches

    def match_all_patterns(self):
        self.df['matched'] = [None]*len(self.df) 
        for row in range(self.table.rowCount()):
            self.table.showRow(row)
                
        if not self.findButton.isChecked():
            self.calcButton1.setEnabled(False)
            self.calcButton2.setEnabled(False)
            return
        
        if self.dataBox.calculator is not None:
            self.calcButton1.setEnabled(True)
            self.calcButton2.setEnabled(True)
        
        mol = self.editor.mol
        for row in range(self.table.rowCount()):
            idx = self.df.index[row]
            matches = []
            for patt in self.df.loc[idx, 'Mol']:
                matches += self.match_pattern(mol, patt)
            if len(matches) > 0:
                self.df.at[idx, 'matched'] = object()
                self.df.at[idx, 'matched'] = matches
            else:
                self.table.hideRow(row)
                
        return

    def draw_pattern(self, row, col, prev_row=-1, prev_col=-1, clicked=False):
        if col > 3: 
            return
        if not clicked and row == prev_row:
            return
        #patt = self.table.item(row, 0).text()
        idx = self.df.index[row]
        self.canvas._selectedAtoms[1] = self.df.loc[idx, 'cut1']
        self.canvas._selectedAtoms[2] = self.df.loc[idx, 'cut2']
        self.canvas.mol = self.df.loc[idx, 'Mol'][0]

        #import pickle
        #with open('mol1.pkl', 'wb') as f:
        #    pickle.dump((self.editor.mol, self.df.loc[idx, 'Mol']), f)
        # Try to find match in the molecule
        

        matches = []
        for patt in self.df.loc[idx, 'Mol']:
            matches += self.match_pattern(self.editor.mol, patt)
            
        n = len(matches)
        if n == 0:
            return
        
        self.editor.clearAtomSelection()
        cut1 = self.df.loc[idx, 'cut1']
        cut2 = self.df.loc[idx, 'cut2']

        # cycle through matches that get drawn
        if clicked:
            self.current_match += 1
            if self.current_match >= n:
                self.current_match = 0
            match = matches[self.current_match]
        else:
            self.current_match = -1
            match = matches[0]
        
        self.editor._selectedAtoms[1].extend([match[a], match[b]] for a, b in cut1)
        self.editor._selectedAtoms[2].extend([match[a], match[b]] for a, b in cut2)
        #self.editor._selectedAtoms[0] = list(np.concatenate(matches))
        self.editor.molChanged.emit()
        
        return
    
if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication, QMainWindow

    app = QApplication(sys.argv)
    window = QMainWindow()
    window.setCentralWidget(PatternsWidget(window))
    window.show()

    sys.exit(app.exec())