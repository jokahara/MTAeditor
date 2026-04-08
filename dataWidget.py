import os, sys
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PySide6 import QtCore
from PySide6.QtCore import Qt, QRunnable, QThreadPool, QTimer, Slot, Signal, QObject
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (QLayout, QApplication, QCheckBox, QComboBox,
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

from molEditWidget import MolEditWidget
#from patternsWidget import PatternsWidget
from tools import *
from orb_calculator import *

LOADING_ENABLED = True

class WorkerSignals(QObject):
    """Signals from a running worker thread.

    finished
        int thread_id

    error
        tuple (exctype, value, traceback.format_exc())

    result
        object data returned from processing, anything

    progress
        tuple (thread_id, progress_value)
    """

    inputs = Signal(object)
    finished = Signal(int)  # thread_id
    error = Signal(tuple)
    result = Signal(object)
    progress = Signal(tuple)  # (thread_id, progress_value)


class Worker(QRunnable):
    """Worker thread."""
        
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.thread_id = kwargs.get("thread_id", 0)
        # Add the callback to our kwargs
        #self.kwargs["progress_callback"] = self.signals.progress

    @Slot()
    def run(self):
        print("Thread start")
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit(self.thread_id)
        print("Thread complete")

class DropLineEdit(QLineEdit):
    fileDropped = QtCore.Signal(list)

    def __init__(self, **kwarks):
        super(DropLineEdit, self).__init__(**kwarks)
        self.setReadOnly(True)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(Qt.CopyAction)

            for url in event.mimeData().urls():
                url = str(url.toLocalFile())
                if url.endswith('.pkl'):
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
                if url.endswith('.pkl'):
                    links.append(url)
            self.fileDropped.emit(links)
            self.setText(' '.join(links))
        else:
            event.ignore()
        

class DataWidget(QGroupBox):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.editor: MolEditWidget = parent.editor
        self.cluster_filter = None
        
        self.i =  0
        self.results = pd.DataFrame(columns=['molecule', 'conformer', 'H-bond', 'energy', 'Hbond_pairs', 'length', 'angle', 'fragments'])
        self.frags = (None, None, None)
        self.mode = ''
        
        self.calculator = None
        if LOADING_ENABLED:
            self.threadpool = QThreadPool.globalInstance()
            thread_count = self.threadpool.maxThreadCount()
            print(f"Multithreading with maximum {thread_count} threads")
            loader = Worker(get_calculator)
            loader.signals.result.connect(self.set_calculator)
            self.threadpool.start(loader)
        
        self.pickle_file = DropLineEdit(placeholderText='file missing')
        self.fileDropped = self.pickle_file.fileDropped
        self.pickle_file.setStatusTip('Pickled data file')
        self.pickle_file.textChanged.connect(self.load_pickled_data)

        self.mol_select = QComboBox()
        self.mol_select.setStatusTip('Select Monomer/Cluster type')
        self.mol_select.currentTextChanged.connect(self.select_cluster_type)
        
        self.conformer = 0
        self.conf_select = QSpinBox(minimum=0, maximum=0)
        self.conf_select.setStatusTip('Select Conformer')
        self.conf_select.valueChanged.connect(self.select_conformer)

        main_layout = QGridLayout(self)
        main_layout.addWidget(QLabel('Pickle File:'), 0, 0)
        main_layout.addWidget(self.pickle_file, 0, 1, 1, 3) # -> Table: level of theory
        main_layout.addWidget(QLabel('Cluster type:'), 1, 0) 
        main_layout.addWidget(self.mol_select, 1, 1) 
        main_layout.addWidget(QLabel('Conformer:'), 1, 2) 
        main_layout.addWidget(self.conf_select, 1, 3) 
        main_layout.setColumnStretch(1,1)

        # units in Ha, eV, kcal/mol
        # sort by Elec. or Gibbs
        # shifted to minimum check

        self.tabs = QTabWidget(self)
        main_layout.addWidget(self.tabs, 5, 0, 1, 4)

        self._pickle_columns = [('info', 'file_basename'), ('temp', 'N_Hbonds'),
                                ('log', 'electronic_energy'), ('log', 'gibbs_free_energy')]
        self.conf_table = QTableWidget(self.tabs, columnCount=4)
        self.conf_table.setHorizontalHeaderLabels(['basename', 'H-bonds', 'Elec. energy', 'Gibbs energy'])
        self.conf_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.conf_table.setColumnWidth(1, 60)       

        self.tabs.addTab(self.conf_table, 'Conformers')

        self.Hbond_table = QTableWidget(self.tabs, columnCount=4)
        self.Hbond_table.setHorizontalHeaderLabels(['D', 'A', 'length (Å)', 'angle (deg)'])
        self.Hbond_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tabs.addTab(self.Hbond_table, 'H-bonds')

        self.Hbond_results = {}
        self.results_tree = QTreeWidget(self.tabs, columnCount=4)
        self.results_tree.setHeaderLabels(['Conformer', 'Hbond', 'Energy (kcal/mol)', ''])
        self.results_tree.setColumnWidth(0,160)
        self.results_tree.setColumnWidth(1,70)
        self.results_tree.setColumnWidth(3,10)
        self.tabs.addTab(self.results_tree, 'Results')

        self.limit_table = QTableWidget(self.tabs, rowCount=3, columnCount=2)
        self.limit_table.setWindowTitle('H-bonding limits')
        self.limit_table.setHorizontalHeaderLabels(['length (Å)', 'angle (deg)'])
        self.limit_table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked)
        self.limit_table.cellChanged.connect(self.update_limits)
        self.tabs.addTab(self.limit_table, 'Limits')

        self.patterns_table = PatternsWidget(self.tabs)
        self.patterns_table.setWindowTitle('Patterns')
        self.load_patterns()
        self.tabs.addTab(self.limit_table, 'Patterns')

        self.setWindowTitle("Data")
        self.setMinimumWidth(440)

    def set_calculator(self, calc):
        self.calculator = calc
        
        self.parent().parent().calcAction.setText("Calculate")

    def calculate_frag_energies(self):
        if self.calculator == None or self.mode == '':
            return
        
        self.parent().parent().calcAction.setEnabled(False)
        self.editor.logger.info(f'Calculating in {self.mode.replace('*', 'Reduced ')} mode')
            
        mol = self._rdmol_simple if '*' in self.mode else self._rdmol
        if 'MTA' in self.mode:
            frags = [mol]+list(self.frags)
            worker = Worker(get_single_point_energies, frags, self.calculator)
        elif 'ROT' in self.mode:
            worker = Worker(get_rotational_energies, mol, self.rotation, self.calculator)
        worker.signals.result.connect(self.add_results)
        self.threadpool.start(worker)

    
    def add_results(self, results):        
        self.parent().parent().calcAction.setEnabled(True)
        conf_name = self.clusters_df[('info', 'file_basename')].values[self.conf_id]
        e0 = results[0][0]
        if 'MTA' in self.mode:
            bondE = results[0][0] - results[1][0] - results[2][0] + results[3][0]
            cuts = (self.cuts[0].copy(), self.cuts[1].copy())
            fragments = {'E0': e0, 'cuts': cuts, 'F1': results[1], 'F2': results[2], 'F12': results[3]}
        elif 'ROT' in self.mode:
            el, angle, peak_idx = results

            # Draw results into a plot
            from MTAeditor.pyplot_figures import plot_rotation
            plot_rotation(el, angle, self.bond_name, conf_name, peak_idx)
            
            closest = np.argmin(np.abs(angle[peak_idx]-180))
            #print(np.array([el, angle]).T)
            bondE = el[0] - el[peak_idx[closest]]
            fragments = {'E0': el[0], 'bond': self.rotation, 'scan': results}

        pairs = []; lengths = []; angles = []
        for b in self.bond_name.split(' + '):
            don, acc, length, angle = self.Hbond_df.loc[b.removeprefix('r').removesuffix('*')]
            pairs.append((int(don), int(acc)))
            lengths.append(length)
            angles.append(angle)

        #self.results = pd.DataFrame(columns=['molecule', 'conformer', 'H-bond', 'energy', 'Hbond_pairs', 'length', 'angle', 'fragments'])
        self.results.loc[self.i] = [self.cluster_type, conf_name, self.bond_name, bondE * E, 
                                    pairs, lengths, angles, fragments]
        self.i += 1

        self.update_results_tree(self.bond_name, bondE, self.i - 1)
        self.tabs.setCurrentIndex(2)

    def load_pickled_data(self, pickle_file, mol_file=None):
        if not os.path.isfile(pickle_file):
            return
        if self.pickle_file.text() == pickle_file:
            return

        self.pickle_file.setText(pickle_file)
        if self.cluster_filter == None:
            self.cluster_filter = load_cluster_filter(pickle_file, mol_file)
        else:
            self.cluster_filter.read_cluster_data(pickle_file)

        self.cluster_types = self.cluster_filter.cluster_types
        self.mol_select.clear()
        self.mol_select.addItems(self.cluster_types)
        self.select_cluster_type(self.cluster_types[0])        

        return

    def select_cluster_type(self, ct, conf=0):
        self.parent().parent().clearFragments()
        self.cluster_type = ct
        cf = self.cluster_filter
        if cf == None or ct == '':
            return
        
        cf.reset()            
        self.set_limits_table(self.cluster_filter.Hbond_limits)
        cf.extract_clusters(ct)
        cf.Hbonded(100, 'X')
        self.clusters_df = cf.get_filtered_data(True)
        if ('log', 'gibbs_free_energy') in self.clusters_df.columns:
            self.clusters_df = self.clusters_df.sort_values(by=[('log', 'gibbs_free_energy'), ('log', 'electronic_energy')])
        else:
            self.clusters_df = self.clusters_df.sort_values(by=('log', 'electronic_energy'))

        self.editor.logger.info(f"Selected {ct} with {str(len(self.clusters_df))} conformer(s)")

        self.conf_select.setValue(0)
        self.conf_select.setRange(0, len(self.clusters_df)-1)
        self.update_conf_table(self.clusters_df)

        self.editor.clearAtomSelection()
        self._rdmol = generate_mol(cf, ct)
        self.select_conformer(conf)

    def select_conformer(self, conf_id=0):
        self.mode = ''
        self._rdmol_simple = None
        self.conf_id = conf_id
        don, acc = self.clusters_df[('temp', 'Hbond_pairs')].values[conf_id]
        atoms = self.clusters_df[('xyz', 'structure')].values[conf_id]
        self._rdmol.GetConformer(0).SetPositions(atoms.positions)

        self.editor.mol = add_hydrogen_bonds(self._rdmol, don, acc)

        lengths = self.clusters_df[('temp', 'Hbond_lengths')].values[conf_id]
        angles = self.clusters_df[('temp', 'Hbond_angles')].values[conf_id]
        n = len(don)
        self.Hbond_table.setRowCount(n)
        self.Hbond_table.setVerticalHeaderLabels(['HB'+str(i+1) for i in range(n)])
        
        self.Hbond_df = pd.DataFrame(columns=['D', 'A', 'length', 'angle'])
        
        for i in range(n):
            self.Hbond_table.setItem(i, 0, QTableWidgetItem(str(don[i])))
            self.Hbond_table.setItem(i, 1, QTableWidgetItem(str(acc[i])))
            length = round(lengths[i], 3)
            self.Hbond_table.setItem(i, 2, QTableWidgetItem(str(length)))
            angle = np.round(angles[i], 2)
            angle = angle[0] if len(angle) == 1 else angle
            self.Hbond_table.setItem(i, 3, QTableWidgetItem(str(angle)))
            self.Hbond_df.loc['HB'+str(i+1)] = np.array([don[i], acc[i], length, angle], dtype=object)

            for j in range(4):
                self.Hbond_table.item(i,j).setTextAlignment(Qt.AlignmentFlag.AlignCenter)

        self.Hbond_table.resizeColumnsToContents()
    
    def get_fragments(self, cut1=[], cut2=[]):
        frag1 = frag2 = frag12 = None
        proceed = True
        
        mol = self._rdmol_simple if '*' in self.mode else self._rdmol
        if len(cut1) > 0:
            frag1 = cut_molecule(mol, cut1)
        if len(cut2) > 0:
            frag2 = cut_molecule(mol, cut2)
        if (len(cut1) > 0) and (len(cut2) > 0):
            try:
                frag12 = cut_molecule(frag1, cut2)
            except:
                frag12 = cut_molecule(frag2, cut1)

        if frag1 == None or frag2 == None or frag12 == None:
            return frag1, frag2, frag12
        
        # check that only H-bonds are calculated:
        bonds0 = get_bonds(self.editor.mol)
        bonds1 = get_bonds(frag1); bonds2 = get_bonds(frag2); bonds12 = get_bonds(frag12)
        remainder = bonds0 - bonds1 - (bonds2 - bonds12)
        props = get_props(self.editor.mol)
        idx1 = get_props(frag1); idx2 = get_props(frag2); idx12 = get_props(frag12)
        bond_names = []
        for a,b in list(remainder):
            bond = self.editor.mol.GetBondBetweenAtoms(*get_idx(props, (a,b)))
            if bond.GetBondType() == Chem.BondType.HYDROGEN:
                remains = int(a in idx1 and b in idx1) \
                            + int(a in idx2 and b in idx2) \
                            - int(a in idx12 and b in idx12)
                if remains == 0:
                    bond_names.append(bond.GetProp('bondNote'))
            else:
                bond_names.append('B'+str(a)+'-'+str(b))
                proceed = False
        if not proceed:
            self.editor.logger.error(f"Non-hydrogen bonds remaining in framents: {bond_names}")
            return

        self.editor.logger.info(f"The following bonds were left over from fragments: {bond_names}")
        self.bond_name = ' + '.join(sorted(bond_names))

        self.cuts = (cut1, cut2)
        self.frags = (frag1, frag2, frag12)
        return self.frags
    
    def internal_rotation(self, rot=[]):
        try:
            assert len(rot) == 2
            self.rotation = rot
            if '*' in self.mode:
                new_mol = rotate_bond(self._rdmol_simple, rot[0], rot[1])
            else:
                new_mol = rotate_bond(self._rdmol, rot[0], rot[1])

            names = []
            d1 = Chem.Get3DDistanceMatrix(self.editor.mol)
            d2 = Chem.Get3DDistanceMatrix(new_mol)
            for bond in self.editor.mol.GetBonds():
                if bond.GetBondType() == Chem.BondType.HYDROGEN:
                    a = bond.GetBeginAtomIdx()
                    b = bond.GetEndAtomIdx()
                    if np.abs(d1[a,b] - d2[a,b]) > 0.01: 
                        names.append('r'+bond.GetProp('bondNote'))
                    
            self.bond_name = ' + '.join(names)
            self.frags = (new_mol, )
            return new_mol
        except: 
            self.editor.logger.error(f"Rotation {rot} failed")

        return
    
    def simplify(self, cuts):
        if len(cuts) == 0:
            return None
        
        self._rdmol_simple = None
        #try:
        self._rdmol_simple = cut_molecule(self._rdmol, cuts)
        don, acc = self.clusters_df[('temp', 'Hbond_pairs')].values[self.conf_id]
        #atoms = self.clusters_df[('xyz', 'structure')].values[self.conf_id]
        new_mol = add_hydrogen_bonds(self._rdmol_simple, don, acc, self.editor.mol)
        return new_mol
    
    def update_conf_table(self, df):
        self.conf_table.setRowCount(len(df))
        self.conf_table.setVerticalHeaderLabels([str(i) for i in range(len(df))])
        
        for i, col in enumerate(self._pickle_columns):
            for row, idx in enumerate(df.index.values):
                if col in df.columns:
                    self.conf_table.setItem(row, i, QTableWidgetItem(str(df.loc[idx, col])))
                else:
                    self.conf_table.setItem(row, i, QTableWidgetItem('NaN'))
                self.conf_table.itemAt(row, i).setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        return

    def update_limits(self, row, col):
        if self.cluster_filter == None:
            return
        
        import re
        text = self.limit_table.currentItem().text()

        donor, acceptor = self.limit_table.verticalHeaderItem(row).text().split('--')
        lengths, angles = self.cluster_filter.Hbond_limits.loc[(donor, acceptor)]

        try: 
            input = re.sub(r'[^0-9,.]+', '',text).split(',')
            assert len(input) == 2

            a, b = float(input[0]), float(input[1])    
            assert a <= b
            
            if col == 0:
                if lengths == (a,b):
                    return
                assert a > 1.0
                assert b < 5
                self.cluster_filter.set_Hbond(donor, acceptor, length=(a, b))
            elif col == 1:
                if angles == (a,b):
                    return
                assert a > 0
                assert b <= 180
                self.cluster_filter.set_Hbond(donor, acceptor, angle=(a, b))
            
            self.select_cluster_type(self.mol_select.currentText(), conf=self.conf_select.value())

        except:
            self.editor.logger.error(f'Invalid limits: {text}')
            self.set_limits_table(self.cluster_filter.Hbond_limits)
        
        return
    
    def set_limits_table(self, df):
        self.limit_table.cellChanged.disconnect()
        self.limit_table.setRowCount(len(df))
        
        for i, idx in enumerate(df.index.values):
            self.limit_table.setVerticalHeaderItem(i, QTableWidgetItem(idx[0]+'--'+idx[1]))
            for j, col in enumerate(df.columns):
                item = QTableWidgetItem(str(df.at[idx, col]))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.limit_table.setItem(i, j, item)
        
        self.limit_table.cellChanged.connect(self.update_limits)
        return
    
    def update_results_tree(self, bond_name, energy, idx):
        
        items = self.results_tree.findItems(self.cluster_type, Qt.MatchFlag.MatchContains)
        if len(items) == 0:
            item = QTreeWidgetItem([self.cluster_type])
            self.results_tree.insertTopLevelItem(0, item)
        else: 
            item = items[0]

        conf = self.clusters_df[('info', 'file_basename')].values[self.conf_id]

        parents = self.results_tree.findItems(conf, Qt.MatchFlag.MatchRecursive)
        if len(parents) == 0:
            parent = QTreeWidgetItem([conf])
            item.addChild(parent)
        else: 
            parent = parents[0]
            
        child = QTreeWidgetItem(parent)
        child.setText(1, bond_name)
        child.setText(2, str(round(energy*E,5)))
        parent.addChild(child)
        
        delete_button = QPushButton('')
        def delete_row():
            child.parent().removeChild(child)
            self.results = self.results.drop(index=idx)

        delete_button.clicked.connect(delete_row)
        delete_button.setIcon(QIcon.fromTheme("icons8-Cancel"))
        self.results_tree.setItemWidget(child, 3, delete_button)
        parent.setExpanded(True)

        item.setExpanded(True)
        for i in range(4):
            self.results_tree.resizeColumnToContents(i)

        return
    
    def rotate_mol(self, radii, axis):
        if self.editor.mol == None:
            return
        
        rotate_mol(self.editor.mol, radii, axis)
        self.editor.molChanged.emit()

    def export_results(self, file='table.csv'):
        df = self.results.copy(deep=True)
        if file.endswith('.pkl'):
            df.to_pickle(file)
        elif file.endswith('.csv'):
            df.to_csv(file)
        elif file.endswith('.json'):
            try:
                write_json(file, df)
            except:
                self.editor.logger.error('Failed to write JSON file.')
                return
            
        self.editor.logger.info('Results saved to '+file)
        return



if __name__ == '__main__':
    app = QApplication(sys.argv)
    dialog = DataWidget()
    sys.exit(dialog.exec())