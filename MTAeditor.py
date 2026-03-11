#!/usr/bin/env python

# Import required modules
import sys
import os
import copy

from PySide6.QtWidgets import QMenu, QApplication, QStatusBar, QMessageBox, QFileDialog, QSplitter, QGroupBox, QGridLayout
from PySide6.QtCore import QSettings
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QUrl, Qt, QEvent, QObject
from PySide6.QtGui import QDesktopServices, QIcon, QAction, QKeySequence, QKeyEvent

# Import model
from .molEditWidget import MolEditWidget
from .mol3d_widget import Mol3DWindow
from .dataWidget import DataWidget

from rdkit import Chem
import qdarktheme


# The main window class
class MainWindow(QtWidgets.QMainWindow):
    # Constructor function
    def __init__(self, fileName=None, loglevel="WARNING"):
        super(MainWindow, self).__init__()
        self.pixmappath = os.path.abspath(os.path.dirname(__file__)) + "/pixmaps/"
        QtGui.QIcon.setThemeSearchPaths(
            # QtGui.QIcon.themeSearchPaths() +
            [os.path.abspath(os.path.dirname(__file__)) + "/icon_themes/"]
        )
        self.loglevels = ["Critical", "Error", "Warning", "Info", "Debug", "Notset"]
        # RDKit draw options, tooltip, default value is read from molViewWidget
        self._drawopts_actions = [
            (
                "prepareMolsBeforeDrawing",
                "Prepare molecules before drawing (i.e. fix stereochemistry and annotations)",
            ),
            (
                "addStereoAnnotation",
                "Add stereo annotation (R/S and E/Z)",
            ),
            (
                "unspecifiedStereoIsUnknown",
                "Show wiggly bond at potential undefined chiral stereo centres "
                + "and cross bonds for undefined doublebonds",
            ),
        ]
        
        self.editor = MolEditWidget()
        self.fragments = [MolEditWidget(), MolEditWidget(), MolEditWidget()]
        self.n_frags = 0 # Number of currently visible fragments
        self.dataBox = DataWidget(self)
        
        self.chemEntityActionGroup = QtGui.QActionGroup(self, exclusive=True)
        #self.mol3d = Mol3D(self)
        self._fileName = None
        self._mol_file = None
        self._pkl_file = None
        self.initGUI(fileName=fileName)
        self.applySettings()


    # Properties
    @property
    def fileName(self):
        return self._fileName

    @fileName.setter
    def fileName(self, filename):
        if filename != self._fileName:
            self._fileName = filename
            self.setWindowTitle(str(filename))

    def fileDropped(self, l):
        for url in l:
            if os.path.exists(url):
                print(url)
                self.fileName = url                
                self.loadFile()
    
    def clearFragments(self):
        for frag in self.fragments:
            frag.setVisible(False)
            frag.mol = None
        self.n_frags = 0
        self.editor.setMinimumSize(650, 650)
        return 

    def addFragment(self, mol_frag):
        if mol_frag == None:
            return
        
        self.fragments[self.n_frags].mol = mol_frag
        self.fragments[self.n_frags].setVisible(True)
        self.fragments[self.n_frags].setMinimumSize(350,350)
        self.n_frags += 1
        return

    def removeFragment(self):
        self.fragments[self.n_frags].setVisible(False)
        self.n_frags -= 1
        return

    def applyCuts(self):
        self.clearFragments()
        if self.editor.mol == None:
            return

        cut1, cut2 = self.editor.GetSelectedCuts()
        for frag in self.dataBox.get_fragments(cut1, cut2):
            self.addFragment(frag)

        if self.n_frags > 0:
            self.editor.setMinimumSize(350, 350)
        
        if self.n_frags == 3:
            self.calcAction.setEnabled(True)
        else:
            self.calcAction.setEnabled(False)

        return

    def showHydrogens(self):
        show = not self.showHsAction.isChecked()
        self.editor._removeHs = show
        self.editor.molChanged.emit()
        for frag in self.fragments:
            frag._removeHs = show
            frag.molChanged.emit()

        return

    def flatten(self):
        self.editor._flatten = self.cleanCoordinatesAction.isChecked()
        self.editor.molChanged.emit()
        for frag in self.fragments:
            frag._flatten = self.cleanCoordinatesAction.isChecked()
            frag.molChanged.emit()
        return

    def initGUI(self, fileName=None):
        self.setWindowTitle("rdEditor")
        self.setWindowIcon(QIcon.fromTheme("appicon"))
        self.setGeometry(100, 100, 200, 150)
        
        self.editor.fileDropped.connect(self.fileDropped)
        self.dataBox.fileDropped.connect(self.fileDropped)

        box = QGroupBox(self)
        box.setMinimumSize(700,700)
        box.sizePolicy().setHeightForWidth(True)
        #box.setSizePolicy(QtWidgets.QSizePolicy.setHeightForWidth())        
        self.center = QGridLayout(box)
        self.center.addWidget(self.editor, 0, 0)
        self.center.addWidget(self.fragments[0], 0, 1)
        self.center.addWidget(self.fragments[1], 1, 0)
        self.center.addWidget(self.fragments[2], 1, 1)
        self.clearFragments()

        split = QSplitter(self)
        split.addWidget(box)
        split.addWidget(self.dataBox)
        self.setCentralWidget(split)
        self.fileName = fileName

        self.filters = "CSV Files (*.csv *.csv);;Pickle Files (*.pkl *.pkl);;JSON Files (*.json *.json);;Any File (*)"

        self.SetupComponents()

        self.infobar = QtWidgets.QLabel("")
        self.myStatusBar.addPermanentWidget(self.infobar, 0)
        
        if self.fileName is not None:
            self.editor.logger.info("Loading molecule from %s" % self.fileName)
            self.loadFile()

        self.editor.sanitizeSignal.connect(self.infobar.setText)
        
        self.show()

    def getAllActionsInMenu(self, qmenu: QMenu):
        all_actions = []

        # Iterate through actions in the current menu
        for action in qmenu.actions():
            if isinstance(action, QAction):
                if action.icon():
                    all_actions.append(action)
            elif isinstance(action, QMenu):  # If the action is a submenu, recursively get its actions
                all_actions.extend(self.getAllActionsInMenu(action))

        return all_actions

    def getAllIconActions(self, qapp: QApplication):
        all_actions = []

        # Iterate through all top-level widgets in the application
        for widget in qapp.topLevelWidgets():
            # Find all menus in the widget
            menus = widget.findChildren(QMenu)
            for menu in menus:
                # Recursively get all actions from each menu
                all_actions.extend(self.getAllActionsInMenu(menu))

        return all_actions

    def resetActionIcons(self):
        actions_with_icons = list(set(self.getAllIconActions(QApplication)))
        for action in actions_with_icons:
            icon_name = action.icon().name()
            self.editor.logger.debug(f"reset icon {icon_name}")
            action.setIcon(QIcon.fromTheme(icon_name))

    def applySettings(self):
        self.settings = QSettings("Cheminformania.com", "rdEditor")
        theme_name = self.settings.value("theme_name", "Fusion")

        self.applyTheme(theme_name)
        self.themeActions[theme_name].setChecked(True)

        loglevel = self.settings.value("loglevel", "Error")

        action = self.loglevelactions.get(loglevel, None)
        if action:
            action.trigger()

        #sanitize_on_cleanup = self.settings.value("sanitize_on_cleanup", True, type=bool)
        #self.editor.sanitize_on_cleanup = sanitize_on_cleanup
        #self.cleanupSettingActions["sanitize_on_cleanup"].setChecked(sanitize_on_cleanup)

        #kekulize_on_cleanup = self.settings.value("kekulize_on_cleanup", True, type=bool)
        #self.editor.kekulize_on_cleanup = kekulize_on_cleanup
        #self.cleanupSettingActions["kekulize_on_cleanup"].setChecked(kekulize_on_cleanup)

        # Draw options
        for key, statusTip in self._drawopts_actions:
            viewer_value = self.editor.getDrawOption(key)
            settings_value = self.settings.value(f"drawoptions/{key}", viewer_value, type=bool)
            if settings_value != viewer_value:
                self.editor.setDrawOption(key, settings_value)
            self.drawOptionsActions[key].setChecked(settings_value)

        if self.settings.contains("drawoptions/fixedBondLength"):
            fixedBondLength = self.settings.value("drawoptions/fixedBondLength", 15, type=int)
            self.editor.setDrawOption("fixedBondLength", fixedBondLength)

    """
    def keyPressEvent(self, event: QKeyEvent):
        match event.key():
            #case Qt.Key.Key_Enter:
                #self.dataBox.limit_table.sele()
            case Qt.Key.Key_D:
                self.rotateR.emit(b'Right')
            case Qt.Key.Key_A:
                self.rotateL.emit(b'Left')
            case Qt.Key.Key_W:
                self.rotateU.emit(b'Up')
            case Qt.Key.Key_S:
                self.rotateD.emit(b'Down')"""
    

    # Function to setup status bar, central widget, menu bar, tool bar
    def SetupComponents(self):
        self.myStatusBar = QStatusBar()
        #        self.molcounter = QLabel("-/-")
        #        self.myStatusBar.addPermanentWidget(self.molcounter, 0)
        self.setStatusBar(self.myStatusBar)
        self.myStatusBar.showMessage("Ready", 10000)

        self.CreateActions()
        self.CreateMenus()
        self.CreateToolBars()

    # Actual menu bar item creation
    def CreateMenus(self):
        self.fileMenu = self.menuBar().addMenu("&File")
        # self.edit_menu = self.menuBar().addMenu("&Edit")

        self.toolMenu = self.menuBar().addMenu("&Tools")
        #self.atomtypeMenu = self.menuBar().addMenu("&AtomTypes")
        #self.bondtypeMenu = self.menuBar().addMenu("&BondTypes")
        #self.templateMenu = self.menuBar().addMenu("Tem&plates")
        self.settingsMenu = self.menuBar().addMenu("&Settings")
        self.helpMenu = self.menuBar().addMenu("&Help")

        self.fileMenu.addAction(self.openAction)
        self.fileMenu.addAction(self.saveAction)
        self.fileMenu.addAction(self.saveAsAction)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.copyAction)
        self.fileMenu.addAction(self.pasteAction)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAction)

        self.toolMenu.addAction(self.selectAction)
        self.toolMenu.addAction(self.cutAction1)
        self.toolMenu.addAction(self.cutAction2)
        self.toolMenu.addAction(self.applyAction)
        self.toolMenu.addAction(self.calcAction)
        #self.toolMenu.addAction(self.addAction)
        # self.toolMenu.addAction(self.addBondAction)
        # self.toolMenu.addAction(self.replaceAction)
        #self.toolMenu.addAction(
        #    self.rsAction
        #)  # TODO, R/S and E/Z could be changed for a single action? it really depends if an atom or a bond is clicked!
        #self.toolMenu.addAction(self.ezAction)
        #self.toolMenu.addAction(self.increaseChargeAction)
        #self.toolMenu.addAction(self.decreaseChargeAction)
        #self.toolMenu.addAction(self.numberAtom)

        self.toolMenu.addSeparator()
        self.toolMenu.addAction(self.showHsAction)
        self.toolMenu.addAction(self.cleanCoordinatesAction)
        #self.toolMenu.addAction(self.removeAction)
        self.toolMenu.addAction(self.clearCanvasAction)

        """# Atomtype menu
        for action in self.atomActions:
            self.atomtypeMenu.addAction(action)
        self.specialatommenu = self.atomtypeMenu.addMenu("All Atoms")
        for atomnumber in self.ptable.ptable.keys():
            atomname = self.ptable.ptable[atomnumber]["Symbol"]
            self.specialatommenu.addAction(self.ptable.atomActions[atomname])

        # Bondtype Menu
        self.bondtypeMenu.addAction(self.singleBondAction)
        self.bondtypeMenu.addAction(self.doubleBondAction)
        self.bondtypeMenu.addAction(self.tripleBondAction)
        self.bondtypeMenu.addSeparator()
        # Bondtype Special types
        self.specialbondMenu = self.bondtypeMenu.addMenu("Special Bonds")
        for key in self.bondActions.keys():
            self.specialbondMenu.addAction(self.bondActions[key])

        # Templates menu
        for key in self.templateActions.keys():
            self.templateMenu.addAction(self.templateActions[key])
        """
        # Settings menu
        self.themeMenu = self.settingsMenu.addMenu("Theme")
        self.populateThemeActions(self.themeMenu)
        self.loglevelMenu = self.settingsMenu.addMenu("Logging Level")
        for loglevel in self.loglevels:
            self.loglevelMenu.addAction(self.loglevelactions[loglevel])
        self.cleanupMenu = self.settingsMenu.addMenu("Cleanup")
        for key, action in self.cleanupSettingActions.items():
            self.cleanupMenu.addAction(action)
        self.drawOptionsMenu = self.settingsMenu.addMenu("Drawing Options")
        for key, statusTip in self._drawopts_actions:
            self.drawOptionsMenu.addAction(self.drawOptionsActions[key])

        # Help menu
        #self.helpMenu.addAction(self.aboutAction)
        self.helpMenu.addSeparator()
        #self.helpMenu.addAction(self.openChemRxiv)
        self.helpMenu.addAction(self.openRepository)
        self.helpMenu.addSeparator()
        self.helpMenu.addAction(self.aboutQtAction)

        # actionListAction = QAction(
        #     "List Actions", self, triggered=lambda: print(set(self.get_all_icon_actions_in_application(QApplication)))
        # )
        # self.helpMenu.addAction(actionListAction)

        # Debug level sub menu

    def populateThemeActions(self, menu: QMenu):
        stylelist = QtWidgets.QStyleFactory.keys() + ["Qdt light", "Qdt dark"]
        self.themeActionGroup = QtGui.QActionGroup(self, exclusive=True)
        self.themeActions = {}
        for style_name in stylelist:
            action = QAction(
                style_name,
                self,
                objectName=style_name,
                triggered=self.setTheme,
                checkable=True,
            )
            self.themeActionGroup.addAction(action)
            self.themeActions[style_name] = action
            menu.addAction(action)

    def CreateToolBars(self):
        self.mainToolBar = self.addToolBar("Main")
        # Main action bar
        self.mainToolBar.addAction(self.openAction)
        self.mainToolBar.addAction(self.saveAction)
        self.mainToolBar.addAction(self.saveAsAction)
        self.mainToolBar.addSeparator()
        self.mainToolBar.addAction(self.selectAction)
        self.mainToolBar.addAction(self.cutAction1)
        self.mainToolBar.addAction(self.cutAction2)
        self.mainToolBar.addAction(self.applyAction)
        self.mainToolBar.addAction(self.calcAction)
        #self.mainToolBar.addAction(self.addAction)
        #self.mainToolBar.addAction(self.addBondAction)
        #self.mainToolBar.addAction(self.replaceAction)
        #self.mainToolBar.addAction(self.rsAction)
        #self.mainToolBar.addAction(self.ezAction)
        #self.mainToolBar.addAction(self.increaseChargeAction)
        #self.mainToolBar.addAction(self.decreaseChargeAction)
        #self.mainToolBar.addAction(self.numberAtom)

        self.mainToolBar.addAction(self.rotateL)
        self.mainToolBar.addAction(self.rotateU)
        self.mainToolBar.addAction(self.rotateD)
        self.mainToolBar.addAction(self.rotateR)

        self.mainToolBar.addSeparator()
        self.mainToolBar.addAction(self.showHsAction)
        self.mainToolBar.addAction(self.cleanCoordinatesAction)
        self.mainToolBar.addAction(self.open3DmolAction)

        #self.mainToolBar.addAction(self.removeAction)
        self.mainToolBar.addAction(self.clearCanvasAction)
        
        # Side Toolbar
        """self.sideToolBar = QtWidgets.QToolBar(self)
        self.addToolBar(QtCore.Qt.LeftToolBarArea, self.sideToolBar)
        
        self.sideToolBar.addAction(self.singleBondAction)
        self.sideToolBar.addAction(self.doubleBondAction)
        self.sideToolBar.addAction(self.tripleBondAction)
        self.sideToolBar.addSeparator()
        self.sideToolBar.addAction(self.templateActions["benzene"])
        self.sideToolBar.addAction(self.templateActions["cyclohexane"])
        self.sideToolBar.addSeparator()
        self.sideToolBar.addAction(self.ptable.atomActions["R"])
        self.sideToolBar.addSeparator()
        for action in self.atomActions:
            self.sideToolBar.addAction(action)
        """

    def loadSmilesFile(self, filename):
        self.fileName = filename
        with open(self.fileName, "r") as file:
            lines = file.readlines()
            if len(lines) > 1:
                self.editor.logger.warning("The SMILES file contains more than one line.")
                self.statusBar().showMessage("The SMILES file contains more than one line.")
                return None
            smiles = lines[0].strip()
            mol = Chem.MolFromSmiles(smiles)
            self.editor.mol = mol
            self.statusBar().showMessage(f"SMILES file {filename} opened")

    def loadMolFile(self, filename):
        self.fileName = filename
        mol = Chem.MolFromMolFile(str(self.fileName), sanitize=False, strictParsing=False)
        self.editor.mol = mol
        self._mol_file = filename
        self.statusBar().showMessage(f"Mol file {filename} opened")

    def loadClusterFilter(self, file_in):
        try:
            self._pkl_file = file_in
            self.dataBox.load_pickled_data(file_in, self._mol_file)
            self.statusBar().showMessage(f"Pickle file {file_in} opened")
        except:
            self.editor.logger.error("Failed to load data.")


    def openFile(self):
        self.fileName, _ = QFileDialog.getOpenFileName(self, caption="Open file", filter=self.filters)
        return self.loadFile()
    
    def loadFile(self):
        if not self.fileName:
            self.editor.logger.warning("No file selected.")
            self.statusBar().showMessage("No file selected.")
            return
        if self.fileName.lower().endswith(".mol"):
            self.loadMolFile(self.fileName)
        elif self.fileName.lower().endswith(".smi"):
            self.loadSmilesFile(self.fileName)
        elif self.fileName.lower().endswith(".sdf"):
            self.loadSmilesFile(self.fileName)
        elif self.fileName.lower().endswith(".pkl"):
            self.loadClusterFilter(self.fileName)
        elif self.fileName.lower().endswith(".dat"):
            self.loadClusterFilter(self.fileName)
        elif self.fileName.lower().endswith(".txt"):
            self.loadClusterFilter(self.fileName)
        else:
            self.editor.logger.warning("Unknown file format. Assuming file as .mol format.")
            self.statusBar().showMessage("Unknown file format. Assuming file as .mol format.")
            self.loadMolFile(self.fileName)
            self.fileName += ".mol"

    def saveFile(self):
        #if self.fileName is not None:
        #    Chem.MolToMolFile(self.editor.mol, str(self.fileName))
        #else:
        self.saveAsFile()

    def saveAsFile(self):
        self.fileName, self.filterName = QFileDialog.getSaveFileName(self, filter=self.filters)
        if self.fileName != "":
            if self.fileName.endswith((".csv", '.pkl', '.json')):
                self.dataBox.export_results(self.fileName)
                self.statusBar().showMessage("File {self.filename} saved", 2000)
            else:
                self.statusBar().showMessage("Invalid file format", 2000)

    def copy(self):
        selected_text = Chem.MolToSmiles(self.editor.mol, isomericSmiles=True)
        clipboard = QApplication.clipboard()
        clipboard.setText(selected_text)

    def paste(self):
        clipboard = QApplication.clipboard()
        text = clipboard.text()
        mol = Chem.MolFromSmiles(text, sanitize=False)
        if mol:
            try:
                Chem.SanitizeMol(copy.deepcopy(mol))  # ).ToBinary()))
            except Exception as e:
                self.editor.logger.warning(f"Pasted SMILES is not sanitizable: {e}")

            self.editor.assign_stereo_atoms(mol)
            Chem.rdmolops.SetBondStereoFromDirections(mol)

            self.editor.mol = mol
        else:
            self.editor.logger.warning(f"Failed to parse the content of the clipboard as a SMILES: {repr(text)}")

    def clearCanvas(self):
        self.clearFragments()
        self.editor.clearAtomSelection()
        self.editor.mol = None
        self.dataBox.select_conformer(self.dataBox.conf_id)
        self.statusBar().showMessage("Canvas Cleared")

    def closeEvent(self, event):
        self.editor.logger.debug("closeEvent triggered")
        self.exitFile()
        event.ignore()

    def exitFile(self):
        response = self.msgApp("Confirmation", "This will quit the application. Do you want to Continue?")
        if response == "Y":
            exit(0)
        else:
            self.editor.logger.debug("Abort closing")

    # Function to show Diaglog box with provided Title and Message
    def msgApp(self, title, msg):
        userInfo = QMessageBox.question(self, title, msg)
        if userInfo == QMessageBox.Yes:
            return "Y"
        if userInfo == QMessageBox.No:
            return "N"
        self.close()

    def setAction(self):
        sender = self.sender()
        self.editor.setAction(sender.objectName())
        self.myStatusBar.showMessage("Action %s selected" % sender.objectName())

    def rotate(self):
        sender = self.sender()
        deg = 18
        match sender.objectName():
            case "Up":
                self.dataBox.rotate_mol(deg,0)
            case "Down":
                self.dataBox.rotate_mol(-deg,0)
            case "Left":
                self.dataBox.rotate_mol(-deg,1)
            case "Right":
                self.dataBox.rotate_mol(deg,1)
        self.myStatusBar.showMessage(sender.objectName())

    def open3Dmol(self):
        if self.editor.mol == None:
            return
        
        self.mol3d = Mol3DWindow()
        self.mol3d.set_molecule(self.editor.mol)
        self.mol3d.show()


    def setLogLevel(self):
        loglevel = self.sender().objectName().split(":")[-1]  # .upper()
        self.editor.logger.setLevel(loglevel.upper())
        self.editor.logger.log(self.editor.logger.getEffectiveLevel(), f"loglevel set to {loglevel}")
        self.settings.setValue("loglevel", loglevel)
        self.settings.sync()

    def setDrawOption(self):
        sender = self.sender()
        option = sender.objectName()
        self.editor.setDrawOption(option, sender.isChecked())
        self.settings.setValue(f"drawoptions/{option}", sender.isChecked())
        self.settings.sync()

    def setTheme(self):
        sender = self.sender()
        theme_name = sender.objectName()
        self.myStatusBar.showMessage(f"Setting theme or style to {theme_name}")
        self.applyTheme(theme_name)
        self.settings.setValue("theme_name", theme_name)
        self.settings.sync()

    def is_dark_mode(self):
        """Hack to detect if we have a dark mode running"""
        app = QApplication.instance()
        palette = app.palette()
        # Get the color of the window background
        background_color = palette.color(QtGui.QPalette.Window)
        # Calculate the luminance (brightness) of the color
        luminance = (
            0.299 * background_color.red() + 0.587 * background_color.green() + 0.114 * background_color.blue()
        ) / 255
        # If the luminance is below a certain threshold, it's considered dark mode
        return luminance < 0.5

    def applyTheme(self, theme_name):
        if "dark" in theme_name:
            self.set_dark()
        elif "light" in theme_name:
            self.set_light()
        elif self.is_dark_mode():
            self.set_dark()
        else:
            self.set_light()

        app = QApplication.instance()
        app.setStyleSheet("")  # resets style
        if theme_name in QtWidgets.QStyleFactory.keys():
            app.setStyle(theme_name)
        else:
            if theme_name == "Qdt light":
                qdarktheme.setup_theme("light")
            elif theme_name == "Qdt dark":
                qdarktheme.setup_theme("dark")
        
        self.resetActionIcons()

    def set_light(self):
        QIcon.setThemeName("light")
        self.editor.darkmode = False
        self.editor.logger.info("Resetting theme for light theme")

    def set_dark(self):
        QIcon.setThemeName("dark")
        self.editor.darkmode = True
        self.editor.logger.info("Resetting theme for dark theme")

    def openUrl(self):
        url = self.sender().data()
        QDesktopServices.openUrl(QUrl(url))

    def set_setting(self):
        action = self.sender()
        if isinstance(action, QAction):
            setting_name = action.objectName()
            if hasattr(self.editor, setting_name):
                if getattr(self.editor, setting_name) != action.isChecked():
                    setattr(self.editor, setting_name, action.isChecked())
                    self.editor.logger.error(f"Changed editor setting {setting_name} to {action.isChecked()}")
                    self.settings.setValue(setting_name, action.isChecked())
                    self.settings.sync()
            else:
                self.editor.logger.error(f"Error, could not find setting, {setting_name}, on editor object!")

    # Function to create actions for menus and toolbars
    def CreateActions(self):
        self.openAction = QAction(
            QIcon.fromTheme("open"),
            "O&pen",
            self,
            shortcut=QKeySequence.Open,
            statusTip="Open an existing file",
            triggered=self.openFile,
        )

        self.saveAction = QAction(
            QIcon.fromTheme("icons8-Save"),
            "S&ave",
            self,
            shortcut=QKeySequence.Save,
            statusTip="Save file",
            triggered=self.saveFile,
        )

        self.saveAsAction = QAction(
            QIcon.fromTheme("icons8-Save as"),
            "Save As",
            self,
            shortcut=QKeySequence.SaveAs,
            statusTip="Save file as ..",
            triggered=self.saveAsFile,
        )

        self.exitAction = QAction(
            QIcon.fromTheme("icons8-Shutdown"),
            "E&xit",
            self,
            shortcut="Esc",
            statusTip="Exit the Application",
            triggered=self.exitFile,
        )

        """self.aboutAction = QAction(
            QIcon.fromTheme("about"),
            "A&bout",
            self,
            statusTip="Displays info about text editor",
            triggered=self.aboutHelp,
        )"""

        self.aboutQtAction = QAction(
            "About &Qt",
            self,
            statusTip="Show the Qt library's About box",
            triggered=QApplication.aboutQt,
        )

        self.open3DmolAction = QAction(
            QIcon.fromTheme("icons8-Molecule"),
            "O&pen 3D view",
            self,
            shortcut=QKeySequence.Open,
            statusTip="Open 3D molecule view",
            triggered=self.open3Dmol,
        )

        # Copy-Paste actions
        self.copyAction = QAction(
            QIcon.fromTheme("icons8-copy-96"),
            "Copy SMILES",
            self,
            shortcut=QKeySequence.Copy,
            statusTip="Copy the current molecule as a SMILES string",
            triggered=self.copy,
        )

        self.pasteAction = QAction(
            QIcon.fromTheme("icons8-paste-100"),
            "Paste SMILES",
            self,
            shortcut=QKeySequence.Paste,
            statusTip="Paste the clipboard and parse assuming it is a SMILES string",
            triggered=self.paste,
        )

        # Edit actions
        self.actionActionGroup = QtGui.QActionGroup(self, exclusive=True)
        self.selectAction = QAction(
            QIcon.fromTheme("icons8-Cursor"),
            "Se&lect",
            self,
            shortcut="Ctrl+L",
            statusTip="Select Atoms",
            triggered=self.setAction,
            objectName="Select",
            checkable=True,
        )
        self.actionActionGroup.addAction(self.selectAction)
        
        self.cutAction1 = QAction(
            QIcon.fromTheme("icons8-scissors-pink"),
            "Cut &1",
            self,
            shortcut="Ctrl+Z",
            statusTip="Select First Cut(s)",
            triggered=self.setAction,
            objectName="Cut1",
            checkable=True,
        )
        self.actionActionGroup.addAction(self.cutAction1)

        self.cutAction2 = QAction(
            QIcon.fromTheme("icons8-scissors-green"),
            "Cut &2",
            self,
            shortcut="Ctrl+X",
            statusTip="Select Second Cut(s)",
            triggered=self.setAction,
            objectName="Cut2",
            checkable=True,
        )
        self.actionActionGroup.addAction(self.cutAction2)

        self.applyAction = QAction(
            "Apply Cuts",
            self,
            shortcut="Ctrl+A",
            statusTip="Apply cuts to create fragments",
            triggered=self.applyCuts,
            objectName="Apply",
            checkable=False,
        )
        self.actionActionGroup.addAction(self.applyAction)

        self.calcAction = QAction(
            "Loading...",
            self,
            shortcut="Ctrl+D",
            statusTip="Calculate Hydrogen bond energies",
            triggered=self.dataBox.calculate_frag_energies,
            objectName="Calculate",
            checkable=False,
        )
        self.calcAction.setEnabled(False)
        self.actionActionGroup.addAction(self.calcAction)


        self.increaseChargeAction = QAction(
            QIcon.fromTheme("icons8-Increase Font"),
            "I&ncrease Charge",
            self,
            shortcut="Ctrl++",
            statusTip="Increase Atom Charge",
            triggered=self.setAction,
            objectName="Increase Charge",
            checkable=True,
        )
        self.actionActionGroup.addAction(self.increaseChargeAction)

        self.decreaseChargeAction = QAction(
            QIcon.fromTheme("icons8-Decrease Font"),
            "D&ecrease Charge",
            self,
            shortcut="Ctrl+-",
            statusTip="Decrease Atom Charge",
            triggered=self.setAction,
            objectName="Decrease Charge",
            checkable=True,
        )
        self.actionActionGroup.addAction(self.decreaseChargeAction)

        self.numberAtom = QAction(
            QIcon.fromTheme("atommapnumber"),
            "Set atommap or R-group number",
            self,
            statusTip="Set atommap or R-group number",
            triggered=self.setAction,
            objectName="Number Atom",
            checkable=True,
        )
        self.actionActionGroup.addAction(self.numberAtom)
        self.selectAction.setChecked(True)

        self.rotateL = QAction(
            QIcon.fromTheme("icons8-Left"),
            "Rotate Left",
            self,
            statusTip="Rotate Left",
            triggered=self.rotate,
            objectName="Left",
        )

        self.rotateU = QAction(
            QIcon.fromTheme("icons8-Up"),
            "Rotate Up",
            self,
            statusTip="Rotate Up",
            triggered=self.rotate,
            objectName="Up",
        )

        self.rotateD = QAction(
            QIcon.fromTheme("icons8-Down"),
            "Rotate Down",
            self,
            statusTip="Rotate Down",
            triggered=self.rotate,
            objectName="Down",
        )

        self.rotateR = QAction(
            QIcon.fromTheme("icons8-Right"),
            "Rotate Right",
            self,
            statusTip="Rotate Right",
            triggered=self.rotate,
            objectName="Right",
        )

        # Misc Actions

        self.clearCanvasAction = QAction(
            QIcon.fromTheme("icons8-Trash"),
            "C&lear Canvas",
            self,
            shortcut="Ctrl+R",
            statusTip="Clear Canvas (no warning)",
            triggered=self.clearCanvas,
            objectName="Clear Canvas",
        )

        self.showHsAction = QAction(
            QIcon.fromTheme("icons8-Hydrogen"),
            "Show/hide all hydrogens",
            self,
            shortcut="Ctrl+H",
            statusTip="Show/hide all hydrogens",
            triggered=self.showHydrogens,
            checkable=True,
            objectName="'Hydrogens'",
        )

        self.cleanCoordinatesAction = QAction(
            QIcon.fromTheme("RecalcCoord"),
            "Recalculate coordinates &F",
            self,
            shortcut="Ctrl+F",
            statusTip="Re-calculates coordinates and redraw",
            triggered=self.flatten,
            checkable=True,
            objectName="Recalculate Coordinates",
        )

        self.cleanupSettingActions = {}

        self.loglevelactions = {}
        self.loglevelActionGroup = QtGui.QActionGroup(self, exclusive=True)
        for key in self.loglevels:
            self.loglevelactions[key] = QAction(
                key,
                self,
                statusTip="Set logging level to %s" % key,
                triggered=self.setLogLevel,
                objectName="loglevel:%s" % key,
                checkable=True,
            )
            self.loglevelActionGroup.addAction(self.loglevelactions[key])

        self.drawOptionsActions = {}
        for key, statusTip in self._drawopts_actions:
            self.drawOptionsActions[key] = QAction(
                key, self, statusTip=statusTip, triggered=self.setDrawOption, objectName=key, checkable=True
            )
            # self.drawOptionsActionGroup.addAction(self.drawOptionsActions[key])

        """self.openChemRxiv = QAction(
            QIcon.fromTheme("icons8-Exit"),
            "ChemRxiv Preprint",
            self,
            # shortcut="Ctrl+F",
            statusTip="Opens the ChemRxiv preprint",
            triggered=self.openUrl,
            data="https://doi.org/10.26434/chemrxiv-2024-jfhmw",
        )"""

        self.openRepository = QAction(
            QIcon.fromTheme("icons8-Exit"),
            "GitHub repository",
            self,
            # shortcut="Ctrl+F",
            statusTip="Opens the GitHub repository",
            triggered=self.openUrl,
            data="https://github.com/jokahara/MTAeditor",
        )

        self.exitAction = QAction(
            QIcon.fromTheme("icons8-Shutdown"),
            "E&xit",
            self,
            shortcut="Ctrl+Q",
            statusTip="Exit the Application",
            triggered=self.exitFile,
        )

def launch(loglevel="WARNING"):
    "Function that launches the mainWindow Application"
    # Exception Handling
    try:
        myApp = QApplication(sys.argv)
        if len(sys.argv) > 1:
            mainWindow = MainWindow(fileName=sys.argv[1], loglevel=loglevel)
        else:
            mainWindow = MainWindow(loglevel=loglevel)
        myApp.exec()
        sys.exit(0)
    except NameError:
        print("Name Error:", sys.exc_info()[1])
    except SystemExit:
        print("Closing Window...")
    except Exception:
        print(sys.exc_info()[1])


if __name__ == "__main__":
    launch(loglevel="DEBUG")
