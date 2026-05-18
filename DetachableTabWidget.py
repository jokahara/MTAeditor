from PySide6 import QtGui, QtCore
from PySide6.QtCore import Qt, QEvent, QPoint, QMimeData, Signal, Slot
from PySide6.QtWidgets import QWidget, QTabWidget, QDialog, QVBoxLayout, QTabBar, QApplication, QMainWindow, QLabel
from PySide6.QtGui import QCursor,QMouseEvent

# The DetachableTabWidget adds additional functionality to Qt's QTabWidget that allows it
# to detach and re-attach tabs.
class DetachableTabWidget(QTabWidget):
    def __init__(self, parent=None):
        QTabWidget.__init__(self, parent)

        self.tabBar = self.TabBar(self)
        self.tabBar.onDetachTabSignal.connect(self.detachTab)
        self.tabBar.onMoveTabSignal.connect(self.moveTab)
        
        self.setTabBar(self.tabBar)
        self.setMovable(True)
        self.setAcceptDrops(True)

    # Move a tab from one position (index) to another
    @Slot(int, int)
    def moveTab(self, fromIndex, toIndex):
        widget = self.widget(fromIndex)
        icon = self.tabIcon(fromIndex)
        text = self.tabText(fromIndex)

        self.removeTab(fromIndex)
        self.insertTab(toIndex, widget, icon, text)
        self.setCurrentIndex(toIndex)

    # Detach the tab by removing it's contents and placing them in
    # a DetachedTab dialog
    @Slot(int, QPoint)
    def detachTab(self, index, point):

        self.blockSignals(True)
        # Get the tab content
        name = self.tabText(index)
        widget = self.widget(index)

        # Create a new detached tab window
        detachedTab = self.DetachedTab(widget, self.parentWidget(), name)
        detachedTab.onCloseSignal.connect(self.attachTab)
        detachedTab.move(point)
        #print('Detach Tab', name)
        detachedTab.show()
        
        self.blockSignals(False)

    # Re-attach the tab by removing the content from the DetachedTab dialog,
    # closing it, and placing the content back into the DetachableTabWidget
    def attachTab(self, widget, name):

        # Make the content widget a child of this widget
        widget.setParent(self)
        index = self.addTab(widget, name)

        # Make this tab the current tab
        if index > -1:
            self.setCurrentWidget(widget)
            #self.setCurrentIndex(index)


    # When a tab is detached, the contents are placed into this QDialog.  The tab
    # can be re-attached by closing the dialog or by double clicking on its window frame.
    class DetachedTab(QDialog):
        onCloseSignal = Signal(QWidget, str)

        def __init__(self, widget, parent=None, name=''):
            super().__init__(parent)
            self.setWindowFlags(Qt.WindowType.Window)

            self.setWindowTitle(name)
            self.setObjectName(name)
            self.setGeometry(widget.frameGeometry())

            layout = QVBoxLayout(self)            
            self.widget = widget            
            layout.addWidget(self.widget)
            self.widget.show()

        def keyPressEvent(self, event):
            if event.key() == Qt.Key.Key_Escape:
                event.accept()
                self.close()
                
            return super().keyPressEvent(event)
        
        def closeEvent(self, event):
            self.onCloseSignal.emit(self.widget, self.objectName())

        def dragEnterEvent(self, event):
            return super().dragEnterEvent(event)
        
        def mouseMoveEvent(self, event):
            return super().mouseMoveEvent(event)

    # The TabBar class re-implements some of the functionality of the QTabBar widget
    class TabBar(QTabBar):
        onDetachTabSignal = Signal(int, QPoint)
        onMoveTabSignal = Signal(int, int)

        def __init__(self, parent=None):
            super().__init__(parent)

            self.setAcceptDrops(True)
            self.setElideMode(Qt.TextElideMode.ElideRight)
            self.setSelectionBehaviorOnRemove(QTabBar.SelectionBehavior.SelectLeftTab)

            self.dragStartPos = QPoint()
            self.dragDropedPos = QPoint()
            self.mouseCursor = QCursor()
            self.dragInitiated = False

        # Send the onDetachTabSignal when a tab is double clicked
        def mouseDoubleClickEvent(self, event: QMouseEvent):
            event.accept()
            self.onDetachTabSignal.emit(self.tabAt(event.position().toPoint()), self.mouseCursor.pos())


        # Set the starting position for a drag event when the mouse button is pressed
        def mousePressEvent(self, event: QMouseEvent):
            if event.button() == Qt.MouseButton.LeftButton:
                self.dragStartPos = event.position().toPoint()

            self.dragDropedPos.setX(0)
            self.dragDropedPos.setY(0)

            self.dragInitiated = False

            super().mousePressEvent(event)


        # Determine if the current movement is a drag.  If it is, convert it into a QDrag.  If the
        # drag ends inside the tab bar, emit an onMoveTabSignal.  If the drag ends outside the tab
        # bar, emit an onDetachTabSignal.
        def mouseMoveEvent(self, event: QMouseEvent):

            distance = (
                event.position().toPoint() - self.dragStartPos
            ).manhattanLength()

            if not self.dragStartPos.isNull() and distance < QApplication.startDragDistance():
                self.dragInitiated = True

            if (((event.buttons() & Qt.MouseButton.LeftButton)) and self.dragInitiated):
                if distance < QApplication.startDragDistance():
                    return

                index = self.tabAt(self.dragStartPos)
                if index < 0:
                    return
                
                # If cursor moved outside tabbar → detach
                if not self.rect().contains(event.position().toPoint()):
                    event.accept()
                    #print('Detach')
                    self.onDetachTabSignal.emit(self.tabAt(self.dragStartPos), self.mouseCursor.pos())
                    self.dragInitiated = False
                    finishMoveEvent = QtGui.QMouseEvent(QEvent.Type.MouseMove, event.position(), 
                                                    Qt.MouseButton.NoButton, Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
                    super().mouseMoveEvent(finishMoveEvent)
                else:
                    #print('Drag')
                    super().mouseMoveEvent(event)
        
                """
                # Convert the move event into a drag
                drag = QtGui.QDrag(self)
                mimeData = QtCore.QMimeData()
                drag.setMimeData(mimeData)"""


        # Get the position of the end of the drag
        def dropEvent(self, event):
            self.dragDropedPos = event.position()
            super().dropEvent(event)


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)

    mainWindow = QMainWindow()
    tabWidget = DetachableTabWidget(mainWindow)

    tab1 = QLabel('Test Widget 1')    
    tabWidget.addTab(tab1, 'Tab1')

    tab2 = QLabel('Test Widget 2')
    tabWidget.addTab(tab2, 'Tab2')

    tab3 = QLabel('Test Widget 3')
    tabWidget.addTab(tab3, 'Tab3')

    tabWidget.show()
    mainWindow.setCentralWidget(tabWidget)
    mainWindow.show()

    try:
        exitStatus = app.exec()
        print('Done...')
        sys.exit(exitStatus)
    except:
        pass