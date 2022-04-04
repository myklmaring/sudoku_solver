#############################################################################
# PyQt5 image viewer reference:
# https://github.com/baoboa/pyqt5/blob/master/examples/widgets/imageviewer.py
# Date: 3/09/21
#
# Last Changed:
# Author: Michael Maring
###############################################################################

from PyQt5.QtCore import QDir, Qt, QPoint
from PyQt5.QtGui import QImage, QPainter, QPalette, QPixmap, QPen, QFont
from PyQt5.QtWidgets import (QAction, QApplication, QFileDialog, QLabel,
        QMainWindow, QMenu, QMessageBox, QScrollArea, QSizePolicy, QStatusBar)
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter

import numpy as np
import matplotlib.pyplot as plt
import sys
import math
import torch
import pickle as pkl
from PIL import Image
from digit_recognition import MyNetwork, ConvLayer
from scipy.ndimage.filters import gaussian_filter
from solve_problem import sudoku_linprog


class Label(QLabel):
    def __init__(self):
        super().__init__()
        self.scaleFactor = 1.0
        self.display = False
        self.display_initial_conds = False
        self.display_solution = False
        self.cell_centers = None
        self.initial_conds = None
        self.initial_conds_index = np.zeros((9, 9)).astype(int)
        self.solution = None

    def paintEvent(self, e):
        if self.pixmap():
            qp = QPainter()
            qp.begin(self)
            qp.drawPixmap(QPoint(), self.pixmap().scaled(self.scaleFactor * self.pixmap().size(), Qt.KeepAspectRatio))
            qp.end()

        if self.display and self.display_initial_conds and self.initial_conds is not None:
            self.draw_centers(0)

        if self.display and self.display_solution and self.solution is not None:
            self.draw_centers(1)

    def resizeEvent(self, e):
        if self.pixmap():
            self.scaleFactor = e.size().height()/self.pixmap().size().height()

    def draw_centers(self, mode):
        qp = QPainter()
        qp.begin(self)

        pen = QPen(Qt.red)
        pen.setWidth(1)
        qp.setPen(pen)

        font = QFont()
        font.setFamily('Times')
        font.setBold(True)
        font.setPointSize(16)
        qp.setFont(font)

        # Draw Solution
        if mode:
            for idx in range(81):
                j = idx % 9
                i = (idx - j) // 9
                qp.drawText(self.scaleFactor * self.cell_centers[0, idx], self.scaleFactor * self.cell_centers[1, idx],
                            "{}".format(self.solution[i, j]))

        # Draw Initial Conditions
        else:
            for idx in range(81):
                j = idx % 9
                i = (idx - j) // 9
                k = self.initial_conds_index[i, j]
                qp.drawText(self.scaleFactor * self.cell_centers[0, idx], self.scaleFactor * self.cell_centers[1, idx],
                            "{}".format(self.initial_conds[i, j, k]))

        qp.end()


class ImageViewer(QMainWindow):
    def __init__(self):
        super(ImageViewer, self).__init__()

        self.printer = QPrinter()
        self.scaleFactor = 0.0

        self.imageLabel = Label()
        self.imageLabel.setBackgroundRole(QPalette.Base)
        self.imageLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.imageLabel.setScaledContents(True)

        self.scrollArea = QScrollArea()
        self.scrollArea.setBackgroundRole(QPalette.Dark)
        self.scrollArea.setWidget(self.imageLabel)
        self.setCentralWidget(self.scrollArea)
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.message = QLabel()
        self.message.setText("Open a Puzzle")
        self.statusBar.addWidget(self.message)

        self.createActions()
        self.createMenus()

        self.setWindowTitle("Sudoku Solver")
        self.resize(500, 400)

        # top left, top right, bottom left, bottom right
        self.corners = []    # corners of sudoku puzzle

        # status checks
        self.finding_points = False
        self.solving_problem = False

        # data members
        self.num_points = 4
        self.side_length = 400        # sudoku puzzle side length in pixels (arbitrary choice)
        self.image = None
        self.solver = sudoku_linprog()

    def open(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File",
                                                  QDir.currentPath())
        if fileName:

            # reset imageLabel, and ImageViewer attributes to original values to allow for serial use
            self.finding_points = False
            self.solving_problem = False
            self.imageLabel.cell_centers = None
            self.imageLabel.initial_conds = None
            self.imageLabel.solution = None
            self.imageLabel.display = False
            self.imageLabel.display_initial_conds = False
            self.imageLabel.display_solution = False

            self.image = QImage(fileName)
            if self.image.isNull():
                QMessageBox.information(self, "Image Viewer",
                                        "Cannot load %s." % fileName)
                return

            self.imageLabel.setPixmap(QPixmap.fromImage(self.image))
            self.scaleFactor = 1.0

            self.printAct.setEnabled(True)
            self.fitToWindowAct.setEnabled(True)
            self.updateActions()

            if not self.fitToWindowAct.isChecked():
                self.imageLabel.adjustSize()



    def print_(self):
        dialog = QPrintDialog(self.printer, self)
        if dialog.exec_():
            painter = QPainter(self.printer)
            rect = painter.viewport()
            size = self.imageLabel.pixmap().size()
            size.scale(rect.size(), Qt.KeepAspectRatio)
            painter.setViewport(rect.x(), rect.y(), size.width(), size.height())
            painter.setWindow(self.imageLabel.pixmap().rect())
            painter.drawPixmap(0, 0, self.imageLabel.pixmap())

    def zoomIn(self):
        self.scaleImage(1.25)

    def zoomOut(self):
        self.scaleImage(0.8)

    def normalSize(self):
        self.imageLabel.adjustSize()
        self.scaleFactor = 1.0

    def fitToWindow(self):
        fitToWindow = self.fitToWindowAct.isChecked()
        self.scrollArea.setWidgetResizable(fitToWindow)
        if not fitToWindow:
            self.normalSize()
        self.updateActions()

    def instructions(self):
        QMessageBox.about(self,
        '',
        "<p> Instructions: "
        "<p> 1. In the File menu tab open sudoku image using the Open action (Ctrl+O)."
        "<p> 2. In the Actions menu tab select the Find Corners action (Ctrl+C).  Left mouse click on the corners of "
            "the sudoku puzzle in order from top left, top right, bottom left, bottom right"
        "<p> 3. In the Actions menu tab select the Determine Digits action (Ctrl+D).  This will automatically determine "
            "which digits are in the cell, if any.  Emtpy cells will be denoted with a zero.  Left mouse click sudoku "
            "cell to increment the prediction for the cell. Right mouse click to return to the previous prediction. "
        "<p> (Optional) In the Actions menu tab select Toggle Number Display (Ctrl+X) to stop/start displaying digits"
        "<p> 4. In the Actions menu tab select Solve Sudoku Puzzle (Ctrl+Z) to solve for the missing numbers. "
            "It will automatically display the correct digits in their corresponding locations if the solve was "
            "successful. Otherwise, a message will pop up in the status bar at the bottom letting you know that "
            "the solve failed; prompting you to go alter the initial conditions so that they are all correct.")


    def createActions(self):
        self.openAct = QAction("&Open...", self, shortcut="Ctrl+O",
                               triggered=self.open)

        self.printAct = QAction("&Print...", self, shortcut="Ctrl+P",
                                enabled=False, triggered=self.print_)

        self.exitAct = QAction("E&xit", self, shortcut="Ctrl+Q",
                               triggered=self.close)

        self.zoomInAct = QAction("Zoom &In (25%)", self, shortcut="Ctrl+=",
                                 enabled=False, triggered=self.zoomIn)

        self.zoomOutAct = QAction("Zoom &Out (25%)", self, shortcut="Ctrl+-",
                                  enabled=False, triggered=self.zoomOut)

        self.normalSizeAct = QAction("&Normal Size", self, shortcut="Ctrl+S",
                                     enabled=False, triggered=self.normalSize)

        self.fitToWindowAct = QAction("&Fit to Window", self, enabled=False,
                                      checkable=True, shortcut="Ctrl+F", triggered=self.fitToWindow)

        self.instructionsAct = QAction("&Use Instructions", self, triggered=self.instructions)

        self.findCorners = QAction("Find &Corners", self, shortcut="Ctrl+c",
                                   triggered=self.findCorners)

        self.findInitialConditions = QAction("Determine &Digits", self, shortcut="Ctrl+d",
                                             triggered=self.findInitialConditions)

        self.toggleDisplay = QAction("Toggle Number Display", self, shortcut="Ctrl+x",
                                               triggered=self.toggleDisplay)

        self.solvePuzzle = QAction("Solve Sudoku Puzzle", self, shortcut="Ctrl+z",
                                   triggered=self.solvePuzzle)

    def createMenus(self):
        self.fileMenu = QMenu("File", self)
        self.fileMenu.addAction(self.openAct)
        self.fileMenu.addAction(self.printAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAct)

        self.viewMenu = QMenu("View", self)
        self.viewMenu.addAction(self.zoomInAct)
        self.viewMenu.addAction(self.zoomOutAct)
        self.viewMenu.addAction(self.normalSizeAct)
        self.viewMenu.addSeparator()
        self.viewMenu.addAction(self.fitToWindowAct)

        self.helpMenu = QMenu("Help", self)
        self.helpMenu.addAction(self.instructionsAct)

        self.actMenu = QMenu("Actions", self)
        self.actMenu.addAction(self.findCorners)
        self.actMenu.addAction(self.findInitialConditions)
        self.actMenu.addAction(self.toggleDisplay)
        self.actMenu.addAction(self.solvePuzzle)

        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.viewMenu)
        self.menuBar().addMenu(self.actMenu)
        self.menuBar().addMenu(self.helpMenu)

        self.menuBar().triggered[QAction].connect(self.processtrigger)


    def processtrigger(self, q):
        """ create an event whenever a menu option is pressed and update the status bar """

        if (q.text() == "Find &Corners"):
            self.message.setText("Click on the Puzzle Corners in Order (Top Left, Top Right, Bottom Left, Bottom Right)")
            self.statusBar.addWidget(self.message)

        if (q.text() == "Determine &Digits"):
            self.message.setText("Check Initial Conditions.  Left/Right Mouse Click to Change Prediction")
            self.statusBar.addWidget(self.message)

        if (q.text() == "Solve Sudoku Puzzle"):
            if self.solver.solution is not None:
                self.message.setText("Solved Sudoku Puzzle")
                self.statusBar.addWidget(self.message)
            else:
                self.message.setText("Solver Failed. Recheck Initial Conditions")
                self.statusBar.addWidget(self.message)

            # self.statusBar.showMessage(q.text() + " is clicked", 2000)


    def updateActions(self):
        self.zoomInAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.zoomOutAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.normalSizeAct.setEnabled(not self.fitToWindowAct.isChecked())


    def scaleImage(self, factor):
        self.scaleFactor *= factor
        self.imageLabel.resize(self.scaleFactor * self.imageLabel.pixmap().size())

        self.adjustScrollBar(self.scrollArea.horizontalScrollBar(), factor)
        self.adjustScrollBar(self.scrollArea.verticalScrollBar(), factor)

        self.zoomInAct.setEnabled(self.scaleFactor < 3.0)
        self.zoomOutAct.setEnabled(self.scaleFactor > 0.333)


    def adjustScrollBar(self, scrollBar, factor):
        scrollBar.setValue(int(factor * scrollBar.value()
                               + ((factor - 1) * scrollBar.pageStep()/2)))

    def findAbsoluteCoordinate(self, event):
        # scale coordinate by the image scaling to get true coordinate
        return (1/self.scaleFactor) * self.imageLabel.mapFromGlobal(self.mapToGlobal(event.pos()))

    def incrementIndex(self, i, j, pos):
        if pos:
            if self.imageLabel.initial_conds_index[i, j] < 9:
                self.imageLabel.initial_conds_index[i, j] += 1
        else:
            if self.imageLabel.initial_conds_index[i, j] > 0:
                self.imageLabel.initial_conds_index[i, j] -= 1

    def closestPoint(self, point):
        mypoint = np.array([[point.x()], [point.y()]])
        diff = self.imageLabel.cell_centers - mypoint
        min_idx = np.argmin((diff ** 2).sum(axis=0))
        j = min_idx % 9
        i = (min_idx - j) // 9

        return i, j

    def mousePressEvent(self, event):

        if event.button() == Qt.LeftButton:

            # 4-point affine transformation for sudoku square
            if self.finding_points:
                if len(self.corners) < self.num_points:
                    mypoint = self.findAbsoluteCoordinate(event)
                    self.corners.append(mypoint)
                else:
                    self.finding_points = False

            # find closest sudoku cell location and increment initial condition index
            if self.imageLabel.display_initial_conds and self.imageLabel.cell_centers.any():
                i, j = self.closestPoint(self.findAbsoluteCoordinate(event))
                self.incrementIndex(i, j, True)
                self.imageLabel.update()

        if event.button() == Qt.RightButton:

            # find closest sudoku cell location and decrement initial condition index
            if self.imageLabel.display_initial_conds and self.imageLabel.cell_centers.any():
                i, j = self.closestPoint(self.findAbsoluteCoordinate(event))
                self.incrementIndex(i, j, False)
                self.imageLabel.update()

    def findCorners(self):
        self.corners = []
        self.finding_points = True

    def toggleDisplay(self):
        self.imageLabel.display = not self.imageLabel.display
        self.imageLabel.update()

    def findDistortion(self):
        """ take sudoku box corners and calculate the affine transformation for a square
            Equation: A = Bx

            Use x to define the affine matrix for the equation (Y = PX)

            P = [[x(0) x(1) x(2)],  Y = [[y(0)],    X = [[x(0)],
                 [x(3) x(4) x(5)],       [y(1)],         [x(1)],
                 [ 0    0    1  ]]       [  1 ]]         [  1 ]]

            P - affine matrix transformation [from new coordinate system to original coordinate system]
            X - pixel location in new coordinate system (new_image), i.e. the arbitrary square we define
            Y - pixel location in image coordinate system (image), i.e. the source image
        """

        # Matrix Coordinate System (i.e. 0,0 is top left, O,L is top right, etc.)
        p1 = (0, 0)                                 # Top Left
        p2 = (0, self.side_length)                  # Top Right
        p3 = (self.side_length, 0)                  # Bottom Left
        p4 = (self.side_length, self.side_length)   # Bottom Right

        # Note to self:
        # This derivation uses homogeneous coordinates
        # i.e. [x,y] <===> [u, v, 1]
        # also note that this coordinate system uses euclidean coordinates not row, col coordinates of matrices
        A = np.array([self.corners[0].x(), self.corners[0].y(), self.corners[1].x(), self.corners[1].y(),
                      self.corners[2].x(), self.corners[2].y(), self.corners[3].x(), self.corners[3].y()]).reshape(8, 1)

        B = np.array([[p1[0], p1[1], 1, 0, 0, 0],
                      [0, 0, 0, p1[0], p1[1], 1],
                      [p2[0], p2[1], 1, 0, 0, 0],
                      [0, 0, 0, p2[0], p2[1], 1],
                      [p3[0], p3[1], 1, 0, 0, 0],
                      [0, 0, 0, p3[0], p3[1], 1],
                      [p4[0], p4[1], 1, 0, 0, 0],
                      [0, 0, 0, p4[0], p4[1], 1]])

        x = np.linalg.pinv(B) @ A
        P = np.vstack((x.reshape(2, 3), np.array([0, 0, 1])))

        new_image = np.zeros((self.side_length, self.side_length, 3))
        xv, yv = np.meshgrid(np.arange(self.side_length), np.arange(self.side_length))
        X = np.vstack((xv.flatten(), yv.flatten(), np.ones(xv.size))).astype(int)
        Y = np.round(P @ X).astype(int)

        for i in range(X.shape[1]):
            new_image[X[0, i], X[1, i], :] = (
                self.image.pixelColor(Y[0, i], Y[1, i]).red(),
                self.image.pixelColor(Y[0, i], Y[1, i]).green(),
                self.image.pixelColor(Y[0, i], Y[1, i]).blue()
            )

        # find sudoku cell centers in image coordinate system
        a = np.linspace(self.side_length/18, self.side_length - self.side_length/18, 9)
        xv, yv = np.meshgrid(a, a)       # sudoku cell centers in new coordinate system, i.e. square
        X_centers = np.vstack((xv.flatten(), yv.flatten(), np.ones(xv.size))).astype(int)
        temp_centers = np.round(P @ X_centers)[:2, :].astype(int)

        # reshape temp_centers because points go down columns not across rows
        x = temp_centers[0, :].reshape(9, 9).T
        y = temp_centers[1, :].reshape(9, 9).T
        self.imageLabel.cell_centers = np.vstack((x.reshape(-1), y.reshape(-1)))


        return new_image

    def numpyToQimage(self, image):
        image = image.astype(np.uint8)
        return QImage(image, image.shape[0], image.shape[1], QImage.Format_Grayscale8)


    def qimageToNumpy(self, qimage):
        # reference https://stackoverflow.com/questions/19902183/qimage-to-numpy-array-using-pyside
        qimage = qimage.convertToFormat(Qimage.Format.Format_RGB888)
        width = qimage.width()
        height = qimage.height()

        temp = qimage.constBits()
        return np.array(temp).reshape(height, width, 3)


    def toGrayscale(self, image):
        cr, cg, cb = 0.3, 0.59, 0.11
        return cr * image[:, :, 0] + cg * image[:, :, 1] + cb * image[:, :, 2]


    def findInitialConditions(self):
        image = self.findDistortion()
        image = self.toGrayscale(image)

        # Get rid of cell borders
        target = 34
        n = (target-28)//2
        pimage = Image.fromarray(image)
        pimage = pimage.resize((target * 9, target * 9))
        myarray = np.array(pimage)

        # Gaussian smoothing
        myarray = gaussian_filter(myarray, sigma=1)

        # Reshape large image into stack of individual cell images
        myarray = myarray.reshape(9, target, 9, target)
        myarray = myarray.swapaxes(1, 2)
        myarray = myarray.reshape(-1, target, target)
        myarray = myarray[:, None, n:-n, n:-n]  # center crop the result to get 28x28 image

        # Enforce pixel values from 0-255
        img_min = myarray.min()
        img_max = myarray.max()
        mprecision = np.finfo(float).eps
        myarray = 255.0 * (myarray - img_min) / (img_max-img_min + mprecision)

        # MNIST images are supposed to be black background, white numbers
        background = np.mean(myarray[:, 0, :2, :2])

        # white background if true
        if background > 127.5:
            myarray = -1 * myarray + 255
            background = -1 * background + 255
        myarray -= background - mprecision

        # set erroneous negative pixel values to zero
        # This happens because the background pixels aren't the lowest values.  There happens to be
        # spurious pixels on the fringes of the digit that are lower.
        negative_inds = np.where(myarray < 0)
        myarray[negative_inds] = mprecision

        # center of mass calculation w/ thresholded pixel values (i.e. center of mass)
        pix_cutoff = 20
        ax = np.arange(28) + 1
        img_thresh = (myarray > pix_cutoff)
        downCols = img_thresh.squeeze().sum(axis=1) + np.finfo(float).eps
        downRows = img_thresh.squeeze().sum(axis=2) + np.finfo(float).eps
        col_c = (ax * downCols).sum(axis=1)/downCols.sum(axis=1) - 1          # back to python indexing
        row_c = (ax * downRows).sum(axis=1)/downRows.sum(axis=1) - 1
        com = np.vstack((row_c, col_c)) - 13.5                                # center of mass relative to image center
        shifts = -com.round().astype(int)                                     # shift COM to image center

        # center numbers in image
        for i in range(81):
            temp = myarray[i, 0, :, :]
            temp = np.roll(temp, (shifts[0, i], shifts[1, i]), axis=(0, 1))     # axis 0: U <-> D  axis 1: L <-> R
            myarray[i, 0, :, :] = temp

        # feed the sequence of sudoku cell images through the trained model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MyNetwork().to(device)
        model.load_state_dict(torch.load('model/mnist_model_epoch_15.pth', map_location=device))

        # evaluate each cell of the sudoku 9x9 array
        mytens = torch.from_numpy(myarray)                                  # shape (81, 1, 28, 28)
        preds = model(mytens.to(device)).squeeze().detach().cpu().numpy()   # shape (81, 10)

        # Predictions that have confidence below threshold are considered empty
        confidence_thresh = 0.35       # prediction probability threshold
        confidence_max = np.max(preds, axis=1)

        # Assign Initial Conditions to imageLabel for Display Purposes
        # (indices that return the sorted predictions for each sudoku cell)
        initial_conds_argsort = np.argsort(preds, axis=1)
        for i, val in enumerate(confidence_max):
            if val < confidence_thresh:
                initial_conds_argsort[i, :] = 0

        self.imageLabel.initial_conds = initial_conds_argsort[:, ::-1].reshape(9, 9, 10)

        # Display Initial Conditions
        self.imageLabel.display = True
        self.imageLabel.display_initial_conds = True
        self.imageLabel.update()

        # save myarray and predictions
        mydict = {'myarray': myarray, 'preds': preds}
        with open('images.pkl', 'wb') as f:
            pkl.dump(mydict, f)

    def solvePuzzle(self):
        initial_problem = np.zeros((9, 9)).astype(int)
        for idx in range(81):
            j = idx % 9
            i = (idx - j) // 9
            k = self.imageLabel.initial_conds_index[i, j]
            initial_problem[i, j] = self.imageLabel.initial_conds[i, j, k]

        if self.imageLabel.initial_conds is not None:
            self.solver.solve(initial_problem)

        # display solution
        if self.solver.solution is not None:
            self.imageLabel.solution = self.solver.solution
            self.imageLabel.display_initial_conds = False
            self.imageLabel.display_solution = True
            self.imageLabel.update()
        else:
            # solver couldn't find solution, update initial conditions to fix problem
            self.imageLabel.display_initial_conds = True

if __name__ == '__main__':
    app = QApplication(sys.argv)
    imageViewer = ImageViewer()
    imageViewer.show()
    sys.exit(app.exec_())
