#!/usr/bin/env python


#############################################################################
##
## Copyright (C) 2013 Riverbank Computing Limited.
## Copyright (C) 2010 Nokia Corporation and/or its subsidiary(-ies).
## All rights reserved.
##
## This file is part of the examples of PyQt.
##
## $QT_BEGIN_LICENSE:BSD$
## You may use this file under the terms of the BSD license as follows:
##
## "Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are
## met:
##   * Redistributions of source code must retain the above copyright
##     notice, this list of conditions and the following disclaimer.
##   * Redistributions in binary form must reproduce the above copyright
##     notice, this list of conditions and the following disclaimer in
##     the documentation and/or other materials provided with the
##     distribution.
##   * Neither the name of Nokia Corporation and its Subsidiary(-ies) nor
##     the names of its contributors may be used to endorse or promote
##     products derived from this software without specific prior written
##     permission.
##
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
## "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
## LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
## A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
## OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
## SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
## LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
## DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
## THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
## (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
## OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
## $QT_END_LICENSE$
##
#############################################################################

#############################################################################
# Reference: https://github.com/baoboa/pyqt5/blob/master/examples/widgets/imageviewer.py
#            https://stackoverflow.com/questions/36768033/pyqt-how-to-open-new-window
# Date: 3/09/21
#
# Last Changed:
# Author: Michael Maring
###############################################################################

from PyQt5.QtCore import QDir, Qt, QPoint
from PyQt5.QtGui import QImage, QPainter, QPalette, QPixmap, QPen, QFont
from PyQt5.QtWidgets import (QAction, QApplication, QFileDialog, QLabel,
        QMainWindow, QMenu, QMessageBox, QScrollArea, QSizePolicy)
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
                qp.drawText(self.cell_centers[0, idx], self.cell_centers[1, idx], "{}".format(self.solution[j, i]))

        # Draw Initial Conditions
        else:
            for idx in range(81):
                j = idx % 9
                i = (idx - j) // 9
                k = self.initial_conds_index[i, j]
                qp.drawText(self.cell_centers[0, idx], self.cell_centers[1, idx], "{}".format(self.initial_conds[j, i, k]))

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

        self.createActions()
        self.createMenus()

        self.setWindowTitle("Image Viewer")
        self.resize(500, 400)

        # top left, top right, bottom left, bottom right
        self.corners = []    # corners of sudoku puzzle

        # status checks
        self.finding_points = False
        self.adjust_initial_conditions = False
        self.solving_problem = False

        # data members
        self.num_points = 4
        self.side_length = 400        # sudoku puzzle side length in pixels (arbitrary choice)
        self.image = None

    def open(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File",
                                                  QDir.currentPath())
        if fileName:
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

    def about(self):
        QMessageBox.about(self, "About Image Viewer",
                          "<p>The <b>Image Viewer</b> example shows how to combine "
                          "QLabel and QScrollArea to display an image. QLabel is "
                          "typically used for displaying text, but it can also display "
                          "an image. QScrollArea provides a scrolling view around "
                          "another widget. If the child widget exceeds the size of the "
                          "frame, QScrollArea automatically provides scroll bars.</p>"
                          "<p>The example demonstrates how QLabel's ability to scale "
                          "its contents (QLabel.scaledContents), and QScrollArea's "
                          "ability to automatically resize its contents "
                          "(QScrollArea.widgetResizable), can be used to implement "
                          "zooming and scaling features.</p>"
                          "<p>In addition the example shows how to use QPainter to "
                          "print an image.</p>")


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

        self.aboutAct = QAction("&About", self, triggered=self.about)

        self.aboutQtAct = QAction("About &Qt", self,
                                  triggered=QApplication.instance().aboutQt)

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
        self.helpMenu.addAction(self.aboutAct)
        self.helpMenu.addAction(self.aboutQtAct)

        self.actMenu = QMenu("Actions", self)
        self.actMenu.addAction(self.findCorners)
        self.actMenu.addAction(self.findInitialConditions)
        self.actMenu.addAction(self.toggleDisplay)
        self.actMenu.addAction(self.solvePuzzle)

        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.viewMenu)
        self.menuBar().addMenu(self.actMenu)
        self.menuBar().addMenu(self.helpMenu)


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
            self.imageLabel.initial_conds_index[i, j] += 1
        else:
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

    def displayText(self):
        pass

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
        self.imageLabel.cell_centers = np.round(P @ X_centers)[:2, :].astype(int)

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
        model.load_state_dict(torch.load('model/mnist_model_epoch_15.pth'))

        # evaluate each cell of the sudoku 9x9 array
        mytens = torch.from_numpy(myarray)                                  # shape (81, 1, 28, 28)
        preds = model(mytens.to(device)).squeeze().detach().cpu().numpy()   # shape (81, 10)

        # Predictions that have confidence below threshold are considered empty
        confidence_thresh = 0.35       # prediction probability threshold
        initial_conds = np.argmax(preds, axis=1)
        confidence_max = np.max(preds, axis=1)
        print(np.where(confidence_max < confidence_thresh, 0, initial_conds).reshape(9, 9))

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

    def solvePuzzle(self):
        solver = sudoku_linprog()

        iv, jv = np.meshgrid(np.arange(9), np.arange(9), indexing='ij')
        kv = self.imageLabel.initial_conds_index
        initial_problem = np.zeros((9, 9))

        for idx in range(81):
            j = idx % 9
            i = (idx - j) // 9

            l = iv[i, j]
            m = jv[i, j]
            n = kv[i, j]
            initial_problem[i, j] = self.imageLabel.initial_conds[l, m, n]


        if self.imageLabel.initial_conds is not None:
            solver.solve(initial_problem.astype(int))

        # display solution
        self.imageLabel.solution = solver.solution
        self.imageLabel.display_initial_conds = False
        self.imageLabel.display_solution = True
        self.imageLabel.update()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    imageViewer = ImageViewer()
    imageViewer.show()
    sys.exit(app.exec_())
