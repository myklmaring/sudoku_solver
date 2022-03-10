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

from PyQt5.QtCore import QDir, Qt
from PyQt5.QtGui import QImage, QPainter, QPalette, QPixmap
from PyQt5.QtWidgets import (QAction, QApplication, QFileDialog, QLabel,
        QMainWindow, QMenu, QMessageBox, QScrollArea, QSizePolicy)
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter

import numpy as np
import matplotlib.pyplot as plt
import sys
from PIL import Image
from digit_recognition import MyNetwork, ConvLayer
import torch
import pickle as pkl
from scipy.ndimage.filters import gaussian_filter




class Digits(QMainWindow):
    def __init__(self, image, parent=None):
        super(Digits, self).__init__(parent)
        self.image = image  # QImage object
        self.imageLabel = QLabel(self)
        self.imageLabel.setScaledContents(True)
        self.resize(400, 400)
        self.setWindowTitle("Check Digits")
        self.pixmap = QPixmap.fromImage(self.image)
        self.imageLabel.setPixmap(self.pixmap)
        self.imageLabel.adjustSize()


class ImageViewer(QMainWindow):
    def __init__(self):
        super(ImageViewer, self).__init__()

        self.printer = QPrinter()
        self.scaleFactor = 0.0

        self.imageLabel = QLabel()
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
        self.finding_points = False
        self.found_points = False
        self.num_points = 4
        self.l = 400        # sudoku puzzle side length in pixels (arbitrary choice)
        self.image = None
        self.digits = None


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

        self.findPoints = QAction("Find &Points", self, shortcut="Ctrl+p",
                                  triggered=self.findPoints)

        self.digitRecognition = QAction("Determine &Digits", self, shortcut="Ctrl+d",
                                        triggered=self.digitRecognition)


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
        self.actMenu.addAction(self.findPoints)
        self.actMenu.addAction(self.digitRecognition)

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


    def mousePressEvent(self, event):

        if event.button() == Qt.LeftButton:

            #  check if finding points
            if self.finding_points:
                # check if you have points left to collect
                if len(self.corners) < self.num_points:
                    # scale coordinate by the image scaling to get true coordinate
                    mypoint = (1/self.scaleFactor) * self.imageLabel.mapFromGlobal(self.mapToGlobal(event.pos()))
                    self.corners.append(mypoint)
                else:
                    self.finding_points = False
                    self.found_points = True        # needed in order to progress to other steps


    def findPoints(self):
        self.corners = []
        self.finding_points = True


    def findDistortion(self):
        """ take sudoku box corners and calculate the affine transformation for a proper square"""

        p1 = (0,0)
        p2 = (0, self.l)
        p3 = (self.l, 0)
        p4 = (self.l, self.l)

        Y = np.array([self.corners[0].x(), self.corners[0].y(), self.corners[1].x(), self.corners[1].y(),
                      self.corners[2].x(), self.corners[2].y(), self.corners[3].x(), self.corners[3].y()]).reshape(8, 1)

        X = np.array([[p1[0], p1[1], 1, 0, 0, 0],
                      [0, 0, 0, p1[0], p1[1], 1],
                      [p2[0], p2[1], 1, 0, 0, 0],
                      [0, 0, 0, p2[0], p2[1], 1],
                      [p3[0], p3[1], 1, 0, 0, 0],
                      [0, 0, 0, p3[0], p3[1], 1],
                      [p4[0], p4[1], 1, 0, 0, 0],
                      [0, 0, 0, p4[0], p4[1], 1]])

        T = np.linalg.pinv(X) @ Y
        T = np.vstack((T.reshape(2,3), np.array([0, 0, 1])))

        new_image = np.zeros((self.l, self.l, 3))
        numbers = np.tile(np.arange(self.l)[:, None], self.l)
        rows = numbers.reshape(-1)
        cols = numbers.T.reshape(-1)
        coord_mat = np.array([cols, rows, np.ones_like(rows)])
        coord_t_mat = np.around(T @ coord_mat)[:2, :]
        coord_mat = coord_mat[:2, :]

        for i in range(coord_mat.shape[1]):
            new_image[coord_mat[0,i], coord_mat[1,i], :] = (
                self.image.pixelColor(coord_t_mat[0,i], coord_t_mat[1,i]).red(),
                self.image.pixelColor(coord_t_mat[0, i], coord_t_mat[1, i]).green(),
                self.image.pixelColor(coord_t_mat[0, i], coord_t_mat[1, i]).blue()
            )

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


    def digitRecognition(self):
        image = self.findDistortion()
        image = self.toGrayscale(image)

        # Display the cropped image
        im = self.numpyToQimage(image)
        self.digits = Digits(im, self)
        self.digits.show()

        # Get rid of cell borders
        target = 34
        n = (target-28)//2
        pimage = Image.fromarray(image)
        pimage = pimage.resize((target * 9, target * 9))
        myarray = np.array(pimage)

        # Gaussian smoothing
        myarray = gaussian_filter(myarray, sigma=0.01)

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

        myarray -= background

        ## Make background value zero
        # img_flat = myarray.squeeze().reshape(81, -1)
        # img_flat = 255.0 * (img_flat - background) / \
        #     (img_flat.max(axis=1)-background + np.finfo(float).eps)[:, None]
        # myarray = img_flat.reshape(81, 1, 28, 28)
        # myarray[np.nonzero(myarray < 0)] = 0

        # # find bounding box containing number in each cell  (bbox didn't word sufficiently)
        # x = 4
        # bg_val = np.hstack((myarray[:, 0, 0:x, 0:x].reshape(81, -1),
        #                     myarray[:, 0, 0:x, -x:-1].reshape(81, -1),
        #                     myarray[:, 0, -x:-1, 0:x].reshape(81, -1),
        #                     myarray[:, 0, -x:-1, -x:-1].reshape(81, -1)))     # check background vals in each corner
        #
        # bg_val_mean = np.mean(bg_val, axis=1)[:, None, None, None]            # must have same num. of dimensions as myarray
        # bg_val_std = np.std(bg_val, axis=1)[:, None, None, None]
        # inds = np.argwhere(myarray > (3 * bg_val_std + bg_val_mean))          # shape (N x 4), when N is num of nonzero values
        #
        # bbox = np.zeros((81, 4))
        # for i in range(81):
        #     img_inds = inds[np.nonzero(inds[:, 0] == i)]
        #
        #     # slice index max is not inclusive so we add 1
        #     # img_inds could be empty if image doesn't contain a digit
        #     if img_inds.any():
        #         bbox[i, 0] = np.max(img_inds[:, 2]) + 1     # row max
        #         bbox[i, 1] = np.min(img_inds[:, 2])         # row min
        #         bbox[i, 2] = np.max(img_inds[:, 3]) + 1     # col max
        #         bbox[i, 3] = np.min(img_inds[:, 3])         # col min
        #
        # # find bbox centers
        # bbox_centers = np.column_stack(((bbox[:, 0] + bbox[:, 1]) // 2, (bbox[:, 2] + bbox[:, 3]) // 2))
        # bbox_centers = np.where(bbox_centers == 0, 13, bbox_centers)    #  if no bbox was found, replace center with img
        #                                                                 #  center so it is not shifted later
        # shifts = bbox_centers - 13      # images are 28x28 so center is (14, 14) or 13 in python indexing
        # shifts = shifts.astype(int)      # enforce integer shifts

        # # center of mass calculation w/ thresholding
        # ax = np.arange(28) + 1
        # img_thresh = (myarray > (3 * bg_val_std + bg_val_mean)).squeeze()     # (img, row, col), i.e. (81, 28, 28)
        # colsum = img_thresh.sum(axis=1) + np.finfo(float).eps
        # rowsum = img_thresh.sum(axis=2) + np.finfo(float).eps
        # x_c = (ax * colsum).sum(axis=1)/colsum.sum(axis=1) - 1                # back to python indexing
        # y_c = (ax * rowsum).sum(axis=1)/rowsum.sum(axis=1) - 1
        # shifts = np.vstack((y_c, x_c)) - 13.5
        # shifts = shifts.round().astype(int)      # enforce integer shifts

        # center of mass calculation w/ pixel values (i.e. weighted COM)
        ax = np.arange(28) + 1
        colsum = myarray.squeeze().sum(axis=1) + np.finfo(float).eps
        rowsum = myarray.squeeze().sum(axis=2) + np.finfo(float).eps
        x_c = (ax * colsum).sum(axis=1)/colsum.sum(axis=1) - 1                # back to python indexing
        y_c = (ax * rowsum).sum(axis=1)/rowsum.sum(axis=1) - 1
        com = np.vstack((y_c, x_c)) - 13.5                                    # center of mass relative to image center
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
        thresh = 0.45       # prediction probability threshold
        a = np.argmax(preds, axis=1)
        b = np.max(preds, axis=1)
        a = np.where(b < thresh, 0, a)
        print(a.reshape(9, 9))

        mydict = {'myarray': myarray, 'preds': preds}
        with open('images.pkl', 'wb') as f:
            pkl.dump(mydict, f)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    imageViewer = ImageViewer()
    imageViewer.show()
    sys.exit(app.exec_())
