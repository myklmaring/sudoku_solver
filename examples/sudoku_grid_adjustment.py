"""
This is an example of using left/right mouse button to increase/decrease
the displayed number in a sudoku board.  The coordinates for this particular
sudoku board are hard coded.
"""


import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import numpy as np

class Label(QLabel):
    def __init__(self, image, points):
        super().__init__()
        self.index = np.zeros((9, 9)).astype(int)
        self.image = image
        self.points = points
        rows, cols = np.meshgrid(np.arange(0, 9), np.arange(0, 9), indexing='ij')
        self.rc_pairs = [(i, j) for i, j in zip(rows.flatten(), cols.flatten())]

    def paintEvent(self, e):
        qp = QPainter()
        qp.begin(self)

        qp.drawImage(QPoint(), self.image)

        pen = QPen(Qt.red)
        pen.setWidth(1)
        qp.setPen(pen)

        font = QFont()
        font.setFamily('Times')
        font.setBold(True)
        font.setPointSize(10)
        qp.setFont(font)

        for idx, point in enumerate(self.points):
            j = idx % 9
            i = (idx - j) // 9
            qp.drawText(point.x(), point.y(), "{}".format(self.index[i][j]))

        qp.end()


class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(50, 50, 660, 620)
        self.setWindowTitle("Add a text on image")
        self.image = QImage("../puzzles/puzzle1_initial.png")
        xv, yv = np.meshgrid(np.linspace(952+21, 1326-21, num=9), np.linspace(423+30, 793-16, num=9))
        self.cell_centers = np.vstack((xv.flatten(), yv.flatten()))
        self.points = [QPoint(i, j) for i, j in zip(xv.flatten(), yv.flatten())]
        self.label = Label(self.image, self.points)
        self.grid = QGridLayout()
        self.grid.addWidget(self.label)
        self.setLayout(self.grid)

    def incrementIndex(self, i, j, pos):
        if pos:
            self.label.index[i, j] += 1
        else:
            self.label.index[i, j] -= 1


    def mousePressEvent(self, event):
        mypoint = np.array([[event.pos().x()], [event.pos().y()]])
        diff = self.cell_centers - mypoint
        min_idx = np.argmin((diff ** 2).sum(axis=0))
        j = min_idx % 9
        i = (min_idx - j) // 9

        if event.button() == Qt.LeftButton:
            self.incrementIndex(i, j, 1)

        if event.button() == Qt.RightButton:
            self.incrementIndex(i, j, 0)

        self.label.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            a = 0


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    ex.show()
    sys.exit(app.exec_())
