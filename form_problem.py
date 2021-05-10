import cv2
import matplotlib.pyplot as plt
import argparse
import numpy as np
import sys
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QGridLayout, QDesktopWidget, QGridLayout
from PyQt5.QtGui import QPixmap



class mygui(QWidget):

    def __init__(self, image):
        super().__init__()
        self.image = image
        self.im_max_dims = (640, 480)   # width, height

        self.im = QPixmap(self.image)
        self.im_width = self.im.width()
        self.im_height = self.im.height()

        # check if dims of loaded image are larger than permitted dimensions
        if self.im_width > self.im_max_dims[0] or self.im_height > self.im_max_dims[1]:

            # figure out which dimension to scale
            if (self.im_max_dims[0]/self.im_width) > (self.im_max_dims[1]/self.im_height):
                print('this executed1')
                self.im = self.im.scaledToWidth(self.im_max_dims[0])
            else:
                print('this executed')
                self.im = self.im.scaledToHeight(self.im_max_dims[1])

        self.label = QLabel()
        self.label.setPixmap(self.im)

        self.grid = QGridLayout()
        self.grid.addWidget(self.label,1,1)
        self.setLayout(self.grid)

        self.setGeometry(50, 50, 1200, 800)
        self.setWindowTitle("PyQT show image")

        self.setup_gui()

        self.show()

    def setup_gui(self):
        self.center()


    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


    def find_puzzle(self, image):

        im = np.zeros((9,9,9))
        return image

    def find_numbers(self):
        pass


def main(image):

    app = QApplication(sys.argv)
    ex = mygui(image=image)
    sys.exit(app.exec_())

    pass



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="path to image with sudoku puzzle")
    args = parser.parse_args()


    # run gui
    main(args.image)

    puzzle = find_puzzle(args.image)
    # puzzle = form_puzzle()