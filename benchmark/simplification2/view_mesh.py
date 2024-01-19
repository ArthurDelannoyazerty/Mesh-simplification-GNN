from pyqtgraph.Qt import QtCore, QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
import sys

class Plot3D():

    def __init__(self):
        app = QtWidgets.QApplication.instance()
        if app is None:
            self.app = QtWidgets.QApplication([])
        else:
            self.app = app

        self.view = gl.GLViewWidget()
        self.view.show()
        self.view.setGeometry(300, 150, 1600, 900)
        self.view.setCameraPosition(distance=700)

    def add_mesh(self, mesh):
        self.view.addItem(mesh)

    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            print('here')
            self.app.instance().exec_()

if __name__ == '__main':

    inputs = sys.argv
    basename = inputs[1]
    fraction = inputs[2]
    p_3d = Plot3D()
    vertices = np.loadtxt(basename + '_vertices_' + fraction + '.txt')
    faces = np.loadtxt(basename + '_faces_' + fraction + '.txt', dtype=int)

    mesh = gl.GLMeshItem(vertexes=vertices, faces=faces, smooth=False, shader='viewNormalColor')
    p_3d.add_mesh(mesh)
    p_3d.start()
