import taichi as ti
import taichi.math as tm
from taichi.math import (vec3, inf)
import numpy as np
import trimesh
from PIL import Image
from math import *
import random
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QSpinBox, QComboBox, QFileDialog)
from PyQt5.QtCore import QRect
from PyQt5.QtGui import QFont
import sys
import os
import time
import threading

filepath = ""
filename = ""
dirname = ""

resolution = ()
data = []

arch = ti.cpu
arch_list = ["CPU", "GPU"]

count = 0

class Widget(QWidget):
    def __init__(self):
        super(Widget,self).__init__()
    
    def closeEvent(self,event):
        os._exit(0)

def OpenFile():
    global filepath,filename,dirname
    path = QFileDialog.getOpenFileName(widget,"Open a 3D Model File",".")
    if type(path) == tuple:
        path = path[0]
    if os.path.exists(path):
        filepath = path
        filename = os.path.basename(filepath)
        filename = ''.join(filename.split(".")[:-1])
        dirname = os.path.dirname(filepath)
        button.setEnabled(True)
        widget.setWindowTitle(filename + ' - ' + 'Grayscale')
        label4.setText("Ready to Generate.")
        label_max.setText("Maximum radius: NaN")
        label_min.setText("Minimum radius: NaN")

    else:
        button.setEnabled(False)
        widget.setWindowTitle("Grayscale")

@ti.dataclass
class Ray:
    origin : vec3
    dir : vec3
    t : float

@ti.dataclass
class Triangle:
    p1 : vec3
    p2 : vec3
    p3 : vec3

@ti.dataclass
class HitResult:
    isHit : ti.u1
    hitPoint : vec3
    dist : float

@ti.func
def Intersect(ray : Ray, tri : Triangle) -> HitResult:
    norm = tm.normalize(tm.cross(tri.p2 - tri.p1, tri.p3 - tri.p2))
    barycenter = (tri.p1 + tri.p2 + tri.p3) / 3
    t = tm.dot(barycenter - ray.origin, norm) / tm.dot(ray.dir, norm)
    hit_res = HitResult(0, vec3(0), inf)
    if t >= 0 and t < ray.t:
        p = ray.origin + t * ray.dir
        b1 = tm.dot(tm.cross(tri.p2 - tri.p1, p - tri.p1), norm)
        b2 = tm.dot(tm.cross(tri.p3 - tri.p2, p - tri.p2), norm)
        b3 = tm.dot(tm.cross(tri.p1 - tri.p3, p - tri.p3), norm)
        if (b1 >= 0 and b2 >= 0 and b3 >= 0) or (b1 <= 0 and b2 <= 0 and b3 <= 0):
            hit_res = HitResult(1, p, t)
    return hit_res

@ti.kernel
def Loop(
        frame : ti.types.ndarray(),
        verts : ti.types.ndarray(),
        faces : ti.types.ndarray(),
    ):
    dh = 2 * pi / frame.shape[1]
    dv = pi / frame.shape[0]
    for i,j in ti.ndrange(frame.shape[0], frame.shape[1]):
        h = j * dh
        v = pi / 2 - i * dv
        isHit : ti.u1 = ti.cast(0, ti.u1)
        ray = Ray(origin=vec3(0),
                  dir=tm.normalize(vec3(tm.cos(v) * tm.cos(h), tm.cos(v) * tm.sin(h), tm.sin(v))),
                  t=inf)
        for n in range(0, faces.shape[0]):
            p1 = vec3(verts[faces[n, 0], 0], verts[faces[n, 0], 1], verts[faces[n, 0], 2])
            p2 = vec3(verts[faces[n, 1], 0], verts[faces[n, 1], 1], verts[faces[n, 1], 2])
            p3 = vec3(verts[faces[n, 2], 0], verts[faces[n, 2], 1], verts[faces[n, 2], 2])
            result = Intersect(ray, Triangle(p1, p2, p3))
            if result.isHit == 1:
                isHit = ti.cast(1, ti.u1)
                ray.t = result.dist
        if isHit == 0:
            frame[i, j] = 0.0
        else:
            frame[i, j] = ray.t
        count = count + 1

def Generate():
    global box_x,box_y,widget,label4,label_min,label_max,box_arch,arch
    global resolution,data,count

    openfile.setEnabled(False)
    box_x.setEnabled(False)
    box_y.setEnabled(False)
    button.setEnabled(False)
    box_arch.setEnabled(False)

    if box_arch.currentText() == "CPU":
        arch = ti.cpu
    else:
        arch = ti.gpu

    label4.setText("Generating Grayscale...")

    ti.init(arch=arch)
    resolution = (box_x.value(),box_y.value())
    data = np.zeros((resolution[1], resolution[0]), dtype=np.float32)
    mesh = trimesh.load(filepath)
    count = 0

    Loop(data, mesh.vertices, mesh.faces)

    localtime = time.localtime(time.time())
    timestr = str(localtime.tm_hour).zfill(2) + str(localtime.tm_min).zfill(2)
    save_name = str(filename) + "_" + str(resolution[0]) + "x" + str(resolution[1]) + "_" + str(timestr) + ".txt"
    np.savetxt(dirname + '\\' + save_name, data)

    min_value = data.min()
    max_value = data.max()
    data = (data-min_value)/(max_value-min_value)*255
    image = Image.fromarray(data)
    image = image.convert('RGB')
    save_name = str(filename) + "_" + str(resolution[0]) + "x" + str(resolution[1]) + "_" + str(timestr) + ".png"
    image.save(dirname + '\\' + save_name)
    label4.setText("Successfully saved to " + save_name)
    widget.setWindowTitle("Grayscale")
    label_min.setText("Minimum radius: %.6f"%(min_value))
    label_max.setText("Maximum radius: %.6f"%(max_value))

    openfile.setEnabled(True)
    box_x.setEnabled(True)
    box_y.setEnabled(True)
    button.setEnabled(False)
    box_arch.setEnabled(True)

if __name__ == '__main__' :
    app = QApplication(sys.argv)
    widget = Widget()
    widget.setWindowTitle("Grayscale")
    widget.setMaximumSize(350,250)
    widget.setMinimumSize(350,250)
    label1 = QLabel("Model File Path: ", widget)
    label1.setGeometry(QRect(10,10,130,25))
    label1.setFont(QFont('Arial'))
    openfile = QPushButton("Select File", widget)
    openfile.setFont(QFont('Arial'))
    openfile.setGeometry(QRect(130,10,100,25))
    openfile.clicked.connect(OpenFile)
    label2 = QLabel("Resolution: ", widget)
    label2.setGeometry(QRect(10,50,80,25))
    label2.setFont(QFont('Arial'))
    box_x = QSpinBox(widget)
    box_x.setGeometry(QRect(100,50,80,25))
    box_x.setMaximum(16384)
    box_x.setMinimum(10)
    box_x.setValue(1024)
    box_x.setFont(QFont('Arial'))
    label3 = QLabel("x", widget)
    label3.setGeometry(QRect(185,50,10,25))
    label3.setFont(QFont('Arial'))
    box_y = QSpinBox(widget)
    box_y.setGeometry(QRect(200,50,80,25))
    box_y.setMaximum(16384)
    box_y.setMinimum(10)
    box_y.setValue(512)
    box_y.setFont(QFont('Arial'))
    label5 = QLabel("Architechture: ", widget)
    label5.setGeometry(QRect(10,90,120,25))
    label5.setFont(QFont('Arial'))
    box_arch = QComboBox(widget)
    box_arch.addItems(arch_list)
    box_arch.setGeometry(QRect(130,90,80,25))
    box_arch.setFont(QFont('Arial'))
    button = QPushButton("Generate", widget)
    button.setGeometry(QRect(10,130,330,30))
    button.setFont(QFont('Arial'))
    button.clicked.connect(Generate)
    label4 = QLabel("Ready to Generate.", widget)
    label4.setGeometry(QRect(10,220,330,25))
    label4.setFont(QFont('Arial',8))
    label_min = QLabel("Minimum radius: NaN", widget)
    label_min.setFont(QFont('Arial',8))
    label_min.setGeometry(QRect(10,170,330,25))
    label_max = QLabel("Maximum radius: NaN", widget)
    label_max.setFont(QFont('Arial',8))
    label_max.setGeometry(QRect(10,190,330,25))

    openfile.setEnabled(True)
    box_x.setEnabled(True)
    box_y.setEnabled(True)
    button.setEnabled(False)
    box_arch.setEnabled(True)

    widget.show()
    sys.exit(app.exec_())
