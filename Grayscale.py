import numpy as np
import trimesh
import mesh_raycast
from PIL import Image
from math import *
import random
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QSpinBox, QFileDialog)
from PyQt5.QtCore import QRect
from PyQt5.QtGui import QFont
import sys
import os
import time
import threading

lock = threading.Lock()

filepath = ""
filename = ""
dirname = ""
PI = 3.14159265359
positive_infinity = float('inf')
resolution = ()
data = []
thread_list = []
cx = 0
cy = 0
cz = 0
delta_rotation_h = 0
delta_rotation_v = 0
thread_num = 8
should_exit = False

class Widget(QWidget):
    def __init__(self):
        super(Widget,self).__init__()
    
    def closeEvent(self,event):
        global should_exit
        should_exit = True

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

def Loop(thread_id, start, end):
    global data,resolution,delta_rotation_h,filepath,thread_list
    mesh = trimesh.load(filepath)
    for v_num in range(start,end+1) :
        for h_num in range(0,resolution[0]) :
            h = h_num * delta_rotation_h
            v = PI/2 - v_num * delta_rotation_v
            triangles = np.array(mesh.vertices[mesh.faces], dtype='f4')
            collision_result = False
            while not collision_result:
                h_rand = h + random.uniform(-delta_rotation_h/10000,delta_rotation_h/10000)
                v_rand = v + random.uniform(-delta_rotation_v/10000,delta_rotation_v/10000)
                collision_result = mesh_raycast.raycast(source=(cx,cy,cz), direction=(cos(v_rand)*cos(h_rand),cos(v_rand)*sin(h_rand),sin(v_rand)), mesh=triangles)
            distance = min(collision_result, key=lambda x: x['distance'])['distance']
            lock.acquire()
            try:
                data[v_num][h_num] = distance
                thread_list[thread_id] = (v_num,h_num)
            finally:
                lock.release()
            if should_exit:
                return

def Calculate():
    global filepath,filename,dirname,PI
    global box_x,box_y,widget,label4,openfile,label_min,label_max,box_thread
    global resolution,cx,cy,cz,delta_rotation_h,delta_rotation_v,data
    global thread_num,thread_list

    openfile.setEnabled(False)
    box_x.setEnabled(False)
    box_y.setEnabled(False)
    button.setEnabled(False)
    box_thread.setEnabled(False)
    resolution = (box_x.value(),box_y.value())
    thread_num = box_thread.value()
    mesh = trimesh.load(filepath)
    center = mesh.center_mass
    cx = center[0]
    cy = center[1]
    cz = center[2]
    delta_rotation_h = 2*PI / resolution[0]
    delta_rotation_v = PI / (resolution[1]-1)
    data = np.zeros((resolution[1],resolution[0]))
    label4.setText("Generating Grayscale 0.00%...")

    threads = []
    thread_list = []
    interval = int(resolution[1]/thread_num)
    for n in range(0,thread_num) :
        start = n*interval
        end = (n+1)*interval-1
        if n == thread_num-1 :
            end = resolution[1]-1
        threads.append(threading.Thread(target=Loop,args=(n,start,end)))
        thread_list.append((start,0))
    for t in threads :
        t.start()

    while threading.active_count() > 1 :
        QApplication.processEvents()
        total = 0
        for n in range(0,thread_num) :
            start = n*interval
            v_num = thread_list[n][0] - start
            h_num = thread_list[n][1]
            total += v_num * resolution[0] + h_num
        label4.setText("Generating Grayscale %.2f%%..."%(total * 100 / (resolution[0]*resolution[1])))
        if should_exit:
            return

    min_distance = data.min()
    max_distance = data.max()
    min_value = min_distance
    max_value = max_distance
    data = (data-min_value)/(max_value-min_value)*255
    image = Image.fromarray(data)
    image = image.convert('RGB')
    localtime = time.localtime(time.time())
    timestr = str(localtime.tm_hour).zfill(2) + str(localtime.tm_min).zfill(2)
    save_name = str(filename) + "_" + str(resolution[0]) + "x" + str(resolution[1]) + "_" + str(timestr) + ".png"
    image.save(dirname + '\\' + save_name)
    label4.setText("Successfully saved to " + save_name)
    widget.setWindowTitle("Grayscale")
    label_min.setText("Minimum radius: %.6f"%(min_distance))
    label_max.setText("Maximum radius: %.6f"%(max_distance))

    openfile.setEnabled(True)
    box_x.setEnabled(True)
    box_y.setEnabled(True)
    button.setEnabled(False)
    box_thread.setEnabled(True)

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
    label5 = QLabel("Thread Number: ", widget)
    label5.setGeometry(QRect(10,90,120,25))
    label5.setFont(QFont('Arial'))
    box_thread = QSpinBox(widget)
    box_thread.setGeometry(QRect(130,90,80,25))
    box_thread.setMaximum(128)
    box_thread.setMinimum(1)
    box_thread.setValue(4)
    box_thread.setFont(QFont('Arial'))
    button = QPushButton("Generate", widget)
    button.setGeometry(QRect(10,130,330,30))
    button.setFont(QFont('Arial'))
    button.clicked.connect(Calculate)
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
    box_thread.setEnabled(True)

    widget.show()
    sys.exit(app.exec_())