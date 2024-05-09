import numpy as np
import trimesh

from PIL import Image

import taichi as ti

import importlib
import inspect
import sys
import os

sys.path.append(os.getcwd())
kernel = importlib.import_module("taichi_kernel.kernel")

import ctypes
import resource
appid = "grayscalegenerator"
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(appid)

from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QSpinBox, QComboBox, QCheckBox, QFileDialog)
from PyQt5.QtCore import QRect
from PyQt5.QtGui import (QFont, QIcon)

from qt_material import apply_stylesheet

import multiprocessing
from multiprocessing import shared_memory

import time

filepath = ""
filename = ""
dirname = ""

arch = ti.cpu
arch_list = ["CPU", "GPU"]

on_exit = False

class Widget(QWidget):
    def __init__(self):
        super(Widget,self).__init__()
    
    def closeEvent(self,event):
        global on_exit
        on_exit = True

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

def Calculate(arch, frame_name, res, verts, faces, useFast):
    ti.init(arch=arch, default_fp=ti.f64, random_seed=int(time.time()), offline_cache=False)
    frame_mem = shared_memory.SharedMemory(name=frame_name)
    frame = np.ndarray((res[1], res[0]), dtype=np.float64, buffer=frame_mem.buf)
    if useFast:
        useFast = ti.cast(1, ti.u1)
    else:
        useFast = ti.cast(0, ti.u1)
    kernel.CreateGrayscaleMap(frame, verts, faces, useFast)
    frame_mem.close()

def Generate():
    global box_x,box_y,widget,label4,label_min,label_max,box_arch,checkbox_fast
    global on_exit

    openfile.setEnabled(False)
    box_x.setEnabled(False)
    box_y.setEnabled(False)
    button.setEnabled(False)
    box_arch.setEnabled(False)
    checkbox_fast.setEnabled(False)

    label4.setText("Generating Grayscale Map...")

    if box_arch.currentText() == "CPU":
        arch = ti.cpu
    else:
        arch = ti.gpu

    useFast = checkbox_fast.isChecked()

    resolution = (box_x.value(),box_y.value())
    data_mem = shared_memory.SharedMemory(create=True, size=resolution[0] * resolution[1] * 8)
    data = np.ndarray((resolution[1], resolution[0]), dtype=np.float64, buffer=data_mem.buf)
    mesh = trimesh.load(filepath)

    process = multiprocessing.Process(target=Calculate, args=(arch, data_mem.name, resolution, mesh.vertices, mesh.faces, useFast))
    process.start()

    while process.is_alive():
        QApplication.processEvents()
        if on_exit == True:
            process.kill()
            return

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
    label_min.setText("Minimum distance: %.8f"%(min_value))
    label_max.setText("Maximum distance: %.8f"%(max_value))

    data_mem.close()
    data_mem.close()

    openfile.setEnabled(True)
    box_x.setEnabled(True)
    box_y.setEnabled(True)
    button.setEnabled(False)
    box_arch.setEnabled(True)
    checkbox_fast.setEnabled(True)

def multiprocessing_win_init():
    # Module multiprocessing is organized differently in Python 3.4+
    try:
        # Python 3.4+
        if sys.platform.startswith('win'):
            import multiprocessing.popen_spawn_win32 as forking
        else:
            import multiprocessing.popen_fork as forking
    except ImportError:
        import multiprocessing.forking as forking

    if sys.platform.startswith('win'):
        # First define a modified version of Popen.
        class _Popen(forking.Popen):
            def __init__(self, *args, **kw):
                if hasattr(sys, 'frozen'):
                    # We have to set original _MEIPASS2 value from sys._MEIPASS
                    # to get --onefile mode working.
                    os.putenv('_MEIPASS2', sys._MEIPASS)
                try:
                    super(_Popen, self).__init__(*args, **kw)
                finally:
                    if hasattr(sys, 'frozen'):
                        # On some platforms (e.g. AIX) 'os.unsetenv()' is not
                        # available. In those cases we cannot delete the variable
                        # but only set it to the empty string. The bootloader
                        # can handle this case.
                        if hasattr(os, 'unsetenv'):
                            os.unsetenv('_MEIPASS2')
                        else:
                            os.putenv('_MEIPASS2', '')

        # Second override 'Popen' class with our modified version.
        forking.Popen = _Popen

if __name__ == '__main__' :
    multiprocessing_win_init()
    multiprocessing.freeze_support()

    app = QApplication(sys.argv)
    icon = QIcon(":/Icon.ico")
    apply_stylesheet(app, theme="dark_blue")

    widget = Widget()
    widget.setWindowTitle("Grayscale")
    widget.setWindowIcon(icon)
    widget.setMaximumSize(350,290)
    widget.setMinimumSize(350,290)

    label1 = QLabel("Model File Path: ", widget)
    label1.setGeometry(QRect(10,10,110,25))
    label1.setFont(QFont('Airal', 8))
    openfile = QPushButton("Select File", widget)
    openfile.setFont(QFont('Airal', 8))
    openfile.setGeometry(QRect(130,10,180,25))
    openfile.clicked.connect(OpenFile)

    label2 = QLabel("Output Resolution: ", widget)
    label2.setGeometry(QRect(10,50,110,25))
    label2.setFont(QFont('Airal', 8))
    box_x = QSpinBox(widget)
    box_x.setGeometry(QRect(130,50,80,25))
    box_x.setMaximum(16384)
    box_x.setMinimum(10)
    box_x.setValue(1024)
    box_x.setFont(QFont('Airal', 8))
    label3 = QLabel("x", widget)
    label3.setGeometry(QRect(215,50,10,25))
    label3.setFont(QFont('Airal', 8))
    box_y = QSpinBox(widget)
    box_y.setGeometry(QRect(230,50,80,25))
    box_y.setMaximum(16384)
    box_y.setMinimum(10)
    box_y.setValue(512)
    box_y.setFont(QFont('Airal', 8))

    label5 = QLabel("Render Device: ", widget)
    label5.setGeometry(QRect(10,90,110,25))
    label5.setFont(QFont('Airal', 8))
    box_arch = QComboBox(widget)
    box_arch.addItems(arch_list)
    box_arch.setGeometry(QRect(130,90,180,25))
    box_arch.setFont(QFont('Airal', 8))

    label6 = QLabel("Fast Intersection: ", widget)
    label6.setGeometry(QRect(10,130,110,25))
    label6.setFont(QFont('Airal', 8))
    checkbox_fast = QCheckBox(widget)
    checkbox_fast.setGeometry(QRect(130,130,25,25))
    checkbox_fast.setChecked(False)

    button = QPushButton("Generate", widget)
    button.setGeometry(QRect(10,170,330,30))
    button.setFont(QFont('Airal', 8))
    button.clicked.connect(Generate)
    label4 = QLabel("Ready to Generate.", widget)
    label4.setGeometry(QRect(10,260,330,25))
    label4.setFont(QFont('Airal', 8))
    label_min = QLabel("Minimum radius: NaN", widget)
    label_min.setFont(QFont('Airal', 8))
    label_min.setGeometry(QRect(10,210,330,25))
    label_max = QLabel("Maximum radius: NaN", widget)
    label_max.setFont(QFont('Airal', 8))
    label_max.setGeometry(QRect(10,230,330,25))

    openfile.setEnabled(True)
    box_x.setEnabled(True)
    box_y.setEnabled(True)
    button.setEnabled(False)
    box_arch.setEnabled(True)
    checkbox_fast.setEnabled(True)

    widget.show()
    sys.exit(app.exec_())
