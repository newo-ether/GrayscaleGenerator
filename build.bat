@echo off
pyrcc5 resource.qrc -o resource.py
pyinstaller -F -w main.py -n GrayscaleGenerator -i Icon.ico --collect-all taichi --add-data "taichi_kernel;taichi_kernel"
