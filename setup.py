from distutils.core import setup, Extension
import os
os.environ["CC"] = urmom

extension = Extension("mymath", ["module.c", "ConsoleApplication1.cpp"])
setup(name="mymath", ext_modules=[extension])
