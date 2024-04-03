from setuptools import setup, Extension
import numpy as np
# from os.path import join

# include the header of NumPy C API
np_inc = np.get_include()

source = ["src/dislocation.c", "src/okada_dc3d.c", "src/okada_disloc3d.c"]

# extension module
ext_module = Extension("dislocation", source, include_dirs=[np_inc])

setup(
    name="dislocation",    # 打包文件名称
    version="1.0",
    ext_modules=[ext_module],
    # script_args=["build_ext", "--build-lib", "./"],
)
