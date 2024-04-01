from setuptools import setup, Extension
import numpy as np
# from os.path import join

# include the header of NumPy C API
np_inc = np.get_include()

# extension module
ext_module = Extension("abc123", ["abc123.c"], include_dirs=[np_inc])

setup(
    name="abc123",    # 打包文件名称
    version="1.0",
    ext_modules=[ext_module],
    # script_args=["build_ext", "--build-lib", "./"],
)
