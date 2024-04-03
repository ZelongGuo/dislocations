
Mind the unit: recommend the International System of Units.

Though strain is a dimensionless quantity, specifying dislocation unit same with fault parameters would help avoid further transformation.


**Point source model codes have not been tested yet, use them at your own risk.**

Reference:  

Papers:  
Okada, Y., 1992, Internal deformation due to shear and tensile faults in a half-space, Bull. Seism. Soc. Am., 82, 1018-1040.

Codes:  
okada_wrapper Ben  
okada4py Romain   
dc3d from stanford


## compile (by shared library/dynamic link library, dll)

```bash
# Using gcc compiler
gcc abc123.c -shared -o abc123.so -I<numpy_include_dir> -I<python_include_dir> -L<python_lib_dir> -l<python_lib_name>

# for instance:
(temp) ➜ ~/codes/CExtension gcc abc123.c -shared -o abc123.so -I/Users/zelong/opt/miniconda3/envs/temp/lib/python3.11/site-packages/numpy/core/include -I/Users/zelong/opt/miniconda3/envs/temp/include/python3.11 -L/Users/zelong/opt/miniconda3/envs/temp/lib -lpython3.11

# It will produce a .so files in the current directory, and under current directory, you could import the module in the corresponding python interpreter.
```

```bash
# build shared library in current directory otherwise in ./build
python setup.py build --build-lib ./
```
or equivalently using gcc or clang:
```bash
# -undefined dynamic_looku is essential for undefined symbols
gcc/clang src/dislocation.c src/okada_dc3d.c src/okada_disloc3d.c -fPIC -O2 -I/Users/zelong/opt/miniconda3/envs/temp/lib/python3.11/site-packages/numpy/core/include -I/Users/zelong/opt/miniconda3/envs/temp/include/python3.11 -shared -undefined dynamic_lookup -o dislocation.so
```
## installation (by installing)
`python setup.py install` has been deprecated, please use `pip install **`
