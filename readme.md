# Dislocation
**Still Under Developing...**  

Calculating elastic deformation from dislocation sources

Mind the unit: recommend the International System of Units.

Though strain is a dimensionless quantity, specifying dislocation unit same with fault parameters would help avoid further transformation.


**Point source model codes have not been tested yet, use them at your own risk.**
## 1. Compile and Install
### 1.1 Compile (by shared library/dynamic link library, dll)
Using `setup.py` file, building shared library in current directory otherwise in ./build
```bash
python setup.py build --build-lib ./
```
or, equivalently using gcc or clang compiler, `-undefined dynamic_lookup` is essential for undefined symbols:
```bash
gcc/clang src/dislocation.c src/okada_dc3d.c src/okada_disloc3d.c -fPIC -O2 -I<NumPy_core_include_path> -I<Python_include_path>/python3.XX -shared -undefined dynamic_lookup -o dislocation.so
```

### 1.2 Installation (by installing)
`python setup.py install` has been deprecated, please use `pip install **`

## Reference:  
Okada, Y., 1992, Internal deformation due to shear and tensile faults in a half-space, Bull. Seism. Soc. Am., 82, 1018-1040.

## More Useful Resources:
Codes:  
okada_wrapper Ben  
okada4py Romain   
dc3d from stanford

> Zelong Guo, Potadam  
zelong.guo@outlook.com



