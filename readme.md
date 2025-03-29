# Dislocations

### **Still Under Developing...**

Calculating elastic deformation from dislocation sources

Mind the unit: recommend the International System of Units.

Though strain is a dimensionless quantity, specifying dislocation unit same with fault parameters would help avoid further transformation.

## Requirments:  
Numpy > 1.7

**Point source model codes have not been tested yet, use them at your own risk.**

## Improvement:
Mehdi's codes is recommended because of numerical instabilities and no artefact singularities, even though I
also included Meade's codes in the package. I integrated and improved Mehdi's code to ensure that the normal vectors of triangular elements are always upward, regardless of the vertex order, making it more user-friendly.

Unified the strike-slip, dip-slip, and tensile coordinate systems for special faults (vertical faults, horizontal faults)

## 1. Compile and Install

### By Pypi
The easiest of building and installing the package is by Pypi:
```python 
pip install Dislocations
```

On MacOS platform it works pretty well, while it is not tested on Win and Linux platforms, though
it is also supposed to work. In case of not working, please try below using CMake building from the
source code directly:

### Build and install from the source codes 
```sh
# In the projrct directory, and strongly recommend executing under a python virtual env
mkdir -p build
cd build
cmake .. 
make install
```
Then a shared library named `dislocations.so` should generated in your `build`  and also
your current python `site-packages` directories (In fact you may copy the library to your python
`site-packages` directory and then you are supposed to import it to python).


### Compile (by shared library/dynamic link library, dll)
Using `setup.py` file, building shared library in current directory otherwise in ./build
```bash
python setup.py build --build-lib ./
```
or, equivalently using gcc or clang compiler, `-undefined dynamic_lookup` is essential for undefined symbols:
```bash
gcc/clang src/dislocation.c src/okada_dc3d.c src/okada_disloc3d.c -fPIC -O2 -I<NumPy_core_include_path> -I<Python_include_path>/python3.XX -shared -undefined dynamic_lookup -o dislocation.so
```
Then you can copy the shared library `.so` file to your site-packages folder of your current python environment, or to your working directory. Then you could import the module successfully.

## 2. Todo
- Reduce singularities of triangular elements with Nikkhoo and Walter GJI algorithms ...
- Install with Pypi 

## 2. Reference:  
- Okada, Y., 1992, Internal deformation due to shear and tensile faults in a half-space, Bull. Seism. Soc. Am., 82, 1018-1040.

- Brendan J. Meade, 2007, Algorithms for the calculation of exact displacements, strains, and stresses for triangular dislocation elements in a uniform elastic half space, Computer & Geosciences, 33, 1064-1075.

- Nikkhoo, M., Walter, T. R., 2015, Triangular dislocation: an analytical, artefact-free solution.  Geophysical Journal International, 201, 2, 1119-1141.

## More Useful Resources:
Codes:  
okada_wrapper   
okada4py   
dc3d from stanford

> Zelong Guo, Potsdam  
zelong.guo@outlook.com



