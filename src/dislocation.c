#ifdef __cplusplus
extern "C"
#endif

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "numpy/arrayobject.h"
#include "okada_dc3d.h"
#include "okada_disloc3d.h"


static PyObject *okada_rect(PyObject *self, PyObject *args) {

	// Initialize NumPy array object
	PyArrayObject *xs, *ys, *zs;
	PyArrayObject *x_uc, *y_uc, *depth;
	PyArrayObject *length, *width;
	PyArrayObject *dip, *strike;
	PyArrayObject *str-slip, *dip-slip, *opening;

	// Parse arguments from Pyhton 
	if (!PyArg_ParseTuple(
		args, "O!O!O!O!O!O!O!O!O!O!O!O!O!dd",
		&PyArray_Type, &xs,        &PyArray_Type, &ys,       &PyArray_Type, &zs,
		&PyArray_Type, &x_uc,      &PyArray_Type, &y_uc,     &PyArray_Type, &depth,
		&PyArray_Type, &length,    &PyArray_Type, &width, 
		&PyArray_Type, &dip,       &PyArray_Type, &strike, 
		&PyArray_Type, &str-slip,  &PyArray_Type, &dip-slip, &PyArray_Type, &opening, 
		)) {
		return NULL;
	}

	/* Type Check
	if (!PyArray_Check(xs) || PyArray_TYPE(xs) != NPY_DOUBLE || !PyArray_IS_C_CONTIGUOUS(xs)) {
	PyErr_SetString(PyExc_TypeError, "Argument must be a c-contiguous numpy array of double!");
	return NULL;
	}
	 */

	int nmodles = PyArray_DIMS(x_uc)[0]; 
	int nobs    = PyArray_DIMS(xs)[0]; 

	double *c_model; 	// fault patches, [nmodles x 10]
	double *c_obs; 		// observation stations, [nobs x 3]
	double mu, nu; 		// shear modulus and Poisson's ratio


	printf("There is %d fault patches!\n", nmodles);
	printf("There is %d stations!\n", nobs);

	/* TODO:
	 *
	 * PyArray_Concatenate
	 * PyArray_AsCArray
	 * PyArray_Free
	 *
	 * */

	// return a Python Object Pointer
	Py_RETURN_NONE;
}

// ---------------------------------------------------------------------- 
static PyObject *add(PyObject *self, PyObject *args) {
	double x; 
	double y;
	PyArg_ParseTuple(args, "dd", &x, &y);
	return PyFloat_FromDouble(x + y);
}

// ---------------------------------------------------------------------- 

PyDoc_STRVAR(
	okada_rect_doc,
	"Okada_rect calculates displacement, stress and strain using Okada rectangle dislocation source\n"
	"\n"
	"okada_rect(xs, ys, zs, x_uc, y_uc, depth, length, width, dip, strike, str-slip, dip-slip, opening, mu, nu)\n"

	"- Arguments:\n"
	"  xs       : x Cartesian coordinates of the stations, \n"
	"  ys       : y Cartesian coordinates of the stations, \n"
	"  zs       : z Cartesian coordinates of the stations, \n"
	"  x_uc     : x Cartesian coordinates of the fault reference point (center of fault upper edge), \n"
	"  y_uc     : y Cartesian coordinates of the fault reference point (center of fault upper edge), \n"
	"  depth    : depth of the fault reference point (center of fault upper edge, always positive vales), \n"
	"  length   : length of the fault patches, \n"

	"  width    : width of the fault patches, \n"
	"  dip      : dip angles of the fault patches, \n"
	"  strike   : strike angles of the fault patches, \n"
	"  str-slip : strike-slip components of the fault patches, \n"
	"  dip-slip : dip-slip components of the fault patches, \n"
	"  opening  : tensile components of the fault patches, \n"

	"- Output:\n"
	"  u     : displacements,\n"
	"  d     : strains,\n"
	"  s     : stress,\n"
	"  flag  : flags.\n"


	"NOTE: the units of displacements are same to dislocation slip (str-slip ...);\n"
	"      strains are dimensionless, Cartesian coordinates and dislocation slip are better kept as 'm', so that there is no need to make conversion to the strain; \n"
	"      the units of stress depends on shear modulus mu. \n"


	
// 4. 模块函数列表
static PyMethodDef method_funcs[] = {
	// function name, function pointer, argument flag, function docs
	{"add", add, METH_VARARGS, "Add two numbers together."},
	{"okada_rect", okada_rect, METH_VARARGS, okada_rect_doc},
	{NULL, NULL, 0, NULL}
};

// ---------------------------------------------------------------------- 
// 3. module definition
static struct PyModuleDef dislocation = {
	PyModuleDef_HEAD_INIT,
	"dislocation", 			  // module name
	"This is a module named dislocation.", // module docs, could be called by help(module_name)
	-1, 
	method_funcs 			  // module functions list 
};

// ---------------------------------------------------------------------- 
// 1. C extension entrance, PyInit_+module name
PyMODINIT_FUNC PyInit_dislocation() {
	printf("Now the module has been imported!\n");
	// 2. create module, the argument type is PyModuleDef
	// return PyModule_Create(&dislocation);
	
	// Initialize Numpy
	PyObject *module = PyModule_Create(&dislocation);
	import_array();
	return module;
}
