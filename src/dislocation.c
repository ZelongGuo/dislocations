#ifdef __cplusplus
extern "C"
#endif

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "numpy/arrayobject.h"


static PyObject *add(PyObject *self, PyObject *args) {
	double x; 
	double y;
	PyArg_ParseTuple(args, "dd", &x, &y);
	return PyFloat_FromDouble(x + y);
}


// ---------------------------------------------------------------------- 
// 4. 模块函数列表
static PyMethodDef method_funcs[] = {
	// function name, function pointer, argument flag, function docs
	{"add", add, METH_VARARGS, "Add two numbers together."},
	{NULL, NULL, 0, NULL}
};

// ---------------------------------------------------------------------- 
// 3. module definition
static struct PyModuleDef abc123 = {
	PyModuleDef_HEAD_INIT,
	"abc123", 			  // module name
	"This is a module named abc123.", // module docs, could be called by help(module_name)
	-1, 
	method_funcs 			  // module functions list 
};

// ---------------------------------------------------------------------- 
// 1. C extension entrance, PyInit_+module name
PyMODINIT_FUNC PyInit_abc123() {
	printf("Now the module has been imported!\n");
	// 2. create module, the argument type is PyModuleDef
	return PyModule_Create(&abc123);
	
	// Initialize Numpy
	PyObject *module = PyModule_Create(&abc123);
	import_array();
	return module;
}
