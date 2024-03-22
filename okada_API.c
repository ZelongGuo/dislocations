#ifdef __cplusplus
extern "C"
#endif

#define PY_SSIZE_T_CLEAN
#include <Python.h>
//#include "dc3d.h"
//#include "dc3d.c"
//#include "disloc3d.h"
#include "disloc3d.c"

static PyObject *disloc3d(PyObject *self, PyObject *args)
{

}

// Module Functions List
static PyMethodDef okada_funcs[] = {
	{
	"disloc3d",  // Function name
 	 disloc3d,	     // Function pointer
	METH_NOARGS, 		     // Args
	"Okada dislocation."			     // help()

	},
	{0, 0, 0, 0}
	

}

// Module definition
static PyModuleDef okada = {
	PyModuleDef_HEAD_INIT,
	"okada", // Module Name
	"okada dislocation for finite fault and pointe sources."	 // Module Information, help()
	-1,
	okada_funcs,

}

/* Module Initialization */
PyMODINIT_FUNC PyInit_okada(void)
{
	printf("PyInit_mymod\n");
	// create module
 	return PyModule_Create()
}

/* Create PyModule */
PyModule_Create


/* Module Information */
PyModuleDef
const char *m_name
const char *m_doc // help()
Py_ssize_t m_size; // -1
PyMethodDef * m_methods



/* Module Function Information */
PyMethodDef


/* Function Definitions */
PyObject *PSystem(PyObject *self, PyObject *args)
