#ifdef __cplusplus
extern "C"
#endif

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN
#include <assert.h>
#include "Python.h"
#include "numpy/arrayobject.h"
#include "okada_dc3d.h"
#include "okada_disloc3d.h"

static PyObject *okada_rect(PyObject *self, PyObject *args) {
    // Initialize NumPy array object
    PyArrayObject *obs     = NULL;       // ndarray of [n x 3], xyz coordinates of observation stations
    PyArrayObject *models  = NULL;       // ndarray of [n x 10]  
    PyArrayObject *obs_    = NULL;       // ndarray of [n x 3], xyz coordinates of observation stations
    PyArrayObject *models_ = NULL;       // ndarray of [n x 10]  
    double mu; 		        	 // shear modulus
    double nu; 		        	 // Poisson's ratio
    
    // Parse arguments from Python and make data type check, borrowed references for "O"
    if (!PyArg_ParseTuple(args, "O!O!dd", &PyArray_Type, &obs, &PyArray_Type, &models, &mu, &nu)) {
    	PyErr_SetString(PyExc_TypeError, "Expected NumPy arrays and Python floats as input.");
    	return NULL;
    }

    // Check if obs has 3 columns and models has 10 columns
    if ((PyArray_SIZE(models) % 10 != 0) || PyArray_SIZE(obs) % 3 != 0) {
        PyErr_SetString(PyExc_ValueError, "The observations should be an array of [n x 3] and models should be [n x 10]!\n");
        return NULL;
    }

    // C contiguous, double and aligned, a new reference or a brand new array would be returned
    obs_    = (PyArrayObject *)PyArray_FROM_OTF((PyObject *)obs, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (obs_ == NULL) return NULL;
    models_ = (PyArrayObject *)PyArray_FROM_OTF((PyObject *)models, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (models_ == NULL) { 
	Py_DECREF(obs_);
	return NULL;
    }

//    // Convert data type to double and C contiguous if needed
//    if ((PyArray_TYPE(obs) != NPY_DOUBLE) || !PyArray_IS_C_CONTIGUOUS(obs)) {
//	printf("Converting obs to double and C contiguous...\n");
//	obs_ = (PyArrayObject *)PyArray_FROMANY((PyObject *)obs, NPY_DOUBLE, 1, 2, NPY_ARRAY_C_CONTIGUOUS);
//	if (obs_ == NULL) {
//	    PyErr_SetString(PyExc_ValueError, "Converting the observations to double type and C contiguous failed! You may also need check its dimension which should be 1-D or 2-D!\n");
//	    Py_XDECREF(obs_);
//	    Py_XDECREF(models_);
//	    return NULL;
//	}
//    }
//    else {
//	Py_INCREF(obs);
//	obs_ = obs; 
//    }
//
//
//    if ((PyArray_TYPE(models) != NPY_DOUBLE) || !PyArray_IS_C_CONTIGUOUS(models)) {
//	printf("Converting models ton double and C contiguous...\n");
//	models_ = (PyArrayObject *)PyArray_FROMANY((PyObject *)models, NPY_DOUBLE, 1, 2, NPY_ARRAY_C_CONTIGUOUS);
//	if (models_ == NULL) {
//	    PyErr_SetString(PyExc_ValueError, "Converting the models to double type and C contiguous failed! You may also need check its dimension which should be 1-D or 2-D!\n");
//	    Py_DECREF(obs_);
//	    Py_XDECREF(models_);
//	    return NULL;
//	}
//    }
//    else {
//	models_ = models;
//	Py_INCREF(models);
//    }

    // Get the numbers of models and stations
    npy_intp nmodels = PyArray_SIZE(models_) / 10;
    npy_intp nobs    = PyArray_SIZE(obs_)    / 3;
    // printf("nmodels:  %ld\n", nmodels); 
    // printf("nobs:  %ld\n", nobs); 
    
    // Accessing data with 1-D C array 
    double *c_models = (double *)PyArray_DATA(models_);
    double *c_obs = (double *)PyArray_DATA(obs_);

    // Initialize U, D, S and flags
    PyArrayObject *U;
    PyArrayObject *D;
    PyArrayObject *S;
    PyArrayObject *flags;
    npy_intp dims1 = (npy_intp) nobs * 3;
    npy_intp dims2 = (npy_intp) nobs * 9;
    npy_intp dims3 = (npy_intp) nobs * nmodels;
    U     = (PyArrayObject *)PyArray_ZEROS(1, &dims1, NPY_DOUBLE, 0);
    D     = (PyArrayObject *)PyArray_ZEROS(1, &dims2, NPY_DOUBLE, 0);
    S     = (PyArrayObject *)PyArray_ZEROS(1, &dims2, NPY_DOUBLE, 0);
    flags = (PyArrayObject *)PyArray_ZEROS(1, &dims3, NPY_INT, 0);
    if ((U == NULL) || (D == NULL) || (S == NULL) || (flags == NULL)) {
	PyErr_SetString(PyExc_MemoryError, "Failed to allocate memories for U, D, S and falgs!");
	Py_XDECREF(U);
	Py_XDECREF(D);
	Py_XDECREF(S);
	Py_XDECREF(flags);
	Py_DECREF(obs_);
	Py_DECREF(models_);
	return NULL;
    }

    double *U_data = (double *)PyArray_DATA(U);
    double *D_data = (double *)PyArray_DATA(D);
    double *S_data = (double *)PyArray_DATA(S);
    int *flags_data = (int *)PyArray_DATA(flags);

    // call disloc3d.c
    disloc3d(c_models, nmodels, c_obs, nobs, mu, nu, U_data, D_data, S_data, flags_data);

    PyObject *results = Py_BuildValue("(NNNN)", U, D, S, flags);

    /*
    printf("\n");
    for (npy_intp i =0; i<nobs; i++) {
        for (int j=0; j< 3; j++) {
    	printf("%f  ", *(U_data+i*3+j));
        }
        printf("\n");
    }  
    
    printf("\n");
    for (npy_intp i =0; i<nobs; i++) {
        for (int j=0; j< 9; j++) {
    	printf("%f  ", *(S_data+3*i+j));
        }
        printf("\n");
    }  
    */
    
    // free memory
    Py_DECREF(obs_);
    Py_DECREF(models_);
    obs_    = NULL;
    models_ = NULL;

    // return a Python Object Pointer
    //Py_RETURN_NONE;
    return results;
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
    "Okada_rect calculates displacement, stress and strain using Okada rectangle dislocation source.\n"
    "\n"
    "okada_rect(observations, models, mu, nu)\n"
    
    "- Arguments:\n"
    "  - observations: An 2-D NumPy array of [nobs x 3], xyz Cartesian coordinates of stations, \n"
    "                  in which z values should <= 0. \n"
    "  - models      : An 2-D Numpy array of [nmodles x 10], in which: \n"
    "                [x_uc, y_uc, depth, length, width, strike, dip, str_slip, dip_slip, opening] \n"
    "                 x_uc     : x Cartesian coordinates of the fault reference point (center of fault upper edge), \n"
    "                 y_uc     : y Cartesian coordinates of the fault reference point (center of fault upper edge), \n"
    "                 depth    : depth of the fault reference point (center of fault upper edge, always positive vales), \n"
    "                 length   : length of the fault patches, \n"
    "                 width    : width of the fault patches, \n"
    "                 strike   : strike angles of the fault patches, \n"
    "                 dip      : dip angles of the fault patches, \n"
    "                 str-slip : strike-slip components of the fault patches, \n"
    "                 dip-slip : dip-slip components of the fault patches, \n"
    "                 opening  : tensile components of the fault patches, \n"
    "  - mu          : Shear modulus, \n"
    "  - nu          : Poisson's ratio, \n"
    "\n"
    "- Output:\n"
    "  - u           : displacements,\n"
    "  - d           : 9 spatial derivatives of the displacements,\n"
    "  - s           : 9 stress tensor components, 6 of them are independent,\n"
    "  - flags       : flags [nobs x nmodels].\n"
    "                  flags = 0 : normal,\n"
    "                  flags = 1: the Z value of the obs > 0,\n"
    "                  flags = 10: the depth of the fault upper center point < 0, \n"
    "                  flags = 100: singular point, observation is on fault edges,\n"
    "                  flags could also be 11, 101, 110, 111, indicating the sum of multiple cases.\n "
    "\n"
    "NOTE: The units of displacements are same to dislocation slip (str-slip ...);\n"
    "      Strains are dimensionless, Cartesian coordinates and dislocation slip are better kept as 'm', so that there is no need to make conversion to the strain; \n"
    "      The units of stress depends on shear modulus mu, Pa is recommended. \n"

);

// 4. module funcs list
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
    printf("Dislocation Module has been imported!\n");
    // 2. create module, the argument type is PyModuleDef
    // return PyModule_Create(&dislocation);
    
    // Initialize Numpy
    PyObject *module = PyModule_Create(&dislocation);
    import_array();
    return module;
}
