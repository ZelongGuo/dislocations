#ifdef __cplusplus
extern "C"
#endif

/* ------------------------------------------------------------------------------------
 * A C script binds C and Python codes as a Python C extension. This module integrates
 * rectangle and triangle dislocation elements for calculating surface deformation and 
 * stress, strain.
 *
 * Zelong Guo
 * 22.03.2025, @ Potsdam, DE
 *
 * ----------------------------------------------------------------------------------*/

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "numpy/arrayobject.h"
#include <assert.h>
#include "okada.h"
#include "mehdi.h"
#include "meade.h"

static PyObject *rde(PyObject *self, PyObject *args) {
    // Initialize NumPy array object
    PyArrayObject *obs     = NULL;   // ndarray of [n x 3], xyz coordinates of observation stations
    PyArrayObject *models  = NULL;   // ndarray of [n x 10]
    PyArrayObject *obs_    = NULL;   // ndarray of [n x 3], xyz coordinates of observation stations
    PyArrayObject *models_ = NULL;   // ndarray of [n x 10]
    double mu;                       // shear modulus
    double nu;                       // Poisson's ratio

    // Parse arguments from Python and make data type check, borrowed references for "O"
    if (!PyArg_ParseTuple(args, "O!O!dd", &PyArray_Type, &obs, &PyArray_Type, &models, &mu, &nu)) {
        PyErr_Print();  // detailed message
        PyErr_SetString(PyExc_TypeError, "Expected NumPy arrays and Python floats as input.");
        return NULL;
    }

     // Check if obs has 3 columns and models has 10 columns
    if ((PyArray_SIZE(models) % 10 != 0) || PyArray_SIZE(obs) % 3 != 0) {
        PyErr_SetString(PyExc_ValueError,
                        "The observations should be an array of [n x 3] and models "
                        "should be [n x 10]!");
        return NULL;
    }

    // C contiguous (row-major), double and aligned, a new reference or a brand new array would be returned
    obs_ = (PyArrayObject *)PyArray_FROM_OTF((PyObject *)obs, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (obs_ == NULL) {
        PyErr_SetString(PyExc_ValueError, "The observations array is NULL!");
        return NULL;
    }

    models_ = (PyArrayObject *)PyArray_FROM_OTF((PyObject *)models, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (models_ == NULL) {
        Py_DECREF(obs_);
        PyErr_SetString(PyExc_ValueError, "The models array is NULL!");
        return NULL;
    }


    // Get the numbers of models and stations
    npy_intp nmodels = PyArray_SIZE(models_) / 10;
    npy_intp nobs = PyArray_SIZE(obs_) / 3;

    // Accessing data with 1-D C array
    double *c_models = (double *)PyArray_DATA(models_);
    double *c_obs = (double *)PyArray_DATA(obs_);

    // Initialize U, S, E and flags
    PyArrayObject *U;
    PyArrayObject *S;
    PyArrayObject *E;
    PyArrayObject *flags;
    /* 1D array
    npy_intp dims1 = (npy_intp)nobs * 3;
    npy_intp dims2 = (npy_intp)nobs * 6;
    npy_intp dims3 = (npy_intp)nobs * nmodels;
    U = (PyArrayObject *)PyArray_ZEROS(1, &dims1, NPY_DOUBLE, 0);
    S = (PyArrayObject *)PyArray_ZEROS(1, &dims2, NPY_DOUBLE, 0);
    E = (PyArrayObject *)PyArray_ZEROS(1, &dims2, NPY_DOUBLE, 0);
    flags = (PyArrayObject *)PyArray_ZEROS(1, &dims3, NPY_INT, 0);
    */

    // Create 2D array, PyArray_ZEROS return a C contiguous araay, it's fine as the input of okada_disloc3d
    npy_intp dims_u[] = {nobs, 3};           // U: (nobs, 3)
    npy_intp dims_ds[] = {nobs, 6};          // S, E: (nobs, 6)
    npy_intp dims_flags[] = {nobs, nmodels}; // flags: (nobs, nmodels)
    U = (PyArrayObject *)PyArray_ZEROS(2, dims_u, NPY_DOUBLE, 0);
    S = (PyArrayObject *)PyArray_ZEROS(2, dims_ds, NPY_DOUBLE, 0);
    E = (PyArrayObject *)PyArray_ZEROS(2, dims_ds, NPY_DOUBLE, 0);
    flags = (PyArrayObject *)PyArray_ZEROS(2, dims_flags, NPY_INT, 0);

    if ((U == NULL) || (S == NULL) || (E == NULL) || (flags == NULL)) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memories for U, S, E and falgs!");
        Py_XDECREF(U);
        Py_XDECREF(S);
        Py_XDECREF(E);
        Py_XDECREF(flags);
        Py_DECREF(obs_);
        Py_DECREF(models_);
        return NULL;
    }

    double *U_data = (double *)PyArray_DATA(U);
    double *S_data = (double *)PyArray_DATA(S);
    double *E_data = (double *)PyArray_DATA(E);
    int *flags_data = (int *)PyArray_DATA(flags);

    // call okada_disloc3d.c
    okada_disloc3d(c_models, nmodels, c_obs, nobs, mu, nu, U_data, S_data, E_data,
                   flags_data);

    PyObject *results = NULL;
    results = Py_BuildValue("(NNNN)", U, S, E, flags);
    /* If failed to create, clear everything avoiding memory leak */
    if (results == NULL) {
        Py_DECREF(U);
        Py_DECREF(S);
        Py_DECREF(E);
        Py_DECREF(flags);
        return NULL;
    }

    // free memory
    Py_DECREF(obs_);
    Py_DECREF(models_);
    obs_ = NULL;
    models_ = NULL;

    return results;
}

// ----------------------------------------------------------------------
static PyObject *tde(PyObject *self, PyObject *args) {
    PyArrayObject *obs     = NULL;   // ndarray of [n x 3], xyz coordinates of observation stations
    PyArrayObject *models  = NULL;   // ndarray of [n x 12]
    PyArrayObject *obs_    = NULL;   // ndarray of [n x 3], xyz coordinates of observation stations
    PyArrayObject *models_ = NULL;   // ndarray of [n x 12]
    double mu;                       // shear modulus
    double nu;                       // Poisson's ratio

    // Parse arguments from Python and make data type check, borrowed references for "O"
    if (!PyArg_ParseTuple(args, "O!O!dd", &PyArray_Type, &obs, &PyArray_Type, &models, &mu, &nu)) {
        PyErr_SetString(PyExc_TypeError, "Expected NumPy arrays and Python floats as input.");
        return NULL;
    }

     // Check if obs has 3 columns and models has 12 columns
    if ((PyArray_SIZE(models) % 12 != 0) || PyArray_SIZE(obs) % 3 != 0) {
        PyErr_SetString(PyExc_ValueError,
                        "The observations should be an array of [n x 3] and models "
                        "should be [n x 12]!");
        return NULL;
    }

    // C contiguous (row-major), double and aligned, a new reference or a brand new array would be returned
    obs_ = (PyArrayObject *)PyArray_FROM_OTF((PyObject *)obs, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (obs_ == NULL) {
        PyErr_SetString(PyExc_ValueError, "The observations array is NULL!");
        return NULL;
    }

    models_ = (PyArrayObject *)PyArray_FROM_OTF((PyObject *)models, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (models_ == NULL) {
        Py_DECREF(obs_);
        PyErr_SetString(PyExc_ValueError, "The models array is NULL!");
        return NULL;
    }


    // Get the numbers of models and stations
    npy_intp nmodels = PyArray_SIZE(models_) / 12;
    npy_intp nobs = PyArray_SIZE(obs_) / 3;

    // Accessing data with 1-D C array
    double *c_models = (double *)PyArray_DATA(models_);
    double *c_obs = (double *)PyArray_DATA(obs_);

    // Initialize U, S, E and flags
    PyArrayObject *U;
    PyArrayObject *S;
    PyArrayObject *E;

    // Create 2D array, PyArray_ZEROS return a C contiguous araay, it's fine as the input of okada_disloc3d
    npy_intp dims_u[] = {nobs, 3};           // U: (nobs, 3)
    npy_intp dims_ds[] = {nobs, 6};          // S, E: (nobs, 6)
    U = (PyArrayObject *)PyArray_ZEROS(2, dims_u, NPY_DOUBLE, 0);
    S = (PyArrayObject *)PyArray_ZEROS(2, dims_ds, NPY_DOUBLE, 0);
    E = (PyArrayObject *)PyArray_ZEROS(2, dims_ds, NPY_DOUBLE, 0);

    if ((U == NULL) || (S == NULL) || (E == NULL)) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memories for U, S, E and falgs!");
        Py_XDECREF(U);
        Py_XDECREF(S);
        Py_XDECREF(E);
        Py_DECREF(obs_);
        Py_DECREF(models_);
        return NULL;
    }

    double *U_data = (double *)PyArray_DATA(U);
    double *S_data = (double *)PyArray_DATA(S);
    double *E_data = (double *)PyArray_DATA(E);

    mehdi_disloc3d(c_models, nmodels, c_obs, nobs, mu, nu, U_data, S_data, E_data);

    PyObject *results = NULL;
    results = Py_BuildValue("(NNN)", U, S, E);
    /* If failed to create, clear everything avoiding memory leak */
    if (results == NULL) {
        Py_DECREF(U);
        Py_DECREF(S);
        Py_DECREF(E);
        return NULL;
    }

    // free memory
    Py_DECREF(obs_);
    Py_DECREF(models_);
    obs_ = NULL;
    models_ = NULL;

    return results;
}

// ----------------------------------------------------------------------
static PyObject *tde_meade(PyObject *self, PyObject *args) {
    PyArrayObject *obs     = NULL;   // ndarray of [n x 3], xyz coordinates of observation stations
    PyArrayObject *models  = NULL;   // ndarray of [n x 12]
    PyArrayObject *obs_    = NULL;   // ndarray of [n x 3], xyz coordinates of observation stations
    PyArrayObject *models_ = NULL;   // ndarray of [n x 12]
    double mu;                       // shear modulus
    double nu;                       // Poisson's ratio

    // Parse arguments from Python and make data type check, borrowed references for "O"
    if (!PyArg_ParseTuple(args, "O!O!dd", &PyArray_Type, &obs, &PyArray_Type, &models, &mu, &nu)) {
        PyErr_SetString(PyExc_TypeError, "Expected NumPy arrays and Python floats as input.");
        return NULL;
    }

     // Check if obs has 3 columns and models has 12 columns
    if ((PyArray_SIZE(models) % 12 != 0) || PyArray_SIZE(obs) % 3 != 0) {
        PyErr_SetString(PyExc_ValueError,
                        "The observations should be an array of [n x 3] and models "
                        "should be [n x 12]!");
        return NULL;
    }

    // C contiguous (row-major), double and aligned, a new reference or a brand new array would be returned
    obs_ = (PyArrayObject *)PyArray_FROM_OTF((PyObject *)obs, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (obs_ == NULL) {
        PyErr_SetString(PyExc_ValueError, "The observations array is NULL!");
        return NULL;
    }

    models_ = (PyArrayObject *)PyArray_FROM_OTF((PyObject *)models, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (models_ == NULL) {
        Py_DECREF(obs_);
        PyErr_SetString(PyExc_ValueError, "The models array is NULL!");
        return NULL;
    }


    // Get the numbers of models and stations
    npy_intp nmodels = PyArray_SIZE(models_) / 12;
    npy_intp nobs = PyArray_SIZE(obs_) / 3;

    // Accessing data with 1-D C array
    double *c_models = (double *)PyArray_DATA(models_);
    double *c_obs = (double *)PyArray_DATA(obs_);

    // Initialize U, S, E and flags
    PyArrayObject *U;
    PyArrayObject *S;
    PyArrayObject *E;

    // Create 2D array, PyArray_ZEROS return a C contiguous araay, it's fine as the input of okada_disloc3d
    npy_intp dims_u[] = {nobs, 3};           // U: (nobs, 3)
    npy_intp dims_ds[] = {nobs, 6};          // S, E: (nobs, 6)
    U = (PyArrayObject *)PyArray_ZEROS(2, dims_u, NPY_DOUBLE, 0);
    S = (PyArrayObject *)PyArray_ZEROS(2, dims_ds, NPY_DOUBLE, 0);
    E = (PyArrayObject *)PyArray_ZEROS(2, dims_ds, NPY_DOUBLE, 0);

    if ((U == NULL) || (S == NULL) || (E == NULL)) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memories for U, S, E and falgs!");
        Py_XDECREF(U);
        Py_XDECREF(S);
        Py_XDECREF(E);
        Py_DECREF(obs_);
        Py_DECREF(models_);
        return NULL;
    }

    double *U_data = (double *)PyArray_DATA(U);
    double *S_data = (double *)PyArray_DATA(S);
    double *E_data = (double *)PyArray_DATA(E);

    meade_disloc3d(c_models, nmodels, c_obs, nobs, mu, nu, U_data, S_data, E_data);

    PyObject *results = NULL;
    results = Py_BuildValue("(NNN)", U, S, E);
    /* If failed to create, clear everything avoiding memory leak */
    if (results == NULL) {
        Py_DECREF(U);
        Py_DECREF(S);
        Py_DECREF(E);
        return NULL;
    }

    // free memory
    Py_DECREF(obs_);
    Py_DECREF(models_);
    obs_ = NULL;
    models_ = NULL;

    return results;
}


// ----------------------------------------------------------------------

PyDoc_STRVAR(rde_doc,
             "rde calculates displacement, stress and strain using Okada rectangle dislocation source in elastic half space.\n"
             "\n"
             "[u, s, e, flags] = dislocations.rde(rbservations, models, mu, nu)\n"

             "- Arguments:\n"
             "  - observations: An 2-D NumPy array of [nobs x 3], xyz Cartesian coordinates of stations, \n"
             "                  in which z values should <= 0. \n"
             "  - models      : An 2-D Numpy array of [nmodles x 10], in which: \n"
             "                [x_uc, y_uc, z_uc, length, width, strike, dip, str_slip, dip_slip, opening] \n"
             "                 x_uc     : x Cartesian coordinates of the fault reference point (center of fault upper edge), \n"
             "                 y_uc     : y Cartesian coordinates of the fault reference point (center of fault upper edge), \n"
             "                 z_uc     : z Cartesian coordinates of the fault reference point (center of fault upper edge), should be <=0,\n"
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
             "  - u           : 2-D array of displacements, ux, uy, uz, [nobs x 3] \n"
             "  - s           : 2-D array of 6 independent stress tensor components, sxx, sxy, sxz, syy, syz, szz, [nobs x 6] \n"
             "  - e           : 2-D array of 6 independent strain tensor components, exx, exy, exz, eyy, eyz, ezz, [nobs x 6] \n"
             "  - flags       : flags [nobs x nmodels].\n"
             "                  flags = 0 : normal,\n"
             "                  flags = 1: the Z value of the obs > 0,\n"
             "                  flags = 10: the depth of the fault upper center point < 0, \n"
             "                  flags = 100: singular point, observation is on fault edges,\n"
             "                  flags could also be 11, 101, 110, 111, indicating the sum of multiple cases.\n "
             "\n"
             "NOTE: The units of displacements are same to dislocation slip (str-slip ...);\n"
             "      Strains are dimensionless, Cartesian coordinates and dislocation slip are better kept as 'm', so that there is no need to make conversion to the strain; \n"
             "      The units of stress depends on shear modulus mu, Pa is recommended. "
             "\n"

);

PyDoc_STRVAR(tde_doc,
             "tde calculates displacement, stress and strain using triangle dislocation source in elastic half space, based on Nikkhoo and Walter 2015.\n"
             "\n"
             "[u, s, e] = dislocations.tde(observations, models, mu, nu)\n"

             "- Arguments:\n"
             "  - observations: An 2-D NumPy array of [nobs x 3], xyz Cartesian coordinates of stations, \n"
             "                  in which z values should <= 0. \n"
             "  - models      : An 2-D Numpy array of [nmodles x 12], in which: \n"
             "                  [x1, y1, z1, x2, y2, z2, x3, y3, z3, str_slip, dip_slip, opening] \n"
             "                  x1, y1, z1  : Vertex 1 of the triangle in Cartesian coordinates system,\n"
             "                  x2, y2, z2  : Vertex 2 of the triangle in Cartesian coordinates system,\n"
             "                  x3, y3, z3  : Vertex 3 of the triangle in Cartesian coordinates system,\n"
             "                  str-slip    : strike-slip components of the fault patches, \n"
             "                  dip-slip    : dip-slip components of the fault patches, \n"
             "                  opening     : tensile components of the fault patches, \n"
             "  - mu          : Shear modulus, \n"
             "  - nu          : Poisson's ratio, \n"
             "\n"
             "- Output:\n"
             "  - u           : 2-D array of displacements, ux, uy, uz, [nobs x 3] \n"
             "  - s           : 2-D array of 6 independent stress tensor components, sxx, sxy, sxz, syy, syz, szz, [nobs x 6] \n"
             "  - e           : 2-D array of 6 independent strain tensor components, exx, exy, exz, eyy, eyz, ezz, [nobs x 6] \n"
             "\n"
             "NOTE: The units of displacements are same to dislocation slip (str-slip ...);\n"
             "      Strains are dimensionless, Cartesian coordinates and dislocation slip are better kept as 'm', so that there is no need to make conversion to the strain; \n"
             "      The units of stress depends on shear modulus mu, Pa is recommended. "
             "\n"

);

PyDoc_STRVAR(tde_meade_doc,
             "tde_meade calculates displacement, stress and strain using triangle dislocation source in elastic half space, based on the algorithm of Meade 2007.\n"
            "Caution: The algorithm have singulaties and numerical unstabilities problem, thus the tde based on Nikkhoo and Walter is strongly recommended.\n"
             "\n"
             "[u, s, e] = dislocations.tde_meade(observations, models, mu, nu)\n"

             "- Arguments:\n"
             "  - observations: An 2-D NumPy array of [nobs x 3], xyz Cartesian coordinates of stations, \n"
             "                  in which z values should <= 0. \n"
             "  - models      : An 2-D Numpy array of [nmodles x 12], in which: \n"
             "                  [x1, y1, z1, x2, y2, z2, x3, y3, z3, str_slip, dip_slip, opening] \n"
             "                  x1, y1, z1  : Vertex 1 of the triangle in Cartesian coordinates system,\n"
             "                  x2, y2, z2  : Vertex 2 of the triangle in Cartesian coordinates system,\n"
             "                  x3, y3, z3  : Vertex 3 of the triangle in Cartesian coordinates system,\n"
             "                  str-slip    : strike-slip components of the fault patches, \n"
             "                  dip-slip    : dip-slip components of the fault patches, \n"
             "                  opening     : tensile components of the fault patches, \n"
             "  - mu          : Shear modulus, \n"
             "  - nu          : Poisson's ratio, \n"
             "\n"
             "- Output:\n"
             "  - u           : 2-D array of displacements, ux, uy, uz, [nobs x 3] \n"
             "  - s           : 2-D array of 6 independent stress tensor components, sxx, sxy, sxz, syy, syz, szz, [nobs x 6] \n"
             "  - e           : 2-D array of 6 independent strain tensor components, exx, exy, exz, eyy, eyz, ezz, [nobs x 6] \n"
             "\n"
             "NOTE: The units of displacements are same to dislocation slip (str-slip ...);\n"
             "      Strains are dimensionless, Cartesian coordinates and dislocation slip are better kept as 'm', so that there is no need to make conversion to the strain; \n"
             "      The units of stress depends on shear modulus mu, Pa is recommended. "
             "\n"

);

// ----------------------------------------------------------------------

// module funcs list
static PyMethodDef method_funcs[] = {
    // function name, function pointer, argument flag, function docs
    {"rde",       rde,       METH_VARARGS, rde_doc},
    {"tde",       tde,       METH_VARARGS, tde_doc},
    {"tde_meade", tde_meade, METH_VARARGS, tde_meade_doc},
    {NULL, NULL, 0, NULL}};

// ----------------------------------------------------------------------
// module definition
static struct PyModuleDef dislocations = {
    PyModuleDef_HEAD_INIT,
    "dislocations",                         // module name
    "This is a module named dislocations, which is a Python C extension integrating rectangle and triangle dislocation elements for calculating the displacements,\n stress and strain of the observation stations in elastic space. The codes are implemented based the papers of Okada 1992, Meade 2007, Nikkhoo and Walter 2015.\n "
    "Zelong Guo, @ Potsdam, DE\n"        // author
    "Email: zelong.guo@outlook.com\n",
                                          // module docs, could be called by help(module_name)
    -1,
    method_funcs // module functions list
};

// ----------------------------------------------------------------------
PyMODINIT_FUNC PyInit_dislocations() {
    // printf("Dislocations Module has been imported!\n");
    // Initialize Numpy
    import_array();
    if (PyErr_Occurred()) {
        return NULL;
    }

    PyObject *module = PyModule_Create(&dislocations);
    if (module == NULL) {
        return NULL;
    }
    return module;
}
