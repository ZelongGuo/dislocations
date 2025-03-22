# This is for CMake File for Dislocation Extension Module
# Zelong Guo, @ Potsdam, DE

# ------------------------ Dislocation ------------------------
# -------- Python related stuff  --------
# Allow user to specify the Python environment path:
# cmake .. -DPYTHON_ENV_PATH=/path/to/your/env
if(NOT DEFINED PYTHON_ENV_PATH)
    set(PYTHON_ENV_PATH "/Users/zelong/opt/miniconda3/envs/temp") # My default path
endif()
message(STATUS "Using Python environment: ${PYTHON_ENV_PATH}")

# Auto-detect Python include directory
execute_process(
    COMMAND "${PYTHON_ENV_PATH}/bin/python" -c "import sysconfig; print(sysconfig.get_path('include'))"
    OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Python include dir: ${PYTHON_INCLUDE_DIR}")

# Auto-detect NumPy include directory
execute_process(
    COMMAND "${PYTHON_ENV_PATH}/bin/python" -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE NUMPY_STATUS
)
# If numpy not found
if(NOT NUMPY_STATUS EQUAL 0)
    message(FATAL_ERROR "NumPy not found in the specified Python environment!")
endif()
message(STATUS "NumPy include dir: ${NUMPY_INCLUDE_DIR}")

# Auto-detect Python site-packages directory
execute_process(
    COMMAND "${PYTHON_ENV_PATH}/bin/python" -c 
        "import sys, site; print(site.getsitepackages()[0] if hasattr(site, 'getsitepackages') else sys.prefix)"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE SITE_PKG_STATUS
)
# If site-packages not found
if(NOT EXISTS ${PYTHON_SITE_PACKAGES})
    message(FATAL_ERROR "Cannot locate Python site-packages directory")
endif()
message(STATUS "Installation path: ${PYTHON_SITE_PACKAGES}")

# -------- Source Files --------
# set(SOURCES
#     ${CMAKE_SOURCE_DIR}/src/core/okada_dc3d.c
#     ${CMAKE_SOURCE_DIR}/src/core/okada_disloc3d.c
#     # ${CMAKE_SOURCE_DIR}/src/core/okada_dc3d0.c
#     # ${CMAKE_SOURCE_DIR}/src/core/okada_disloc3d0.c
# )
aux_source_directory (${CMAKE_SOURCE_DIR}/src/core/ SOURCES)

# -------- Objectives --------
# Pythin C Extension Module:
add_library(dislocation SHARED)
target_sources(dislocation PRIVATE ${CMAKE_SOURCE_DIR}/src/bindings/dislocation.c ${SOURCES})

# Including header files of Python and Numpy:
target_include_directories(dislocation PRIVATE
    ${CMAKE_SOURCE_DIR}/include/
    ${PYTHON_INCLUDE_DIR}
    ${NUMPY_INCLUDE_DIR}
)

# ------ Build Options -------
# Complier Options, if you want debugging:
target_compile_options(dislocation PRIVATE
    -O2
    -fPIC
)
# Linking Options go here:
if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")  # for macOS
    set(CMAKE_SHARED_LINKER_FLAGS "-undefined dynamic_lookup")
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(CMAKE_SHARED_LINKER_FLAGS "-Wl,-rpath,/usr/local/lib")
elseif (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--export-all-symbols -Wl,--enable-auto-import")
endif()

set_target_properties(dislocation PROPERTIES
    SUFFIX ".so"
    PREFIX ""
)

# Install the library to the Python site-packages:
install(
    TARGETS dislocation
    LIBRARY
    DESTINATION ${PYTHON_SITE_PACKAGES}
)
