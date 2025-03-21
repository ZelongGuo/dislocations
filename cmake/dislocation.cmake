# This is for CMake File for Dislocation Extension Module
# Zelong Guo, @ Potsdam, DE

# ------------------------ Dislocation ------------------------
# -------- Source Files --------
# set(SOURCES
#     ${CMAKE_SOURCE_DIR}/src/core/okada_dc3d.c
#     ${CMAKE_SOURCE_DIR}/src/core/okada_disloc3d.c
#     # ${CMAKE_SOURCE_DIR}/src/core/okada_dc3d0.c
#     # ${CMAKE_SOURCE_DIR}/src/core/okada_disloc3d0.c
# )

aux_source_directory (${CMAKE_SOURCE_DIR}/src/core/ SOURCES)

# option(CONDA_ENV_PATH "Path to the conda virtual environment" "/Users/zelong/opt/miniconda3/envs/temp")


# From CONDA Virtual Environment:
set(CONDA_ENV_PATH "/Users/zelong/opt/miniconda3/envs/temp")
set(PYTHON_INCLUDE_DIR "${CONDA_ENV_PATH}/include/python3.11")
set(NUMPY_INCLUDE_DIR "${CONDA_ENV_PATH}/lib/python3.11/site-packages/numpy/core/include")
set(PYTHON_LIBRARY "${CONDA_ENV_PATH}/lib/libpython3.11.dylib")  # For link options

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
install(TARGETS dislocation LIBRARY DESTINATION ${CONDA_ENV_PATH}/lib/python3.11/site-packages/)

