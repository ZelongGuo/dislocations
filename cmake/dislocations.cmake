# This is for CMake File for Dislocations Extension Module
# Zelong Guo, @ Potsdam, DE

# ------------------------ Dislocations ------------------------
# Activate the vitual environment, then CMake would find the python and numpy in this env, necessary for Mac
find_program(Python3_EXECUTABLE python REQUIRED)

# Find Python and NumPy in current environment
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
if(NOT Python3_FOUND)
    message(FATAL_ERROR "Python3 not found")
endif()
if(NOT Python3_NumPy_FOUND)
    message(FATAL_ERROR "NumPy not found")
endif()

message(STATUS "Python include dir: ${Python3_INCLUDE_DIRS}")
message(STATUS "NumPy include dir: ${Python3_NumPy_INCLUDE_DIRS}")

# -------- Source Files --------
aux_source_directory (${CMAKE_SOURCE_DIR}/src/core/ SOURCES)

# -------- Objectives --------
# Pythin C Extension Module:
add_library(dislocations SHARED)
target_sources(dislocations PRIVATE ${CMAKE_SOURCE_DIR}/src/bindings/dislocations.c ${SOURCES})

# Including header files of Python and Numpy:
target_include_directories(dislocations PRIVATE
    ${CMAKE_SOURCE_DIR}/include/
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
)

# ------ Build Options -------
# Complier Options, if you want debugging:
target_compile_options(dislocations PRIVATE
    -O2
    -fPIC
)

# Linking Options go here:
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(CMAKE_SHARED_LINKER_FLAGS "-undefined dynamic_lookup")
endif()
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    target_link_libraries(dislocations PRIVATE ${Python3_LIBRARIES})
endif()

set_target_properties(dislocations PROPERTIES
    PREFIX ""
    SUFFIX ".so"
)

# Get Python site-packages directory
execute_process(
    COMMAND "python" -c
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

install(
    TARGETS dislocations
    LIBRARY
    DESTINATION ${PYTHON_SITE_PACKAGES}
)
