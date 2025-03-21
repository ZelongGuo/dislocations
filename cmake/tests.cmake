# This is for CMakeLists only for Test
# Zelong Guo, @ Potsdam, DE

# ------------------ Source Files ------------------

aux_source_directory (${CMAKE_SOURCE_DIR}/src/core/ SOURCES)

# set(SOURCES ${CMAKE_SOURCE_DIR}/src/core/)

# ------------------ Objectives  ------------------
# Executable files:
add_executable(test)

# Source files:
target_sources(test PUBLIC ${CMAKE_SOURCE_DIR}/tests/test.c ${SOURCES})

# Header files (directories):
target_include_directories(test PUBLIC ${CMAKE_SOURCE_DIR}/include/)

# -------------------- Build Options --------------------
# Complier Options, if you want debugging using "-g":
target_compile_options(test PUBLIC
    -g
    -O2
    -fPIC
)

