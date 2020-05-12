CMAKE_MINIMUM_REQUIRED(VERSION 3.8)
PROJECT(TranslationCUDA LANGUAGES CXX CUDA)
set(CMAKE_CUDA_SEPARABLE_COMPILATION true)

# Load some default paths for installation (e.g., /usr/local/)
INCLUDE(GNUInstallDirs)

# Default build type is release
set(default_build_type "Release")

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
            STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
            "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif ()

if (NOT CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install CACHE PATH "Where to install the library." FORCE)
endif (NOT CMAKE_INSTALL_PREFIX)

# Set some default flags for build types
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-std=c++11 -O2 -DNDEBUG")

include_directories(src)
include_directories(ext/cub-1.8.0/cub)

# Decide whether to build tests and where to install them
OPTION(BUILD_TESTS "Build tests" ON)
if (BUILD_TESTS)
    set(TESTS_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/tests CACHE STRING "Installation directory for tests.")
endif ()

# Decide whether to build examples
OPTION(BUILD_EXAMPLES "Build examples" ON)
if (BUILD_EXAMPLES)
    set(EXAMPLES_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/examples CACHE STRING "Installation directory for examples.")
endif ()

# Where to put the built files prior to installation--We'll put it right at the 'build' directory
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_subdirectory(src)

if (BUILD_TESTS)
    add_subdirectory(tests)
endif()

if (BUILD_EXAMPLES)
    add_subdirectory(examples)
endif (BUILD_EXAMPLES)

