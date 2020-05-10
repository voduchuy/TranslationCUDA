CMAKE_MINIMUM_REQUIRED(VERSION 3.8)

set(CUDA_STATIC_LIBS )

set(CUDA_SSA_LIBS )

set(TESTS
        ssa
        )

set(BENCHMARKS
        )

set(PROBLEMS

        )

set (EXTERNAL_LIBS
        -lcusparse -lcudart -lcublas
        )

if (APPLE)
    PROJECT(TranslationCUDA)

    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
    #set(CMAKE_CXX_COMPILER g++-8)

    message("Apple")

    find_package(CUDA)

    set(CMAKE_CUDA_FLAGS "-lineinfo")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O2")

    set(CMAKE_CUDA_SEPARABLE_COMPILATION true)

    set(CMAKE_CXX_FLAGS " -O3 -std=c++11 -g")

    set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR})

    # Where to put .a files (archive for static libraries)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

    # Where to put .so files (dynamic libraries)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

    # Where to put the executable files
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

    # Set the folders to include
    include_directories(${CMAKE_SOURCE_DIR}/src)
    #include_directories(${CMAKE_SOURCE_DIR}/include)

    include_directories(/usr/local/include)

    foreach (lib ${CUDA_STATIC_LIBS})
        cuda_add_library(${lib} src/${lib}.h src/${lib}.cu)
        target_compile_features(${lib} PUBLIC cxx_std_11)
        set_target_properties(${lib} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    endforeach ()

    foreach (lib ${CUDA_FSP_LIBS})
        cuda_add_library(${lib} SHARED src/${lib}.h src/${lib}.cu)
        target_compile_features(${lib} PUBLIC cxx_std_11)
        set_target_properties(${lib} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        target_link_libraries(${lib} ${CUDA_STATIC_LIBS})
    endforeach ()
    #target_link_libraries(FSPMat cme_util)

    add_executable(dummy tests/zero_test.cpp)
    target_link_libraries(dummy)


    foreach (PROG ${TESTS})
        cuda_add_executable(${PROG} ${CMAKE_SOURCE_DIR}/tests/${PROG}.cu)
        target_link_libraries(${PROG} ${CUDA_FSP_LIBS} -lcusparse)
        target_compile_features(${PROG} PUBLIC cxx_std_11)
        set_target_properties(${PROG} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    endforeach ()

    foreach (PROG ${BENCHMARKS})
        cuda_add_executable(${PROG} ${CMAKE_SOURCE_DIR}/benchmarks/${PROG}.cu)
        target_link_libraries(${PROG} ${CUDA_FSP_LIBS} ${CUDA_STATIC_LIBS} -lcusparse -lcudart -lcublas )
        target_compile_features(${PROG} PUBLIC cxx_std_11)
        set_target_properties(${PROG} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        set_target_properties(${PROG} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endforeach ()


    foreach (PROG ${PROBLEMS})
        cuda_add_executable(${PROG} ${CMAKE_SOURCE_DIR}/problems/${PROG}.cu)
        target_link_libraries(${PROG} ${CUDA_FSP_LIBS} ${CUDA_STATIC_LIBS} -lcusparse -lcudart -lcublas )
        target_compile_features(${PROG} PUBLIC cxx_std_11)
        set_target_properties(${PROG} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        set_target_properties(${PROG} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endforeach ()

else ()


    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
    PROJECT(TranslationCUDA LANGUAGES CXX CUDA)

    #    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --compiler-options '-fPIC'")
    set(CMAKE_CUDA_FLAGS "-lineinfo -O2 -DKEXPV_VERBOSE")
    set(CMAKE_CUDA_SEPARABLE_COMPILATION true)
    set(CMAKE_CXX_FLAGS " -g -O3 -std=c++11 ")

    set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR})

    # Where to put .a files (archive for static libraries)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

    # Where to put .so files (dynamic libraries)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

    # Where to put the executable files
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

    # Set the folders to include
    include_directories(${CMAKE_SOURCE_DIR}/src)
    #include_directories(${CMAKE_SOURCE_DIR}/include)

    include_directories(/usr/local/include)
    include_directories(/usr/local/cuda/include)
    include_directories(src)

    add_executable(dummy tests/zero_test.cpp)
    target_link_libraries(dummy)

    foreach (lib ${CUDA_STATIC_LIBS})
        add_library(${lib} STATIC src/${lib}.h src/${lib}.cu)
        target_compile_features(${lib} PUBLIC cxx_std_11)
        set_target_properties(${lib} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        set_target_properties(${lib} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endforeach ()

    foreach (lib ${CUDA_SSA_LIBS})
        add_library(${lib} src/${lib}.h src/${lib}.cu)
        target_compile_features(${lib} PUBLIC cxx_std_11)
        set_target_properties(${lib} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        set_target_properties(${lib} PROPERTIES POSITION_INDEPENDENT_CODE ON)
        target_link_libraries(${lib} ${CUDA_STATIC_LIBS} -larmadillo)
    endforeach ()

    foreach (PROG ${TESTS})
        add_executable(${PROG} ${CMAKE_SOURCE_DIR}/tests/${PROG}.cu)
        target_link_libraries(${PROG} ${CUDA_FSP_LIBS} ${CUDA_STATIC_LIBS} ${EXTERNAL_LIBS} )
        target_compile_features(${PROG} PUBLIC cxx_std_11)
        set_target_properties(${PROG} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        set_target_properties(${PROG} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endforeach ()

    foreach (PROG ${BENCHMARKS})
        add_executable(${PROG} ${CMAKE_SOURCE_DIR}/benchmarks/${PROG}.cu)
        target_link_libraries(${PROG} ${CUDA_FSP_LIBS} ${CUDA_STATIC_LIBS} ${EXTERNAL_LIBS} )
        target_compile_features(${PROG} PUBLIC cxx_std_11)
        set_target_properties(${PROG} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        set_target_properties(${PROG} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endforeach ()

    foreach (PROG ${PROBLEMS})
        add_executable(${PROG} ${CMAKE_SOURCE_DIR}/problems/${PROG}.cu)
        target_link_libraries(${PROG} ${CUDA_FSP_LIBS} ${CUDA_STATIC_LIBS} ${EXTERNAL_LIBS} )
        target_compile_features(${PROG} PUBLIC cxx_std_11)
        set_target_properties(${PROG} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        set_target_properties(${PROG} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endforeach ()
endif ()