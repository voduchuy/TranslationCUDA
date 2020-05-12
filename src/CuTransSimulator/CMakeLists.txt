set(module_name CuTransSimulator)

set(${module_name}_files
        CuTransKernels
        CuTransSimulator
        )

foreach(f ${${module_name}_files})
    list(APPEND src_files ${f}.cu)
    list(APPEND header_files ${f}.h)
endforeach()

add_library(${module_name}_OBJ OBJECT ${src_files} ${header_files})
target_compile_features(${module_name}_OBJ PUBLIC cxx_std_11)
set_target_properties(${module_name}_OBJ PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${module_name}_OBJ PROPERTIES POSITION_INDEPENDENT_CODE ON)

list(APPEND CUTRANS_OBJECTS "$<TARGET_OBJECTS:${module_name}_OBJ>")
set(CUTRANS_OBJECTS ${CUTRANS_OBJECTS} PARENT_SCOPE)

install(FILES ${header_files} DESTINATION include/${module_name})