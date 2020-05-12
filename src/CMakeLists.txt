set(EXT_LIBS
        -lcudart
        )

add_subdirectory(CuTransSimulator)

add_library(CUTRANS SHARED ${CUTRANS_OBJECTS} cutrans.h)
target_link_libraries(CUTRANS ${EXT_LIBS})

## Rules for installing cutrans library
SET_TARGET_PROPERTIES(CUTRANS
        PROPERTIES OUTPUT_NAME cutrans CLEAN_DIRECT_OUTPUT 1)
install(TARGETS CUTRANS DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(FILES cutrans.h DESTINATION include)
