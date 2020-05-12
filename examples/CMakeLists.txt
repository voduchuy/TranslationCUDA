set(PROGRAMS
        demo
        benchmark
        )

foreach ( PROG ${PROGRAMS})
    add_executable(${PROG} ${PROG}.cu)
    target_link_libraries(${PROG} CUTRANS)
    install(TARGETS ${PROG} DESTINATION ${EXAMPLES_INSTALL_DIR})
endforeach()
