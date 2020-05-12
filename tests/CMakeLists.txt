set(TEST_PROGRAMS
        test_simple_gene
        )

foreach ( PROG ${TEST_PROGRAMS})
    add_executable(${PROG} ${PROG}.cu)
    target_link_libraries(${PROG} CUTRANS)
    install(TARGETS ${PROG} DESTINATION ${TESTS_INSTALL_DIR})
endforeach()
