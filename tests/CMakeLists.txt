include(FetchContent)

FetchContent_Declare(ut
    GIT_REPOSITORY https://github.com/boost-ext/ut.git
    GIT_TAG v2.3.1
)
FetchContent_MakeAvailable(ut)

# Number of ranks used to run the MPI (multi-rank) tests.
set(INSTRUMENT_MPI_TEST_RANKS 2 CACHE STRING "MPI ranks used for *_mpi tests")

# CONFIGURE_DEPENDS: cmake reruns glob if files are added/removed
file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

foreach(test_source IN LISTS TEST_SOURCES)
    # NAME_WE: filename Without Extension (e.g. test_foo.cpp -> test_foo)
    get_filename_component(test_name "${test_source}" NAME_WE)
    add_executable(${test_name} "${test_source}")
    target_link_libraries(${test_name} PRIVATE
        ${PROJECT_NAME}::${PROJECT_NAME}
        ${PROJECT_NAME}_warnings
        ${PROJECT_NAME}_sanitizers
        ${PROJECT_NAME}_coverage
        Boost::ut
    )

    # Convention: a *_mpi test exercises real MPI collectives and is launched on
    # several ranks; every other test is serial and launched on a single rank.
    if(test_name MATCHES "_mpi$")
        # Multi-rank tests only make sense when the library is built with MPI.
        if(INSTRUMENT_WITH_MPI)
            add_test(NAME ${test_name}
                COMMAND ${MPIEXEC_EXECUTABLE}
                    ${MPIEXEC_NUMPROC_FLAG} ${INSTRUMENT_MPI_TEST_RANKS}
                    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${test_name}> ${MPIEXEC_POSTFLAGS}
            )
        endif()
    elseif(INSTRUMENT_WITH_MPI)
        # Serial tests run on a single MPI rank when MPI is available.
        add_test(NAME ${test_name}
            COMMAND ${MPIEXEC_EXECUTABLE}
                ${MPIEXEC_NUMPROC_FLAG} 1
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${test_name}> ${MPIEXEC_POSTFLAGS}
        )
    else()
        add_test(NAME ${test_name} COMMAND ${test_name})
    endif()
endforeach()
