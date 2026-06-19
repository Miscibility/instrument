include(FetchContent)

FetchContent_Declare(ut
    GIT_REPOSITORY https://github.com/boost-ext/ut.git
    GIT_TAG v2.3.1
)
FetchContent_MakeAvailable(ut)

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
        Boost::ut
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
    # Ginkgo's hwloc-backed machine_topology singleton is never freed, so
    # LeakSanitizer flags it on every Ginkgo-using test; suppress those while
    # keeping leak detection for our own code.
    set_tests_properties(${test_name} PROPERTIES
        ENVIRONMENT "LSAN_OPTIONS=suppressions=${CMAKE_CURRENT_SOURCE_DIR}/lsan_suppressions.txt:print_suppressions=0"
    )
endforeach()

# The Ginkgo smoke test triggers Ginkgo's hwloc machine_topology singleton, whose
# process-lifetime allocations LeakSanitizer reports as leaks. Point LSan at a
# suppression file for just this test so the sanitizer build stays green.
if(TEST test_ginkgo_smoke)
    set_tests_properties(test_ginkgo_smoke PROPERTIES
        ENVIRONMENT "LSAN_OPTIONS=suppressions=${CMAKE_CURRENT_SOURCE_DIR}/ginkgo.supp"
    )
endif()
