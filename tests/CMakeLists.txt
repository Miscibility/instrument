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
endforeach()
