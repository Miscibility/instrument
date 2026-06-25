cmake_minimum_required(VERSION 3.21)

project(Instrument
    VERSION 0.1.0
    DESCRIPTION "A C++ library containing useful utilities used throughout the Miscibility project"
    LANGUAGES CXX
)

# Prevent in-source builds
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR
        "In-source builds are not allowed.\n"
        "Run: cmake --preset debug  (or release / release-max)"
    )
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Symlink compile_commands.json into the source root for clangd
file(REMOVE "${CMAKE_SOURCE_DIR}/compile_commands.json")
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E create_symlink
        "${CMAKE_BINARY_DIR}/compile_commands.json"
        "${CMAKE_SOURCE_DIR}/compile_commands.json"
)

# Warnings — linked only to our targets so FetchContent deps are unaffected
add_library(${PROJECT_NAME}_warnings INTERFACE)
target_compile_options(${PROJECT_NAME}_warnings INTERFACE
    -Wall -Wextra -Wpedantic
)

# Sanitizers — empty unless ENABLE_SANITIZERS=ON (set by the debug preset)
add_library(${PROJECT_NAME}_sanitizers INTERFACE)
option(ENABLE_SANITIZERS "Enable AddressSanitizer and UBSan" OFF)
if(ENABLE_SANITIZERS)
    target_compile_options(${PROJECT_NAME}_sanitizers INTERFACE
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
    )
    target_link_options(${PROJECT_NAME}_sanitizers INTERFACE
        -fsanitize=address,undefined
    )
endif()

# Coverage — empty unless ENABLE_COVERAGE=ON (set by the coverage preset)
add_library(${PROJECT_NAME}_coverage INTERFACE)
option(ENABLE_COVERAGE "Instrument for gcov/llvm-cov code coverage" OFF)
if(ENABLE_COVERAGE)
    target_compile_options(${PROJECT_NAME}_coverage INTERFACE
        --coverage -O0 -g -fprofile-update=atomic
    )
    target_link_options(${PROJECT_NAME}_coverage INTERFACE
        --coverage
    )
endif()

# clang-tidy — enable with ENABLE_CLANG_TIDY=ON or add to a preset
option(ENABLE_CLANG_TIDY "Run clang-tidy during compilation" OFF)
if(ENABLE_CLANG_TIDY)
    find_program(CLANG_TIDY_EXE 
    NAMES 
        clang-tidy 
        clang-tidy-22
        clang-tidy-21
        clang-tidy-20
        clang-tidy-19
        clang-tidy-18
    REQUIRED)
    set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
endif()


# Header-only interface library
add_library(${PROJECT_NAME} INTERFACE)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME} INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_23)

# MPI — optional. When enabled, MISCIBILITY_INSTRUMENT_WITH_MPI is defined for
# anything linking the library, compiling in the report_mpi() API.
option(INSTRUMENT_WITH_MPI "Build with MPI support (enables report_mpi)" OFF)
if(INSTRUMENT_WITH_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
    target_link_libraries(${PROJECT_NAME} INTERFACE MPI::MPI_CXX)
    target_compile_definitions(${PROJECT_NAME} INTERFACE MISCIBILITY_INSTRUMENT_WITH_MPI)
endif()

option(BUILD_TESTING "Build tests" ON)
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

# Merged coverage report — `cmake --build --preset coverage --target coverage`
# runs the whole test suite and merges every test executable's .gcda counters
# into a single report, so coverage is correct no matter how many test files exist.
if(ENABLE_COVERAGE)
    find_program(GCOVR_EXE gcovr REQUIRED)
    # gcovr must use a gcov that matches the compiler that produced the .gcno data.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        string(REGEX MATCH "^[0-9]+" _gcc_major "${CMAKE_CXX_COMPILER_VERSION}")
        find_program(GCOV_EXE NAMES "gcov-${_gcc_major}" gcov REQUIRED)
        set(_gcov_exec "${GCOV_EXE}")
    else()
        # Clang: drive gcov-format data through llvm-cov.
        find_program(LLVM_COV_EXE NAMES llvm-cov llvm-cov-22 REQUIRED)
        set(_gcov_exec "${LLVM_COV_EXE} gcov")
    endif()
    add_custom_target(coverage
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        COMMAND ${GCOVR_EXE}
            --root "${CMAKE_SOURCE_DIR}"
            --filter "${CMAKE_SOURCE_DIR}/include/"
            --gcov-executable "${_gcov_exec}"
            --exclude-unreachable-branches
            --print-summary
            --xml-pretty --output "${CMAKE_BINARY_DIR}/coverage.xml"
            --txt "${CMAKE_BINARY_DIR}/coverage.txt"
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
        COMMENT "Running tests and generating merged coverage report (coverage.xml)"
        VERBATIM
    )
endif()
