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

# clang-tidy — enable with ENABLE_CLANG_TIDY=ON or add to a preset
option(ENABLE_CLANG_TIDY "Run clang-tidy during compilation" OFF)
if(ENABLE_CLANG_TIDY)
    find_program(CLANG_TIDY_EXE NAMES clang-tidy REQUIRED)
    set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
endif()

# Google Highway — explicit SIMD backing instrument/vector.hpp. Fetched here (not
# under tests/) so the whole library, and any consumer of it, links to Highway.
include(FetchContent)
set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_CONTRIB ON CACHE BOOL "" FORCE) # contrib math/dot kernels
FetchContent_Declare(highway
    GIT_REPOSITORY https://github.com/google/highway.git
    GIT_TAG 1.2.0
)
FetchContent_MakeAvailable(highway)

# Header-only interface library
add_library(${PROJECT_NAME} INTERFACE)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME} INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Propagate Highway to every consumer; vector.hpp includes its headers. Marked
# SYSTEM so the project's -Wall -Wextra -Wpedantic do not fire inside Highway.
target_link_libraries(${PROJECT_NAME} INTERFACE hwy hwy_contrib)
target_include_directories(${PROJECT_NAME} SYSTEM INTERFACE
    $<BUILD_INTERFACE:${highway_SOURCE_DIR}>
)

target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_23)

option(BUILD_TESTING "Build tests" ON)
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
