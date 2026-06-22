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
    # -fno-sanitize=vptr: Ginkgo's CRTP solver bases call self() (a downcast to the
    # most-derived type) during base-class construction, before the derived vptr is
    # installed. UBSan's vptr check flags this benign UB in third-party headers; on GCC
    # the resulting diagnostic demangles type names via __cxa_demangle, whose buffer
    # LeakSanitizer then reports as a leak, failing the debug/gcc CI run.
    target_compile_options(${PROJECT_NAME}_sanitizers INTERFACE
        -fsanitize=address,undefined
        -fno-sanitize=vptr
        -fno-omit-frame-pointer
    )
    target_link_options(${PROJECT_NAME}_sanitizers INTERFACE
        -fsanitize=address,undefined
        -fno-sanitize=vptr
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

# Google Highway — explicit SIMD backing instrument/array.hpp. Pulled in here (not
# under tests/) so the whole library, and any consumer of it, links to Highway.
# Prefer a system install (fast in CI when cached); fall back to FetchContent and
# build from source if none is found. The two paths expose different target names:
# an installed Highway exports namespaced targets (hwy::hwy, hwy::hwy_contrib),
# while FetchContent defines un-namespaced ones (hwy, hwy_contrib). HIGHWAY_LIBRARIES
# captures whichever applies so the link line below stays the same.
find_package(hwy QUIET)
if(hwy_FOUND)
    set(HIGHWAY_LIBRARIES hwy::hwy hwy::hwy_contrib)
else()
    include(FetchContent)
    set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
    set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(HWY_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(HWY_ENABLE_CONTRIB ON CACHE BOOL "" FORCE) # contrib math/dot kernels
    FetchContent_Declare(highway
        GIT_REPOSITORY https://github.com/google/highway.git
        GIT_TAG 1.4.0
    )
    FetchContent_MakeAvailable(highway)
    set(HIGHWAY_LIBRARIES hwy hwy_contrib)
endif()


# LAPACKE — C interface to LAPACK, used by the dense Factorization wrappers.
# Provides the system BLAS/LAPACK via find_package(LAPACK) and then locates the
# LAPACKE header + library. Point CMake at a non-default install with
# -DCMAKE_PREFIX_PATH=... ; do not hardcode a path here.
find_package(LAPACK QUIET)
find_path(LAPACKE_INCLUDE_DIR lapacke.h)
find_library(LAPACKE_LIBRARY NAMES lapacke)
if(NOT LAPACK_FOUND OR NOT LAPACKE_INCLUDE_DIR OR NOT LAPACKE_LIBRARY)
    message(FATAL_ERROR
        "LAPACKE was not found.\n"
        "Install LAPACKE (e.g. 'liblapacke-dev' on Debian/Ubuntu) or point CMake "
        "at it with -DCMAKE_PREFIX_PATH=/path/to/lapacke.\n"
        "Need: lapacke.h (LAPACKE_INCLUDE_DIR) and the lapacke library "
        "(LAPACKE_LIBRARY), plus a LAPACK implementation (find_package(LAPACK))."
    )
endif()

# Ginkgo — the linear-algebra backend 
find_package(Ginkgo QUIET)
if(NOT Ginkgo_FOUND)
    message(FATAL_ERROR
        "Ginkgo was not found.\n"
        "Install or build Ginkgo (https://github.com/ginkgo-project/ginkgo) and "
        "tell CMake where it is, e.g.:\n"
        "  cmake --preset debug -DGinkgo_ROOT=/path/to/ginkgo/install\n"
        "The directory you pass must contain GinkgoConfig.cmake (an install "
        "prefix, or Ginkgo's build tree)."
    )
endif()

find_package(OpenMP REQUIRED)

# Header-only interface library
add_library(${PROJECT_NAME} INTERFACE)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME} INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Propagate Highway to every consumer; array.hpp includes its headers. Marked
# SYSTEM so the project's -Wall -Wextra -Wpedantic do not fire inside Highway.
target_link_libraries(${PROJECT_NAME} INTERFACE ${HIGHWAY_LIBRARIES} Ginkgo::ginkgo
    ${LAPACKE_LIBRARY} ${LAPACK_LIBRARIES})
target_include_directories(${PROJECT_NAME} SYSTEM INTERFACE
    ${LAPACKE_INCLUDE_DIR}
)
# An installed Highway is an IMPORTED target, so CMake already treats its include
# dirs as SYSTEM. The FetchContent targets are not imported, so add Highway's
# source dir explicitly as SYSTEM to keep our warnings out of its headers.
if(NOT hwy_FOUND)
    target_include_directories(${PROJECT_NAME} SYSTEM INTERFACE
        $<BUILD_INTERFACE:${highway_SOURCE_DIR}>
    )
endif()

# Ginkgo — the imported target carries its own usage requirements (include dirs,
# device sub-libraries, MPI). Linking it propagates Ginkgo to every consumer.
target_link_libraries(${PROJECT_NAME} INTERFACE Ginkgo::ginkgo)

# OpenMP
target_link_libraries(${PROJECT_NAME} INTERFACE OpenMP::OpenMP_CXX)

target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_23)

option(BUILD_TESTING "Build tests" ON)
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
