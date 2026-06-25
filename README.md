# Instrument

[![CI](https://github.com/Miscibility/instrument/actions/workflows/ci.yml/badge.svg)](https://github.com/Miscibility/instrument/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Miscibility/instrument/branch/main/graph/badge.svg)](https://codecov.io/gh/Miscibility/instrument)

A header-only C++ library of utilities used throughout the Miscibility project.

## Building and testing

```bash
cmake --preset debug
cmake --build --preset debug --parallel
ctest --preset debug
```

Available configure presets: `debug`, `release`, `release-max`, `debug-tidy`, and
`coverage`.

## Code coverage

The `coverage` preset instruments the build with gcov and enables MPI so the MPI
code paths are measured too. The `coverage` target runs the full test suite and
merges every test executable's counters into a single report:

```bash
cmake --preset coverage
cmake --build --preset coverage --parallel
cmake --build --preset coverage --target coverage
```

This writes a Cobertura report to `build/coverage/coverage.xml` and prints a
line/branch summary. Requires `gcovr` and an MPI implementation (OpenMPI).
