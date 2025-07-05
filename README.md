# OpenMP C++ Setup and Usage

## Environment Setup

Your system is now configured to compile and run OpenMP C++ programs with:

- **Compiler**: clang++ (Apple clang version 14.0.3)
- **OpenMP Library**: libomp 20.1.7 (installed via Homebrew)
- **Alternative**: g++-15 (Homebrew GCC 15.1.0) - available but may have linking issues

## Files in this directory
- `compile_openmp.sh` - Compilation script for OpenMP programs

## Quick Start

### 1. Compile and run the test program:
```bash
./compile_openmp.sh testcpp.cpp
./testcpp
```

### 2. Manual compilation:
```bash
clang++ -std=c++11 -Xpreprocessor -fopenmp \
    -I/opt/homebrew/opt/libomp/include \
    -L/opt/homebrew/opt/libomp/lib \
    -lomp testcpp.cpp -o testcpp
```

## Compilation Script Usage

```bash
# Basic usage
./compile_openmp.sh source_file.cpp

# Specify output file
./compile_openmp.sh source_file.cpp my_program

# Examples
./compile_openmp.sh testcpp.cpp
./compile_openmp.sh my_program.cpp my_exe
```

## Notes

- The linking warning about macOS version compatibility can be safely ignored
- OpenMP version 5.0 (201811) is available
- System has 10 processors available
- Default thread count can be controlled with `omp_set_num_threads()`

## Creating New OpenMP Programs

When creating new OpenMP C++ programs, make sure to:

1. Include the OpenMP header: `#include <omp.h>`
2. Use OpenMP pragmas: `#pragma omp parallel`, `#pragma omp for`, etc.
3. Compile with the provided script or manual compilation command
4. Consider thread safety when accessing shared variables 