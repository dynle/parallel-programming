#!/bin/bash

# OpenMP C++ Compilation Script
# Usage: ./compile_openmp.sh <source_file> [output_file]

if [ $# -eq 0 ]; then
    echo "Usage: $0 <source_file> [output_file]"
    echo "Example: $0 testcpp.cpp"
    echo "Example: $0 testcpp.cpp my_program"
    exit 1
fi

SOURCE_FILE="$1"
OUTPUT_FILE="${2:-${SOURCE_FILE%.*}}"

# Check if source file exists
if [ ! -f "$SOURCE_FILE" ]; then
    echo "Error: Source file '$SOURCE_FILE' not found"
    exit 1
fi

echo "Compiling $SOURCE_FILE with OpenMP support..."

# Compile with OpenMP
clang++ -std=c++11 -Xpreprocessor -fopenmp \
    -I/opt/homebrew/opt/libomp/include \
    -L/opt/homebrew/opt/libomp/lib \
    -lomp "$SOURCE_FILE" -o "$OUTPUT_FILE"

if [ $? -eq 0 ]; then
    echo "✅ Compilation successful!"
    echo "   Output: $OUTPUT_FILE"
    echo "   To run: ./$OUTPUT_FILE"
else
    echo "❌ Compilation failed!"
    exit 1
fi 