#!/bin/bash
# Rebuild _NIRScanner.so after C++ source changes
# Run this on the Raspberry Pi (ARM aarch64)

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$SCRIPT_DIR/src"
BUILD_DIR="$SRC_DIR/cmake-build-debug"
PYTHON_INCLUDE=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
PYTHON_LIBDIR=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")

echo "=== Step 1: Rebuild static libraries (C + C++ source) ==="
cd "$BUILD_DIR"
cmake .. && make -j4

echo "=== Step 2: Regenerate SWIG wrapper ==="
cd "$SRC_DIR"
swig -python -c++ -outdir . NIRScanner.i
echo "SWIG wrapper regenerated: NIRScanner_wrap.cxx"

echo "=== Step 3: Compile SWIG wrapper into _NIRScanner.so ==="
c++ -shared -fPIC -O2 \
    -I"$PYTHON_INCLUDE" \
    -I"$SRC_DIR" \
    -I"$BUILD_DIR" \
    -o "$SCRIPT_DIR/_NIRScanner.so" \
    "$SRC_DIR/NIRScanner_wrap.cxx" \
    "$BUILD_DIR/libobjcpp.a" \
    "$BUILD_DIR/libobjc.a" \
    -ludev -lpthread

echo "=== Done ==="
echo "New _NIRScanner.so built at: $SCRIPT_DIR/_NIRScanner.so"
ls -la "$SCRIPT_DIR/_NIRScanner.so"
