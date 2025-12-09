#!/bin/bash

set -e  # Exit on any error

echo "=============================================="
echo "         LiDAR Odometry Build Script"
echo "=============================================="

# Get number of CPU cores for parallel compilation (use half of available cores)
NPROC=$(($(nproc) / 2))
if [ $NPROC -lt 1 ]; then
    NPROC=1
fi
echo "Using $NPROC cores for compilation (half of available)"

# Install system dependencies
echo ""
echo "Step 0: Installing system dependencies..."
echo "========================================"

# Check if running in Docker container
if [ -n "$DOCKER_CONTAINER" ]; then
    echo "Running in Docker container - skipping apt updates (dependencies pre-installed)"
else
    sudo apt update
    sudo apt install -y \
        cmake \
        build-essential \
        libeigen3-dev \
        libgl1-mesa-dev \
        libglu1-mesa-dev \
        libglew-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libatlas-base-dev \
        libsuitesparse-dev \
        libyaml-cpp-dev

    echo "System dependencies installed successfully!"
fi

# Build third-party dependencies
echo ""
echo "Step 1: Checking third-party dependencies..."
echo "=============================================="

# Check if thirdparty directory exists
if [ ! -d "thirdparty" ]; then
    echo "Error: thirdparty directory not found!"
    echo "Please ensure the thirdparty directory with dependencies exists."
    exit 1
fi

# Pangolin, Sophus, spdlog are built via CMake add_subdirectory
echo "Third-party libraries will be built automatically via CMake"

# Build main project
echo ""
echo "Step 2: Building main project..."
echo "================================="

# Create build directory for main project
if [ ! -d "build" ]; then
    mkdir build
fi

cd build

# Configure and build main project
cmake ..
make -j$NPROC

echo ""
echo "=============================================="
echo "  Build completed successfully!"
echo "=============================================="
echo ""
echo "To run the LiDAR odometry system:"
echo "  cd build"
echo ""
echo "For KITTI dataset:"
echo "  ./kitti_lidar_odometry ../config/kitti.yaml"
echo ""
echo "For PLY files (MID360, OS128, etc):"
echo "  ./lidar_odometry ../config/mid360.yaml"
echo ""
