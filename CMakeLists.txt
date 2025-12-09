cmake_minimum_required(VERSION 3.16)
project(lidar_odometry)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build" FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set OpenGL preference to avoid warnings
set(OpenGL_GL_PREFERENCE GLVND)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Find system packages
find_package(PkgConfig REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED)

# Build third-party dependencies from source
set(THIRDPARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty)

# Build Pangolin from thirdparty
set(BUILD_PANGOLIN_EXAMPLES OFF CACHE BOOL "Build Pangolin examples" FORCE)
set(BUILD_PANGOLIN_TESTS OFF CACHE BOOL "Build Pangolin tests" FORCE)
set(BUILD_PANGOLIN_TOOLS OFF CACHE BOOL "Build Pangolin tools" FORCE)
add_subdirectory(${THIRDPARTY_DIR}/pangolin ${CMAKE_BINARY_DIR}/pangolin EXCLUDE_FROM_ALL)

# Sophus is header-only, no need to build
# Just include the headers

# Build GTSAM from thirdparty
set(GTSAM_DIR ${THIRDPARTY_DIR}/gtsam)
set(GTSAM_BUILD_EXAMPLES_ALWAYS OFF CACHE BOOL "Build GTSAM examples" FORCE)
set(GTSAM_BUILD_TESTS OFF CACHE BOOL "Build GTSAM tests" FORCE)
set(GTSAM_BUILD_UNSTABLE OFF CACHE BOOL "Build GTSAM unstable" FORCE)
set(GTSAM_BUILD_WITH_MARCH_NATIVE OFF CACHE BOOL "Build with march native" FORCE)
set(GTSAM_INSTALL_CPPUNITLITE OFF CACHE BOOL "Install CppUnitLite" FORCE)
set(GTSAM_INSTALL_GEOGRAPHICLIB OFF CACHE BOOL "Install GeographicLib" FORCE)
set(GTSAM_USE_SYSTEM_EIGEN ON CACHE BOOL "Use system Eigen" FORCE)
set(GTSAM_WITH_TBB OFF CACHE BOOL "Use TBB" FORCE)
set(GTSAM_BUILD_WRAP OFF CACHE BOOL "Build wrap" FORCE)
set(GTSAM_BUILD_PYTHON OFF CACHE BOOL "Build Python wrapper" FORCE)
set(GTSAM_INSTALL_MATLAB_TOOLBOX OFF CACHE BOOL "Install MATLAB toolbox" FORCE)
add_subdirectory(${GTSAM_DIR} ${CMAKE_BINARY_DIR}/gtsam EXCLUDE_FROM_ALL)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${EIGEN3_INCLUDE_DIRS}
    ${THIRDPARTY_DIR}/pangolin/include
    ${CMAKE_BINARY_DIR}/thirdparty/pangolin/include
    ${THIRDPARTY_DIR}/gtsam
    ${CMAKE_BINARY_DIR}/gtsam
    ${THIRDPARTY_DIR}/nanoflann
    ${THIRDPARTY_DIR}/LidarIris
    ${OpenCV_INCLUDE_DIRS}
)

# Common source files for all executables
set(COMMON_SOURCES
    # Database module
    src/database/LidarFrame.cpp
    
    # Processing module
    src/processing/Estimator.cpp
    src/processing/LoopClosureDetector.cpp
    
    # Map module
    src/map/VoxelMap.cpp
    
    # Util module
    src/util/LieUtils.cpp
    
    # Third-party
    thirdparty/LidarIris/LidarIris.cpp
    thirdparty/LidarIris/fftm/fftm.cpp
    
    # Optimization module
    src/optimization/AdaptiveMEstimator.cpp
    src/optimization/PoseGraphOptimizer.cpp
    src/optimization/IterativeClosestPointOptimizer.cpp
    
    # Viewer module
    src/viewer/PangolinViewer.cpp
    
    # Utility module
    src/util/ConfigUtils.cpp
    src/util/MathUtils.cpp
    src/util/PointCloudUtils.cpp
)

# KITTI LiDAR odometry executable
set(KITTI_SOURCES
    ${COMMON_SOURCES}
    app/kitti_lidar_odometry.cpp
    app/player/kitti_player.cpp
)

# LiDAR odometry executable (PLY player)
set(LIDAR_ODOMETRY_SOURCES
    ${COMMON_SOURCES}
    app/lidar_odometry.cpp
    app/player/ply_player.cpp
)

# Create executables
add_executable(kitti_lidar_odometry ${KITTI_SOURCES})
add_executable(lidar_odometry ${LIDAR_ODOMETRY_SOURCES})

# Link Eigen3 to targets
if(TARGET Eigen3::Eigen)
    target_link_libraries(kitti_lidar_odometry Eigen3::Eigen)
    target_link_libraries(lidar_odometry Eigen3::Eigen)
else()
    # Fallback for older CMake versions
    target_include_directories(kitti_lidar_odometry PRIVATE ${EIGEN3_INCLUDE_DIR})
    target_include_directories(lidar_odometry PRIVATE ${EIGEN3_INCLUDE_DIR})
endif()

# Link libraries for KITTI executable
target_link_libraries(kitti_lidar_odometry
    OpenGL::GL
    GLEW::GLEW
    Threads::Threads
    gtsam
    tbb
    pango_display
    pango_windowing
    pango_opengl
    pango_core
    pango_image
    ${OpenCV_LIBS}
    m  # Math library for M_PI
    stdc++fs  # For std::filesystem
)

# Link libraries for lidar_odometry executable
target_link_libraries(lidar_odometry
    OpenGL::GL
    GLEW::GLEW
    Threads::Threads
    gtsam
    tbb
    pango_display
    pango_windowing
    pango_opengl
    pango_core
    pango_image
    ${OpenCV_LIBS}
    m  # Math library for M_PI
    stdc++fs  # For std::filesystem
)

# Install targets
install(TARGETS kitti_lidar_odometry lidar_odometry
    RUNTIME DESTINATION bin
)

# Install configuration files
install(DIRECTORY config/
    DESTINATION share/lidar_odometry/config
    FILES_MATCHING PATTERN "*.yaml"
)

# Print configuration info
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Eigen3 include dir: ${EIGEN3_INCLUDE_DIR}")
message(STATUS "ThirdParty directory: ${THIRDPARTY_DIR}")
