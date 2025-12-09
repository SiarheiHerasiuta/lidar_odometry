# LiDAR Odometry with Probabilistic Kernel Optimization (PKO)

A high-performance real-time LiDAR odometry system designed for SLAM applications. It utilizes a 2-level hierarchical voxel map with precomputed surfels (hVox), point-to-plane ICP registration with Gauss-Newton optimization on Lie manifold, pose graph optimization for loop closure, and Pangolin for 3D visualization.

The system incorporates techniques from the following papers:

**Hierarchical Voxel Map with Precomputed Surfels:**
> S. Choi, D.-G. Park, S.-Y. Hwang, and T.-W. Kim, "Surfel-LIO: Fast LiDAR-Inertial Odometry with Pre-computed Surfels and Hierarchical Z-order Voxel Hashing," *arXiv preprint arXiv:2512.03397*, 2025.
>
> **Paper**: [https://arxiv.org/abs/2512.03397](https://arxiv.org/abs/2512.03397)

**Probabilistic Kernel Optimization (PKO)** for robust state estimation:

> S. Choi and T.-W. Kim, "Probabilistic Kernel Optimization for Robust State Estimation," *IEEE Robotics and Automation Letters*, vol. 10, no. 3, pp. 2998-3005, 2025, doi: 10.1109/LRA.2025.3536294.
> 
> **Paper**: [https://ieeexplore.ieee.org/document/10857458](https://ieeexplore.ieee.org/document/10857458)

ROS Wrapper: https://github.com/93won/lidar_odometry_ros_wrapper


## Features

- ⚡ **Ultra-fast processing** (~400 FPS on KITTI dataset)
- 🗺️ **2-Level VoxelMap** with precomputed surfels for O(1) correspondence lookup
- 🎯 **Point-to-Plane ICP** with Gauss-Newton optimization on Lie manifold
- 📈 **Adaptive M-estimator** for robust estimation (PKO)
- 🔧 **Asynchronous loop closure detection** with LiDAR Iris and pose graph optimization


## Demo

[![LiDAR Odometry Demo](https://img.youtube.com/vi/FANz9mhIAQQ/0.jpg)](https://www.youtube.com/watch?v=FANz9mhIAQQ)

*Click to watch the demo video showing real-time LiDAR odometry on KITTI dataset*

## Quick Start

### 1. Build Options

#### Native Build (Ubuntu 22.04)
```bash
git clone https://github.com/93won/lidar_odometry
cd lidar_odometry
chmod +x build.sh
./build.sh
```

### 2. Download Sample Data

Choose one of the sample datasets:

#### Option A: KITTI Dataset (Outdoor/Vehicle)
Download the sample KITTI sequence 07 from [Google Drive](https://drive.google.com/drive/folders/13YL4H9EIfL8oq1bVp0Csm0B7cMF3wT_0?usp=sharing) and extract to `data/kitti/`

#### Option B: MID360 Dataset (Indoor/Handheld)
Download the sample MID360 dataset from [Google Drive](https://drive.google.com/file/d/1psjoqrX9CtMvNCUskczUlsmaysh823CO/view?usp=sharing) and extract to `data/MID360/`

*MID360 dataset source: https://www.youtube.com/watch?v=u8siB0KLFLc*

### 3. Update Configuration

Choose the appropriate configuration file for your dataset:

#### For KITTI Dataset
Edit `config/kitti.yaml` to set your dataset paths:
```yaml
# Data paths - Update these paths to your dataset location
data_directory: "/path/to/your/kitti_dataset/sequences"
ground_truth_directory: "/path/to/your/kitti_dataset/poses"  
output_directory: "/path/to/your/output/directory"
seq: "07"  # Change this to your sequence number
```

#### For MID360 Dataset  
Edit `config/mid360.yaml` to set your dataset paths:
```yaml
# Data paths - Update these paths to your dataset location
data_directory: "/path/to/your/MID360_dataset"
output_directory: "/path/to/your/output/directory"
seq: "slam"  # Subdirectory name containing PLY files
```

### 4. Run LiDAR Odometry

Choose the appropriate executable for your dataset:

#### For KITTI Dataset (Outdoor/Vehicle)
```bash
cd build
./kitti_lidar_odometry ../config/kitti.yaml
```

#### For MID360 Dataset (Indoor/Handheld)
```bash
cd build
./mid360_lidar_odometry ../config/mid360.yaml
```

## Full KITTI Dataset

For complete evaluation, download the full KITTI dataset from:
- **Official Website**: [http://www.cvlibs.net/datasets/kitti/](http://www.cvlibs.net/datasets/kitti/)
- **Odometry Dataset**: [http://www.cvlibs.net/datasets/kitti/eval_odometry.php](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)

## Project Structure

```
lidar_odometry/
├── app/                          # Main applications
│   ├── kitti_lidar_odometry.cpp  # KITTI dataset runner
│   ├── mid360_lidar_odometry.cpp # PLY file runner (MID360, OS128, etc.)
│   └── player/                   # Dataset-specific players
│       ├── kitti_player.h/cpp
│       └── ply_player.h/cpp
├── src/
│   ├── database/                 # Data structures
│   │   ├── LidarFrame.h/cpp      # Point cloud frame representation
│   │   └── VoxelMap.h/cpp        # 2-Level hierarchical voxel map with surfels
│   ├── processing/               # Core algorithms
│   │   ├── Estimator.h/cpp       # Main odometry estimator
│   │   └── LoopClosureDetector.h/cpp
│   ├── optimization/             # Optimization modules
│   │   ├── IterativeClosestPointOptimizer.h/cpp  # Point-to-plane ICP
│   │   ├── AdaptiveMEstimator.h/cpp              # PKO robust estimator
│   │   └── PoseGraphOptimizer.h/cpp              # Loop closure optimization
│   ├── viewer/                   # Visualization
│   │   └── PangolinViewer.h/cpp
│   └── util/                     # Utilities
│       ├── ConfigUtils.h/cpp     # YAML configuration parser
│       ├── MathUtils.h/cpp       # Lie algebra (SO3/SE3), math functions
│       ├── PointCloudUtils.h/cpp # Point cloud operations
│       └── LogUtils.h            # Logging utilities
├── thirdparty/                   # External libraries
│   ├── pangolin/                 # 3D visualization
│   ├── nanoflann/                # KD-tree for nearest neighbor search
│   ├── LidarIris/                # Loop closure detection
│   └── unordered_dense/          # Fast hash map
├── config/                       # Configuration files
│   ├── kitti.yaml
│   ├── mid360.yaml
│   └── os128.yaml
└── build.sh                      # Build script
```

## System Requirements

- **Ubuntu 20.04/22.04** (recommended)
- **C++17 Compiler** (g++ or clang++)
- **CMake** (>= 3.16)

### Dependencies (installed via build.sh)
- Eigen3
- OpenGL / GLEW
- yaml-cpp
- ATLAS / SuiteSparse


## License

This project is released under the MIT License.

## References

### Hierarchical Voxel Map with Precomputed Surfels (hVox)

```bibtex
@article{choi2025surfel,
  title={Surfel-LIO: Fast LiDAR-Inertial Odometry with Pre-computed Surfels and Hierarchical Z-order Voxel Hashing},
  author={Choi, Seungwon and Park, Dong-Gyu and Hwang, Seo-Yeon and Kim, Tae-Wan},
  journal={arXiv preprint arXiv:2512.03397},
  year={2025}
}
```

### Probabilistic Kernel Optimization (PKO)

```bibtex
@ARTICLE{10857458,
  author={Choi, Seungwon and Kim, Tae-Wan},
  journal={IEEE Robotics and Automation Letters}, 
  title={Probabilistic Kernel Optimization for Robust State Estimation}, 
  year={2025},
  volume={10},
  number={3},
  pages={2998-3005},
  keywords={Kernel;Optimization;State estimation;Probabilistic logic;Tuning;Robustness;Cost function;Point cloud compression;Oceans;Histograms;Robust state estimation;SLAM},
  doi={10.1109/LRA.2025.3536294}
}
```

### Loop Closure Detection (LiDAR Iris)

```bibtex
@inproceedings{wang2020iris,
  title={LiDAR Iris for Loop-Closure Detection},
  author={Wang, Ying and Sun, Zezhou and Xu, Cheng-Zhong and Sarma, Sanjay and Yang, Jian and Kong, Hui},
  booktitle={2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={5769--5775},
  year={2020},
  organization={IEEE}
}
```
