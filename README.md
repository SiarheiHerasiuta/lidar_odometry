# Fast LiDAR SLAM

A high-performance real-time LiDAR SLAM system (~400 FPS on KITTI).


[![Demo](https://img.youtube.com/vi/eBdYP55IMEY/0.jpg)](https://www.youtube.com/watch?v=eBdYP55IMEY)

## Features

- 2-Level hierarchical voxel map with precomputed surfels
- Point-to-plane ICP with Gauss-Newton optimization on Lie manifold
- Adaptive M-estimator for robust estimation (PKO)
- Loop closure detection with LiDAR Iris

## Build

```bash
git clone https://github.com/93won/lidar_odometry
cd lidar_odometry
./build.sh
```

## Run

```bash
# KITTI dataset
./build/kitti_lidar_odometry config/kitti.yaml

# MID360 / PLY files
./build/lidar_odometry config/mid360.yaml
```

## Sample Data

- [KITTI Sequence 07](https://drive.google.com/drive/folders/13YL4H9EIfL8oq1bVp0Csm0B7cMF3wT_0?usp=sharing)
- [MID360 Dataset](https://drive.google.com/file/d/1psjoqrX9CtMvNCUskczUlsmaysh823CO/view?usp=sharing)

## ROS Wrapper

https://github.com/93won/lidar_odometry_ros_wrapper

## References

```bibtex
@article{choi2025surfel,
  title={Surfel-LIO: Fast LiDAR-Inertial Odometry with Pre-computed Surfels and Hierarchical Z-order Voxel Hashing},
  author={Choi, Seungwon and Park, Dong-Gyu and Hwang, Seo-Yeon and Kim, Tae-Wan},
  journal={arXiv preprint arXiv:2512.03397},
  year={2025}
}

@article{choi2025pko,
  author={Choi, Seungwon and Kim, Tae-Wan},
  journal={IEEE Robotics and Automation Letters}, 
  title={Probabilistic Kernel Optimization for Robust State Estimation}, 
  year={2025},
  volume={10},
  number={3},
  pages={2998-3005},
  doi={10.1109/LRA.2025.3536294}
}

@inproceedings{wang2020iris,
  title={LiDAR Iris for Loop-Closure Detection},
  author={Wang, Ying and Sun, Zezhou and Xu, Cheng-Zhong and Sarma, Sanjay and Yang, Jian and Kong, Hui},
  booktitle={IROS},
  year={2020}
}
```

## License

MIT License
