/**
 * @file      MathUtils.h
 * @brief     Math utility functions for LiDAR odometry
 * @author    Seungwon Choi
 * @date      2025-09-24
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#pragma once

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <cmath>

namespace lidar_odometry {
namespace util {

/**
 * @brief Utility functions for mathematical operations
 */
class MathUtils {
public:
    /**
     * @brief Normalize a rotation matrix using SVD to ensure orthogonality
     * 
     * This function takes a potentially non-orthogonal matrix and finds the
     * closest orthogonal matrix using Singular Value Decomposition (SVD).
     * This is useful when dealing with accumulated numerical errors in rotation matrices.
     * 
     * @param R Input rotation matrix (3x3)
     * @return Normalized orthogonal rotation matrix with determinant +1
     */
    static Eigen::Matrix3f normalize_rotation_matrix(const Eigen::Matrix3f& R);

    /**
     * @brief Double precision version of normalize_rotation_matrix
     * @param R Input rotation matrix (3x3, double precision)
     * @return Normalized orthogonal rotation matrix with determinant +1
     */
    static Eigen::Matrix3d normalize_rotation_matrix(const Eigen::Matrix3d& R);

    /**
     * @brief Check if a matrix is orthogonal within tolerance
     * @param R Input matrix
     * @param tolerance Tolerance for orthogonality check
     * @return True if matrix is orthogonal within tolerance
     */
    static bool is_orthogonal(const Eigen::Matrix3f& R, float tolerance = 1e-6f);

    /**
     * @brief Check if a matrix is orthogonal within tolerance (double precision)
     * @param R Input matrix
     * @param tolerance Tolerance for orthogonality check
     * @return True if matrix is orthogonal within tolerance
     */
    static bool is_orthogonal(const Eigen::Matrix3d& R, double tolerance = 1e-12);

    /**
     * @brief Clamp angle to [-pi, pi] range
     * @param angle Input angle in radians
     * @return Clamped angle in [-pi, pi]
     */
    static float wrap_to_pi(float angle);

    /**
     * @brief Clamp angle to [-pi, pi] range (double precision)
     * @param angle Input angle in radians
     * @return Clamped angle in [-pi, pi]
     */
    static double wrap_to_pi(double angle);

    /**
     * @brief Wrap angle to [-PI, PI] range
     * @param angle Input angle in radians
     * @return Wrapped angle
     */
    static double wrap_angle(double angle);

    /**
     * @brief Convert degrees to radians
     * @param degrees Input angle in degrees
     * @return Angle in radians
     */
    static double deg_to_rad(double degrees);

    /**
     * @brief Convert radians to degrees
     * @param radians Input angle in radians
     * @return Angle in degrees
     */
    static double rad_to_deg(double radians);

    /**
     * @brief Check if two floating point numbers are approximately equal
     * @param a First number
     * @param b Second number
     * @param epsilon Tolerance
     * @return True if numbers are approximately equal
     */
    template<typename T>
    static bool is_approx_equal(T a, T b, T epsilon);

    /**
     * @brief Clamp value between min and max
     * @param value Value to clamp
     * @param min_val Minimum value
     * @param max_val Maximum value
     * @return Clamped value
     */
    template<typename T>
    static T clamp(T value, T min_val, T max_val);
};

} // namespace util
} // namespace lidar_odometry
