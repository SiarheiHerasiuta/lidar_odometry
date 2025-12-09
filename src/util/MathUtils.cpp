/**
 * @file      MathUtils.cpp
 * @brief     Implementation of math utility functions
 * @author    Seungwon Choi
 * @date      2025-09-24
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "MathUtils.h"
#include <cmath>
#include "util/LogUtils.h"

namespace lidar_odometry {
namespace util {

Eigen::Matrix3f MathUtils::normalize_rotation_matrix(const Eigen::Matrix3f& R) {
    // Perform SVD: R = U * S * V^T
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3f U = svd.matrixU();
    Eigen::Matrix3f V = svd.matrixV();
    
    // The closest orthogonal matrix is U * V^T
    Eigen::Matrix3f R_normalized = U * V.transpose();
    
    // Ensure proper rotation (determinant = +1, not reflection)
    if (R_normalized.determinant() < 0) {
        // If determinant is negative, flip the last column of U
        U.col(2) *= -1;
        R_normalized = U * V.transpose();
    }
    
    // Verify the result
    float det = R_normalized.determinant();
    float orthogonality_error = (R_normalized * R_normalized.transpose() - Eigen::Matrix3f::Identity()).norm();
    
    // if (std::abs(det - 1.0f) > 1e-6f || orthogonality_error > 1e-6f) {
    //     LOG_WARN("[MathUtils] Rotation normalization may have failed: det={}, orth_error={}", det, orthogonality_error);
    // }
    
    return R_normalized;
}

Eigen::Matrix3d MathUtils::normalize_rotation_matrix(const Eigen::Matrix3d& R) {
    // Perform SVD: R = U * S * V^T
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    
    // The closest orthogonal matrix is U * V^T
    Eigen::Matrix3d R_normalized = U * V.transpose();
    
    // Ensure proper rotation (determinant = +1, not reflection)
    if (R_normalized.determinant() < 0) {
        // If determinant is negative, flip the last column of U
        U.col(2) *= -1;
        R_normalized = U * V.transpose();
    }
    
    // Verify the result
    double det = R_normalized.determinant();
    double orthogonality_error = (R_normalized * R_normalized.transpose() - Eigen::Matrix3d::Identity()).norm();
    
    // if (std::abs(det - 1.0) > 1e-12 || orthogonality_error > 1e-12) {
    //     LOG_WARN("[MathUtils] Rotation normalization may have failed: det={}, orth_error={}", det, orthogonality_error);
    // }
    
    return R_normalized;
}

bool MathUtils::is_orthogonal(const Eigen::Matrix3f& R, float tolerance) {
    Eigen::Matrix3f should_be_identity = R * R.transpose();
    float error = (should_be_identity - Eigen::Matrix3f::Identity()).norm();
    return error < tolerance && std::abs(R.determinant() - 1.0f) < tolerance;
}

bool MathUtils::is_orthogonal(const Eigen::Matrix3d& R, double tolerance) {
    Eigen::Matrix3d should_be_identity = R * R.transpose();
    double error = (should_be_identity - Eigen::Matrix3d::Identity()).norm();
    return error < tolerance && std::abs(R.determinant() - 1.0) < tolerance;
}

float MathUtils::wrap_to_pi(float angle) {
    while (angle > M_PI) {
        angle -= 2.0f * M_PI;
    }
    while (angle < -M_PI) {
        angle += 2.0f * M_PI;
    }
    return angle;
}

double MathUtils::wrap_to_pi(double angle) {
    while (angle > M_PI) {
        angle -= 2.0 * M_PI;
    }
    while (angle < -M_PI) {
        angle += 2.0 * M_PI;
    }
    return angle;
}

double MathUtils::wrap_angle(double angle) {
    while (angle > M_PI) {
        angle -= 2.0 * M_PI;
    }
    while (angle < -M_PI) {
        angle += 2.0 * M_PI;
    }
    return angle;
}

double MathUtils::deg_to_rad(double degrees) {
    return degrees * M_PI / 180.0;
}

double MathUtils::rad_to_deg(double radians) {
    return radians * 180.0 / M_PI;
}

template<typename T>
bool MathUtils::is_approx_equal(T a, T b, T epsilon) {
    return std::abs(a - b) < epsilon;
}

template<typename T>
T MathUtils::clamp(T value, T min_val, T max_val) {
    return std::max(min_val, std::min(value, max_val));
}

// Explicit template instantiations
template bool MathUtils::is_approx_equal<float>(float, float, float);
template bool MathUtils::is_approx_equal<double>(double, double, double);
template float MathUtils::clamp<float>(float, float, float);
template double MathUtils::clamp<double>(double, double, double);
template int MathUtils::clamp<int>(int, int, int);

} // namespace util
} // namespace lidar_odometry
