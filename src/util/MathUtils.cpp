/**
 * @file      MathUtils.cpp
 * @brief     Implementation of math utility functions and Lie group operations
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

namespace lidar_slam {
namespace util {

// ============================================================================
// SO3 Implementation (float)
// ============================================================================

SO3 SO3::Exp(const Eigen::Vector3f& omega) {
    const float theta = omega.norm();
    
    if (theta < math_internal::kEps) {
        // Small angle approximation: R ≈ I + [omega]_×
        return SO3(Eigen::Matrix3f::Identity() + Hat(omega));
    }
    
    const float theta_inv = 1.0f / theta;
    const Eigen::Vector3f k = omega * theta_inv;  // unit axis
    const Eigen::Matrix3f K = Hat(k);
    
    // Rodrigues' formula: R = I + sin(θ)[k]_× + (1-cos(θ))[k]_×²
    return SO3(Eigen::Matrix3f::Identity() + 
               std::sin(theta) * K + 
               (1.0f - std::cos(theta)) * K * K);
}

Eigen::Vector3f SO3::Log() const {
    const float trace = m_matrix.trace();
    const float cos_theta = (trace - 1.0f) * 0.5f;
    
    // Clamp to valid range
    const float cos_theta_clamped = std::max(-1.0f, std::min(1.0f, cos_theta));
    const float theta = std::acos(cos_theta_clamped);
    
    if (theta < math_internal::kEps) {
        // Small angle: ω ≈ vee(R - I)
        return Vee(m_matrix - Eigen::Matrix3f::Identity());
    }
    
    const float sin_theta = std::sin(theta);
    if (std::abs(sin_theta) < math_internal::kEps) {
        // θ ≈ π case: need special handling for sign ambiguity
        Eigen::Vector3f axis;
        
        // Check which diagonal element is largest for numerical stability
        int max_idx = 0;
        if (m_matrix(1, 1) > m_matrix(0, 0)) max_idx = 1;
        if (m_matrix(2, 2) > m_matrix(max_idx, max_idx)) max_idx = 2;
        
        axis[max_idx] = std::sqrt((m_matrix(max_idx, max_idx) + 1.0f) * 0.5f);
        
        for (int i = 0; i < 3; ++i) {
            if (i != max_idx) {
                axis[i] = m_matrix(max_idx, i) / (2.0f * axis[max_idx]);
            }
        }
        
        // Fix sign ambiguity
        Eigen::Vector3f skew_axis = Vee(m_matrix - m_matrix.transpose()) * 0.5f;
        float dot = axis.dot(skew_axis);
        if (dot < 0) {
            axis = -axis;
        }
        
        return axis * theta;
    }
    
    const float factor = theta / (2.0f * sin_theta);
    return factor * Vee(m_matrix - m_matrix.transpose());
}

SO3::SO3(const Eigen::Matrix3f& R) {
    // Use SVD to find the closest rotation matrix
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3f U = svd.matrixU();
    Eigen::Matrix3f V = svd.matrixV();
    
    m_matrix = U * V.transpose();
    
    // Ensure det(R) = +1 (proper rotation, not reflection)
    if (m_matrix.determinant() < 0.0f) {
        U.col(2) *= -1.0f;
        m_matrix = U * V.transpose();
    }
}

void SO3::Normalize() {
    *this = SO3(m_matrix);
}

// ============================================================================
// SE3 Implementation (float)
// ============================================================================

SE3::SE3(const Eigen::Matrix4f& matrix) {
    m_rotation = SO3(matrix.block<3, 3>(0, 0));
    m_translation = matrix.block<3, 1>(0, 3);
}

SE3 SE3::FromMatrix(const Eigen::Matrix4f& T) {
    return SE3(T);
}

SE3 SE3::Exp(const Eigen::Matrix<float, 6, 1>& xi) {
    const Eigen::Vector3f rho = xi.head<3>();     // translation part
    const Eigen::Vector3f phi = xi.tail<3>();     // rotation part
    
    const SO3 R = SO3::Exp(phi);
    
    const float theta = phi.norm();
    Eigen::Vector3f t;
    
    if (theta < math_internal::kEps) {
        // Small angle: V ≈ I
        t = rho;
    } else {
        // V = I + (1-cos(θ))/θ²[φ]_× + (θ-sin(θ))/θ³[φ]_×²
        const Eigen::Matrix3f phi_hat = Hat(phi);
        const float theta2 = theta * theta;
        const float sin_theta = std::sin(theta);
        const float cos_theta = std::cos(theta);
        
        const Eigen::Matrix3f V = Eigen::Matrix3f::Identity() +
            (1.0f - cos_theta) / theta2 * phi_hat +
            (theta - sin_theta) / (theta2 * theta) * phi_hat * phi_hat;
        
        t = V * rho;
    }
    
    return SE3(R, t);
}

Eigen::Matrix<float, 6, 1> SE3::Log() const {
    Eigen::Matrix<float, 6, 1> xi;
    
    const Eigen::Vector3f phi = m_rotation.Log();
    const float theta = phi.norm();
    
    if (theta < math_internal::kEps) {
        // Small angle: V^-1 ≈ I
        xi.head<3>() = m_translation;
    } else {
        const Eigen::Matrix3f phi_hat = Hat(phi);
        const float theta2 = theta * theta;
        const float sin_theta = std::sin(theta);
        const float cos_theta = std::cos(theta);
        
        const float coeff = (2.0f * sin_theta - theta * (1.0f + cos_theta)) / 
                           (2.0f * theta2 * sin_theta);
        
        const Eigen::Matrix3f V_inv = Eigen::Matrix3f::Identity() -
            0.5f * phi_hat +
            coeff * phi_hat * phi_hat;
        
        xi.head<3>() = V_inv * m_translation;
    }
    
    xi.tail<3>() = phi;
    return xi;
}

Eigen::Matrix4f SE3::Matrix() const {
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    T.block<3, 3>(0, 0) = m_rotation.Matrix();
    T.block<3, 1>(0, 3) = m_translation;
    return T;
}

// ============================================================================
// SO3d Implementation (double)
// ============================================================================

SO3d SO3d::Exp(const Eigen::Vector3d& omega) {
    const double theta = omega.norm();
    
    if (theta < math_internal::kEpsD) {
        return SO3d(Eigen::Matrix3d::Identity() + Hatd(omega));
    }
    
    const double theta_inv = 1.0 / theta;
    const Eigen::Vector3d k = omega * theta_inv;
    const Eigen::Matrix3d K = Hatd(k);
    
    return SO3d(Eigen::Matrix3d::Identity() + 
               std::sin(theta) * K + 
               (1.0 - std::cos(theta)) * K * K);
}

Eigen::Vector3d SO3d::Log() const {
    const double trace = m_matrix.trace();
    const double cos_theta = (trace - 1.0) * 0.5;
    const double cos_theta_clamped = std::max(-1.0, std::min(1.0, cos_theta));
    const double theta = std::acos(cos_theta_clamped);
    
    if (theta < math_internal::kEpsD) {
        return Veed(m_matrix - Eigen::Matrix3d::Identity());
    }
    
    const double sin_theta = std::sin(theta);
    if (std::abs(sin_theta) < math_internal::kEpsD) {
        Eigen::Vector3d axis;
        int max_idx = 0;
        if (m_matrix(1, 1) > m_matrix(0, 0)) max_idx = 1;
        if (m_matrix(2, 2) > m_matrix(max_idx, max_idx)) max_idx = 2;
        
        axis[max_idx] = std::sqrt((m_matrix(max_idx, max_idx) + 1.0) * 0.5);
        
        for (int i = 0; i < 3; ++i) {
            if (i != max_idx) {
                axis[i] = m_matrix(max_idx, i) / (2.0 * axis[max_idx]);
            }
        }
        
        Eigen::Vector3d skew_axis = Veed(m_matrix - m_matrix.transpose()) * 0.5;
        if (axis.dot(skew_axis) < 0) {
            axis = -axis;
        }
        
        return axis * theta;
    }
    
    const double factor = theta / (2.0 * sin_theta);
    return factor * Veed(m_matrix - m_matrix.transpose());
}

SO3d::SO3d(const Eigen::Matrix3d& R) {
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    
    m_matrix = U * V.transpose();
    
    if (m_matrix.determinant() < 0.0) {
        U.col(2) *= -1.0;
        m_matrix = U * V.transpose();
    }
}

void SO3d::Normalize() {
    *this = SO3d(m_matrix);
}

// ============================================================================
// SE3d Implementation (double)
// ============================================================================

SE3d::SE3d(const Eigen::Matrix4d& matrix) {
    m_rotation = SO3d(matrix.block<3, 3>(0, 0));
    m_translation = matrix.block<3, 1>(0, 3);
}

SE3d SE3d::FromMatrix(const Eigen::Matrix4d& T) {
    return SE3d(T);
}

SE3d SE3d::Exp(const Eigen::Matrix<double, 6, 1>& xi) {
    const Eigen::Vector3d rho = xi.head<3>();
    const Eigen::Vector3d phi = xi.tail<3>();
    
    const SO3d R = SO3d::Exp(phi);
    const double theta = phi.norm();
    Eigen::Vector3d t;
    
    if (theta < math_internal::kEpsD) {
        t = rho;
    } else {
        const Eigen::Matrix3d phi_hat = Hatd(phi);
        const double theta2 = theta * theta;
        const double sin_theta = std::sin(theta);
        const double cos_theta = std::cos(theta);
        
        const Eigen::Matrix3d V = Eigen::Matrix3d::Identity() +
            (1.0 - cos_theta) / theta2 * phi_hat +
            (theta - sin_theta) / (theta2 * theta) * phi_hat * phi_hat;
        
        t = V * rho;
    }
    
    return SE3d(R, t);
}

Eigen::Matrix<double, 6, 1> SE3d::Log() const {
    Eigen::Matrix<double, 6, 1> xi;
    
    const Eigen::Vector3d phi = m_rotation.Log();
    const double theta = phi.norm();
    
    if (theta < math_internal::kEpsD) {
        xi.head<3>() = m_translation;
    } else {
        const Eigen::Matrix3d phi_hat = Hatd(phi);
        const double theta2 = theta * theta;
        const double sin_theta = std::sin(theta);
        const double cos_theta = std::cos(theta);
        
        const double coeff = (2.0 * sin_theta - theta * (1.0 + cos_theta)) / 
                           (2.0 * theta2 * sin_theta);
        
        const Eigen::Matrix3d V_inv = Eigen::Matrix3d::Identity() -
            0.5 * phi_hat +
            coeff * phi_hat * phi_hat;
        
        xi.head<3>() = V_inv * m_translation;
    }
    
    xi.tail<3>() = phi;
    return xi;
}

Eigen::Matrix4d SE3d::Matrix() const {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = m_rotation.Matrix();
    T.block<3, 1>(0, 3) = m_translation;
    return T;
}

// ============================================================================
// Utility Functions (Hat/Vee)
// ============================================================================

Eigen::Matrix3f Hat(const Eigen::Vector3f& v) {
    Eigen::Matrix3f S;
    S <<     0.0f, -v(2),  v(1),
          v(2),     0.0f, -v(0),
         -v(1),  v(0),     0.0f;
    return S;
}

Eigen::Vector3f Vee(const Eigen::Matrix3f& S) {
    return Eigen::Vector3f(S(2, 1), S(0, 2), S(1, 0));
}

Eigen::Matrix3d Hatd(const Eigen::Vector3d& v) {
    Eigen::Matrix3d S;
    S <<    0.0, -v(2),  v(1),
         v(2),    0.0, -v(0),
        -v(1),  v(0),    0.0;
    return S;
}

Eigen::Vector3d Veed(const Eigen::Matrix3d& S) {
    return Eigen::Vector3d(S(2, 1), S(0, 2), S(1, 0));
}

// ============================================================================
// MathUtils Implementation
// ============================================================================

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
} // namespace lidar_slam
