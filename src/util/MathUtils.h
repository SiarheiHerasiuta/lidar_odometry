/**
 * @file      MathUtils.h
 * @brief     Math utility functions and Lie group operations for LiDAR odometry
 * @author    Seungwon Choi
 * @date      2025-09-24
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 * 
 * @note This file includes Lie group utilities (SO3/SE3) that were previously
 *       in a separate LieUtils.h file.
 */

#pragma once

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <cmath>

namespace lidar_slam {
namespace util {

// ===== Eigen Type Aliases =====
using Vector3f = Eigen::Vector3f;
using Vector4f = Eigen::Vector4f;
using Matrix3f = Eigen::Matrix3f;
using Matrix4f = Eigen::Matrix4f;
using VectorXf = Eigen::VectorXf;
using MatrixXf = Eigen::MatrixXf;
using Vector3d = Eigen::Vector3d;
using Vector4d = Eigen::Vector4d;
using Matrix3d = Eigen::Matrix3d;
using Matrix4d = Eigen::Matrix4d;
using VectorXd = Eigen::VectorXd;
using MatrixXd = Eigen::MatrixXd;

// ===== Internal Constants =====
namespace math_internal {
constexpr float kEps = 1e-6f;
constexpr double kEpsD = 1e-10;
} // namespace math_internal

// ============================================================================
// Lie Group Classes
// ============================================================================

/**
 * @brief SO(3) Lie group utilities
 * 
 * Implements essential operations for 3D rotations:
 * - exp: axis-angle → rotation matrix
 * - log: rotation matrix → axis-angle  
 * - hat: vector → skew-symmetric matrix
 * - vee: skew-symmetric matrix → vector
 */
class SO3 {
public:
    SO3() = default;
    
    /// Create from rotation matrix (automatically normalizes to SO(3))
    explicit SO3(const Eigen::Matrix3f& R);
    
    /// Normalize current rotation matrix to ensure it's in SO(3)
    void Normalize();
    
    /// Create from axis-angle vector
    static SO3 Exp(const Eigen::Vector3f& omega);
    
    /// Convert to axis-angle vector
    Eigen::Vector3f Log() const;
    
    /// Get rotation matrix
    const Eigen::Matrix3f& Matrix() const { return m_matrix; }
    Eigen::Matrix3f& Matrix() { return m_matrix; }
    
    /// Composition
    SO3 operator*(const SO3& other) const {
        return SO3(m_matrix * other.m_matrix);
    }
    
    /// Apply rotation to vector
    Eigen::Vector3f operator*(const Eigen::Vector3f& v) const {
        return m_matrix * v;
    }
    
    /// Inverse
    SO3 Inverse() const {
        return SO3(m_matrix.transpose());
    }
    
    /// Identity
    static SO3 Identity() {
        return SO3(Eigen::Matrix3f::Identity());
    }
    
private:
    Eigen::Matrix3f m_matrix = Eigen::Matrix3f::Identity();
};

/**
 * @brief SE(3) Lie group utilities
 * 
 * Implements essential operations for 3D poses:
 * - exp: 6D twist → transformation matrix
 * - log: transformation matrix → 6D twist
 * - composition and inverse operations
 * 
 * Convention: 6D twist is [translation, rotation] (rho, phi)
 */
class SE3 {
public:
    SE3() = default;
    SE3(const SO3& rotation, const Eigen::Vector3f& translation) 
        : m_rotation(rotation), m_translation(translation) {}
    SE3(const Eigen::Matrix3f& R, const Eigen::Vector3f& t)
        : m_rotation(R), m_translation(t) {}
    explicit SE3(const Eigen::Matrix4f& matrix);
    
    /// Create from transformation matrix
    static SE3 FromMatrix(const Eigen::Matrix4f& T);
    
    /// Create from 6D twist vector [rho, phi] where rho=translation, phi=rotation
    static SE3 Exp(const Eigen::Matrix<float, 6, 1>& xi);
    
    /// Convert to 6D twist vector
    Eigen::Matrix<float, 6, 1> Log() const;
    
    /// Get transformation matrix
    Eigen::Matrix4f Matrix() const;
    
    /// Get rotation part
    const SO3& Rotation() const { return m_rotation; }
    SO3& Rotation() { return m_rotation; }
    
    /// Get rotation matrix
    Eigen::Matrix3f RotationMatrix() const { return m_rotation.Matrix(); }
    
    /// Get translation part
    const Eigen::Vector3f& Translation() const { return m_translation; }
    Eigen::Vector3f& Translation() { return m_translation; }
    
    /// Composition
    SE3 operator*(const SE3& other) const {
        return SE3(m_rotation * other.m_rotation,
                   m_translation + m_rotation * other.m_translation);
    }
    
    /// Apply transformation to point
    Eigen::Vector3f operator*(const Eigen::Vector3f& p) const {
        return m_rotation * p + m_translation;
    }
    
    /// Inverse
    SE3 Inverse() const {
        SO3 R_inv = m_rotation.Inverse();
        return SE3(R_inv, R_inv * (-m_translation));
    }
    
    /// Identity
    static SE3 Identity() {
        return SE3(SO3::Identity(), Eigen::Vector3f::Zero());
    }
    
private:
    SO3 m_rotation;
    Eigen::Vector3f m_translation = Eigen::Vector3f::Zero();
};

// ===== Double Precision Versions =====

/**
 * @brief SO(3) Lie group utilities (double precision)
 */
class SO3d {
public:
    SO3d() = default;
    
    explicit SO3d(const Eigen::Matrix3d& R);
    
    void Normalize();
    
    static SO3d Exp(const Eigen::Vector3d& omega);
    
    Eigen::Vector3d Log() const;
    
    const Eigen::Matrix3d& Matrix() const { return m_matrix; }
    Eigen::Matrix3d& Matrix() { return m_matrix; }
    
    SO3d operator*(const SO3d& other) const {
        return SO3d(m_matrix * other.m_matrix);
    }
    
    Eigen::Vector3d operator*(const Eigen::Vector3d& v) const {
        return m_matrix * v;
    }
    
    SO3d Inverse() const {
        return SO3d(m_matrix.transpose());
    }
    
    static SO3d Identity() {
        return SO3d(Eigen::Matrix3d::Identity());
    }
    
private:
    Eigen::Matrix3d m_matrix = Eigen::Matrix3d::Identity();
};

/**
 * @brief SE(3) Lie group utilities (double precision)
 */
class SE3d {
public:
    SE3d() = default;
    SE3d(const SO3d& rotation, const Eigen::Vector3d& translation) 
        : m_rotation(rotation), m_translation(translation) {}
    SE3d(const Eigen::Matrix3d& R, const Eigen::Vector3d& t)
        : m_rotation(R), m_translation(t) {}
    explicit SE3d(const Eigen::Matrix4d& matrix);
    
    static SE3d FromMatrix(const Eigen::Matrix4d& T);
    
    static SE3d Exp(const Eigen::Matrix<double, 6, 1>& xi);
    
    Eigen::Matrix<double, 6, 1> Log() const;
    
    Eigen::Matrix4d Matrix() const;
    
    const SO3d& Rotation() const { return m_rotation; }
    SO3d& Rotation() { return m_rotation; }
    
    Eigen::Matrix3d RotationMatrix() const { return m_rotation.Matrix(); }
    
    const Eigen::Vector3d& Translation() const { return m_translation; }
    Eigen::Vector3d& Translation() { return m_translation; }
    
    SE3d operator*(const SE3d& other) const {
        return SE3d(m_rotation * other.m_rotation,
                   m_translation + m_rotation * other.m_translation);
    }
    
    Eigen::Vector3d operator*(const Eigen::Vector3d& p) const {
        return m_rotation * p + m_translation;
    }
    
    SE3d Inverse() const {
        SO3d R_inv = m_rotation.Inverse();
        return SE3d(R_inv, R_inv * (-m_translation));
    }
    
    static SE3d Identity() {
        return SE3d(SO3d::Identity(), Eigen::Vector3d::Zero());
    }
    
private:
    SO3d m_rotation;
    Eigen::Vector3d m_translation = Eigen::Vector3d::Zero();
};

// ===== Lie Algebra Utility Functions =====

/// Convert vector to skew-symmetric matrix (float)
Eigen::Matrix3f Hat(const Eigen::Vector3f& v);

/// Convert skew-symmetric matrix to vector (float)
Eigen::Vector3f Vee(const Eigen::Matrix3f& S);

/// Convert vector to skew-symmetric matrix (double)
Eigen::Matrix3d Hatd(const Eigen::Vector3d& v);

/// Convert skew-symmetric matrix to vector (double)
Eigen::Vector3d Veed(const Eigen::Matrix3d& S);

// ============================================================================
// Math Utility Class
// ============================================================================

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

// ============================================================================
// Type Aliases (for convenience)
// ============================================================================

// SE3/SO3 float aliases
using SE3f = SE3;
using SO3f = SO3;

// Common constants
constexpr float kEpsilonF = 1e-6f;      ///< Small epsilon for float comparisons
constexpr double kEpsilonD = 1e-9;      ///< Small epsilon for double comparisons
constexpr float kInfF = 1e9f;           ///< Large float value
constexpr double kInfD = 1e12;          ///< Large double value

} // namespace util
} // namespace lidar_slam
