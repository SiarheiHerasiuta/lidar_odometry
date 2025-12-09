/**
 * @file      PoseGraphOptimizer.h
 * @brief     Batch Gauss-Newton pose graph optimization.
 * @author    Seungwon Choi
 * @date      2025-12-09
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 * 
 * @par Reference
 * This implementation follows GTSAM's Pose3 conventions:
 * - Lie group SE(3) with [rotation, translation] tangent vector ordering
 * - BetweenFactor error: log(measured^{-1} * T_from^{-1} * T_to)
 * - Adjoint representation for Jacobian computation
 * 
 * @note Implementation details:
 *       - Batch Gauss-Newton optimization
 *       - Sparse Cholesky decomposition (Eigen::SimplicialLDLT)
 *       - SE3 on-manifold optimization with analytical Jacobians
 *       - Internal computation uses [rot, trans] ordering (GTSAM convention)
 *       - Input/output uses our [trans, rot] convention (auto-converted)
 */

#pragma once

#include "../util/MathUtils.h"
#include "../util/PointCloudUtils.h"
#include "../util/MathUtils.h"
#include <Eigen/Sparse>
#include <memory>
#include <vector>
#include <map>
#include <set>
#include <mutex>

namespace lidar_slam {
namespace optimization {

using namespace lidar_slam::util;

// Type aliases for convenience
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
using SpMat = Eigen::SparseMatrix<double>;
using Triplet = Eigen::Triplet<double>;

/**
 * @brief Prior factor: fixes a pose to a measured value
 */
struct PriorFactor {
    int key;                    // Variable index
    SE3d measured;              // Measured pose
    Matrix6d sqrt_info;         // Square root of information matrix
    
    PriorFactor(int k, const SE3d& m, const Matrix6d& info)
        : key(k), measured(m) {
        // Compute sqrt of information matrix
        Eigen::LLT<Matrix6d> llt(info);
        sqrt_info = llt.matrixL().transpose();
    }
};

/**
 * @brief Between factor: relative pose constraint between two poses
 */
struct BetweenFactor {
    int key_from;               // From variable index
    int key_to;                 // To variable index
    SE3d measured;              // Measured relative pose (T_from^{-1} * T_to)
    Matrix6d sqrt_info;         // Square root of information matrix
    
    BetweenFactor(int from, int to, const SE3d& m, const Matrix6d& info)
        : key_from(from), key_to(to), measured(m) {
        Eigen::LLT<Matrix6d> llt(info);
        sqrt_info = llt.matrixL().transpose();
    }
};

/**
 * @brief Pose Graph Optimizer using Batch Gauss-Newton
 * 
 * Provides pose graph optimization for loop closure correction.
 * Uses sparse Cholesky decomposition for efficient solving.
 */
class PoseGraphOptimizer {
public:
    PoseGraphOptimizer();
    ~PoseGraphOptimizer();
    
    // ===== Public API =====
    
    bool add_first_keyframe(int keyframe_id, const SE3f& pose);
    
    bool add_keyframe_with_odom(int prev_keyframe_id, int curr_keyframe_id,
                                const SE3f& curr_pose,
                                const SE3f& relative_pose,
                                double odom_trans_noise = 0.1,
                                double odom_rot_noise = 0.1);
    
    bool add_loop_and_optimize(int from_keyframe_id, int to_keyframe_id,
                               const SE3f& relative_pose,
                               double loop_trans_noise = 0.05,
                               double loop_rot_noise = 0.05);
    
    bool get_optimized_pose(int keyframe_id, SE3f& optimized_pose) const;
    
    std::map<int, SE3f> get_all_optimized_poses() const;
    
    bool has_keyframe(int keyframe_id) const { 
        return m_keyframe_set.find(keyframe_id) != m_keyframe_set.end(); 
    }
    
    size_t get_keyframe_count() const { return m_keyframe_ids.size(); }
    
    size_t get_loop_closure_count() const { return m_loop_closure_count; }
    
    void clear();
    
private:
    // ===== Optimization Methods =====
    
    /**
     * @brief Run batch Gauss-Newton optimization
     * @param max_iterations Maximum number of GN iterations
     * @param convergence_threshold Stop if ||dx|| < threshold
     * @return true if converged
     */
    bool optimize(int max_iterations = 10, double convergence_threshold = 1e-6);
    
    /**
     * @brief Build sparse Hessian matrix and RHS vector
     */
    void buildLinearSystem(SpMat& H, Eigen::VectorXd& b);
    
    /**
     * @brief Compute SE3 BetweenFactor error and Jacobians
     * 
     * Error: log(measured^{-1} * T_from^{-1} * T_to)
     * 
     * @param T_from From pose
     * @param T_to To pose
     * @param measured Measured relative pose
     * @param J_from Output: Jacobian w.r.t. T_from (6x6)
     * @param J_to Output: Jacobian w.r.t. T_to (6x6)
     * @return 6D error vector
     */
    Vector6d computeBetweenError(const SE3d& T_from, const SE3d& T_to,
                                  const SE3d& measured,
                                  Matrix6d& J_from, Matrix6d& J_to) const;
    
    /**
     * @brief Compute SE3 PriorFactor error and Jacobian
     * 
     * Error: log(measured^{-1} * T)
     */
    Vector6d computePriorError(const SE3d& T, const SE3d& measured,
                                Matrix6d& J) const;
    
    /**
     * @brief Compute right Jacobian of SE3
     * Used for Jacobian computation on Lie group
     */
    Matrix6d rightJacobianSE3(const Vector6d& xi) const;
    
    /**
     * @brief Compute inverse of right Jacobian of SE3
     */
    Matrix6d rightJacobianInverseSE3(const Vector6d& xi) const;
    
    /**
     * @brief Compute Adjoint matrix of SE3
     */
    Matrix6d adjointSE3(const SE3d& T) const;
    
    /**
     * @brief Convert SE3f to SE3d
     */
    SE3d toDouble(const SE3f& pose) const;
    
    /**
     * @brief Convert SE3d to SE3f
     */
    SE3f toFloat(const SE3d& pose) const;
    
    /**
     * @brief Create diagonal information matrix from noise sigmas
     */
    Matrix6d makeInformationMatrix(double trans_noise, double rot_noise) const;
    
    // ===== Member Variables =====
    
    mutable std::mutex m_mutex;
    
    // Graph structure
    std::vector<PriorFactor> m_priors;
    std::vector<BetweenFactor> m_betweens;
    
    // Current estimates (double precision for optimization)
    std::map<int, SE3d> m_poses;
    
    // Keyframe tracking
    std::vector<int> m_keyframe_ids;
    std::set<int> m_keyframe_set;
    std::map<int, int> m_keyframe_to_index;  // keyframe_id -> variable index
    
    // Statistics
    size_t m_loop_closure_count = 0;
    size_t m_odometry_count = 0;
    bool m_is_initialized = false;
};

} // namespace optimization
} // namespace lidar_slam
