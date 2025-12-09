/**
 * @file      PoseGraphOptimizer.cpp
 * @brief     Implementation of Batch Gauss-Newton pose graph optimization.
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
 * @note Internal computation uses GTSAM convention [rot, trans] ordering.
 *       Input/output uses our convention [trans, rot] and is converted internally.
 */

#include "PoseGraphOptimizer.h"
#include "../util/LogUtils.h"
#include <chrono>

namespace lidar_slam {
namespace optimization {

using lidar_slam::util::Hatd;

namespace {
    constexpr double kEpsLie = 1e-10;
    
    // ===== GTSAM-style helpers (using [rot, trans] ordering) =====
    
    // Skew symmetric matrix
    Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
        Eigen::Matrix3d S;
        S <<     0, -v(2),  v(1),
              v(2),     0, -v(0),
             -v(1),  v(0),     0;
        return S;
    }
    
    // SO3 Logmap: rotation matrix -> axis-angle
    Eigen::Vector3d SO3_Logmap(const Eigen::Matrix3d& R) {
        double tr = R.trace();
        double theta = std::acos(std::clamp((tr - 1.0) / 2.0, -1.0, 1.0));
        
        if (theta < kEpsLie) {
            return Eigen::Vector3d(R(2,1) - R(1,2), R(0,2) - R(2,0), R(1,0) - R(0,1)) / 2.0;
        }
        
        double k = theta / (2.0 * std::sin(theta));
        return Eigen::Vector3d(R(2,1) - R(1,2), R(0,2) - R(2,0), R(1,0) - R(0,1)) * k;
    }
    
    // SO3 Expmap: axis-angle -> rotation matrix
    Eigen::Matrix3d SO3_Expmap(const Eigen::Vector3d& w) {
        double theta = w.norm();
        if (theta < kEpsLie) {
            return Eigen::Matrix3d::Identity() + skew(w);
        }
        Eigen::Matrix3d W = skew(w / theta);
        return Eigen::Matrix3d::Identity() + std::sin(theta) * W + (1.0 - std::cos(theta)) * W * W;
    }
    
    // SO3 Right Jacobian inverse
    Eigen::Matrix3d SO3_JrInv(const Eigen::Vector3d& w) {
        double theta = w.norm();
        if (theta < kEpsLie) {
            return Eigen::Matrix3d::Identity() + 0.5 * skew(w);
        }
        Eigen::Matrix3d W = skew(w);
        double theta2 = theta * theta;
        double cot_half = 1.0 / std::tan(theta / 2.0);
        return Eigen::Matrix3d::Identity() + 0.5 * W + 
               (1.0 / theta2 - cot_half / (2.0 * theta)) * W * W;
    }
    
    // SE3 Logmap: (R, t) -> [w, u] where w=rot, u=trans (GTSAM order!)
    Vector6d SE3_Logmap(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
        Eigen::Vector3d w = SO3_Logmap(R);
        double theta = w.norm();
        
        Vector6d xi;
        if (theta < kEpsLie) {
            xi << w, t;
        } else {
            Eigen::Matrix3d W = skew(w / theta);
            double tan_half = std::tan(0.5 * theta);
            Eigen::Vector3d Wt = W * t;
            Eigen::Vector3d u = t - (0.5 * theta) * Wt + (1.0 - theta / (2.0 * tan_half)) * (W * Wt);
            xi << w, u;
        }
        return xi;
    }
    
    // SE3 Expmap: [w, u] -> (R, t) where w=rot, u=trans (GTSAM order!)
    void SE3_Expmap(const Vector6d& xi, Eigen::Matrix3d& R, Eigen::Vector3d& t) {
        Eigen::Vector3d w = xi.head<3>();
        Eigen::Vector3d u = xi.tail<3>();
        
        R = SO3_Expmap(w);
        double theta = w.norm();
        
        if (theta < kEpsLie) {
            t = u;
        } else {
            Eigen::Matrix3d W = skew(w);
            double theta2 = theta * theta;
            double sin_t = std::sin(theta);
            double cos_t = std::cos(theta);
            Eigen::Matrix3d V = Eigen::Matrix3d::Identity() + 
                (1.0 - cos_t) / theta2 * W + 
                (theta - sin_t) / (theta2 * theta) * W * W;
            t = V * u;
        }
    }
    
    // SE3 Adjoint: [rot, trans] ordering
    // Ad_T = | R    0   |
    //        | tR   R   |
    Matrix6d SE3_AdjointMap(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
        Matrix6d Ad = Matrix6d::Zero();
        Eigen::Matrix3d tR = skew(t) * R;
        Ad.block<3,3>(0,0) = R;
        Ad.block<3,3>(3,0) = tR;
        Ad.block<3,3>(3,3) = R;
        return Ad;
    }
    
    // SE3 LogmapDerivative (Jacobian of Logmap w.r.t. pose)
    Matrix6d SE3_LogmapDerivative(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
        Vector6d xi = SE3_Logmap(R, t);
        Eigen::Vector3d w = xi.head<3>();
        Eigen::Matrix3d Jw = SO3_JrInv(w);
        
        // Q matrix computation (simplified)
        double theta = w.norm();
        Eigen::Matrix3d Q = Eigen::Matrix3d::Zero();
        if (theta > kEpsLie) {
            Eigen::Vector3d u = xi.tail<3>();
            Eigen::Matrix3d W = skew(w);
            Eigen::Matrix3d U = skew(u);
            double theta2 = theta * theta;
            double sin_t = std::sin(theta);
            double cos_t = std::cos(theta);
            
            double c1 = (1.0 - cos_t) / theta2;
            double c2 = (theta - sin_t) / (theta2 * theta);
            double c3 = (1.0 - theta2/2.0 - cos_t) / (theta2 * theta2);
            
            Q = 0.5 * U + c1 * (W*U + U*W) + c2 * (W*W*U + U*W*W) 
                - c3 * w.dot(u) * W;
        }
        
        Matrix6d J = Matrix6d::Zero();
        J.block<3,3>(0,0) = Jw;
        J.block<3,3>(3,0) = -Jw * Q * Jw;
        J.block<3,3>(3,3) = Jw;
        return J;
    }
}

PoseGraphOptimizer::PoseGraphOptimizer() {
    LOG_DEBUG("[PGO-Manual] Initialized with Batch Gauss-Newton solver");
}

PoseGraphOptimizer::~PoseGraphOptimizer() = default;

// ===== Public API =====

bool PoseGraphOptimizer::add_first_keyframe(int keyframe_id, const SE3f& pose) {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (!m_keyframe_ids.empty()) {
        LOG_WARN("[PGO-Manual] add_first_keyframe called but graph is not empty");
        return false;
    }
    
    SE3d pose_d = toDouble(pose);
    
    // Add prior factor with tight noise
    Matrix6d info = makeInformationMatrix(1e-4, 1e-4);
    m_priors.emplace_back(0, pose_d, info);
    
    // Add initial estimate
    m_poses[keyframe_id] = pose_d;
    m_keyframe_ids.push_back(keyframe_id);
    m_keyframe_set.insert(keyframe_id);
    m_keyframe_to_index[keyframe_id] = 0;
    m_is_initialized = true;
    
    LOG_INFO("[PGO-Manual] Added first keyframe {} with prior", keyframe_id);
    return true;
}

bool PoseGraphOptimizer::add_keyframe_with_odom(int prev_keyframe_id, int curr_keyframe_id,
                                                       const SE3f& curr_pose,
                                                       const SE3f& relative_pose,
                                                       double odom_trans_noise,
                                                       double odom_rot_noise) {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (m_keyframe_set.find(curr_keyframe_id) != m_keyframe_set.end()) {
        LOG_DEBUG("[PGO-Manual] Keyframe {} already exists in graph", curr_keyframe_id);
        return true;
    }
    
    SE3d curr_pose_d = toDouble(curr_pose);
    SE3d relative_pose_d = toDouble(relative_pose);
    
    // Assign new index
    int curr_index = static_cast<int>(m_keyframe_ids.size());
    
    // Check if prev keyframe exists
    bool prev_exists = (m_keyframe_set.find(prev_keyframe_id) != m_keyframe_set.end());
    
    if (prev_exists) {
        int prev_index = m_keyframe_to_index[prev_keyframe_id];
        
        // Add odometry (between) factor
        Matrix6d info = makeInformationMatrix(odom_trans_noise, odom_rot_noise);
        m_betweens.emplace_back(prev_index, curr_index, relative_pose_d, info);
    } else {
        LOG_WARN("[PGO-Manual] Previous keyframe {} not found, adding {} with loose prior", 
                    prev_keyframe_id, curr_keyframe_id);
        
        // Add loose prior instead
        Matrix6d info = makeInformationMatrix(0.5, 0.1);
        m_priors.emplace_back(curr_index, curr_pose_d, info);
    }
    
    // Add initial estimate
    m_poses[curr_keyframe_id] = curr_pose_d;
    m_keyframe_ids.push_back(curr_keyframe_id);
    m_keyframe_set.insert(curr_keyframe_id);
    m_keyframe_to_index[curr_keyframe_id] = curr_index;
    m_odometry_count++;
    
    LOG_DEBUG("[PGO-Manual] Added keyframe {} with odom from {}", curr_keyframe_id, prev_keyframe_id);
    return true;
}

bool PoseGraphOptimizer::add_loop_and_optimize(int from_keyframe_id, int to_keyframe_id,
                                                      const SE3f& relative_pose,
                                                      double loop_trans_noise,
                                                      double loop_rot_noise) {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (m_keyframe_set.find(from_keyframe_id) == m_keyframe_set.end()) {
        LOG_ERROR("[PGO-Manual] Loop from-keyframe {} not found", from_keyframe_id);
        return false;
    }
    if (m_keyframe_set.find(to_keyframe_id) == m_keyframe_set.end()) {
        LOG_ERROR("[PGO-Manual] Loop to-keyframe {} not found", to_keyframe_id);
        return false;
    }
    
    int from_index = m_keyframe_to_index[from_keyframe_id];
    int to_index = m_keyframe_to_index[to_keyframe_id];
    
    SE3d relative_pose_d = toDouble(relative_pose);
    
    // Add loop closure factor
    Matrix6d info = makeInformationMatrix(loop_trans_noise, loop_rot_noise);
    m_betweens.emplace_back(from_index, to_index, relative_pose_d, info);
    
    // Run optimization
    auto start = std::chrono::high_resolution_clock::now();
    
    bool converged = optimize(10, 1e-6);
    
    auto end = std::chrono::high_resolution_clock::now();
    double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    m_loop_closure_count++;
    
    LOG_INFO("[PGO-Manual] Loop closure {}->{}, optimized in {:.2f}ms, converged: {}", 
             from_keyframe_id, to_keyframe_id, duration_ms, converged);
    
    return true;
}

bool PoseGraphOptimizer::get_optimized_pose(int keyframe_id, SE3f& optimized_pose) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    auto it = m_poses.find(keyframe_id);
    if (it == m_poses.end()) {
        return false;
    }
    
    optimized_pose = toFloat(it->second);
    return true;
}

std::map<int, SE3f> PoseGraphOptimizer::get_all_optimized_poses() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    std::map<int, SE3f> result;
    for (const auto& [key, pose] : m_poses) {
        result[key] = toFloat(pose);
    }
    return result;
}

void PoseGraphOptimizer::clear() {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    m_priors.clear();
    m_betweens.clear();
    m_poses.clear();
    m_keyframe_ids.clear();
    m_keyframe_set.clear();
    m_keyframe_to_index.clear();
    
    m_loop_closure_count = 0;
    m_odometry_count = 0;
    m_is_initialized = false;
    
    LOG_DEBUG("[PGO-Manual] Cleared pose graph");
}

// ===== Private Methods =====

bool PoseGraphOptimizer::optimize(int max_iterations, double convergence_threshold) {
    const int n_vars = static_cast<int>(m_keyframe_ids.size());
    const int n_dims = n_vars * 6;
    
    if (n_vars == 0) return true;
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Build linear system
        SpMat H(n_dims, n_dims);
        Eigen::VectorXd b = Eigen::VectorXd::Zero(n_dims);
        
        buildLinearSystem(H, b);
        
        // Solve using Sparse Cholesky (SimplicialLDLT)
        Eigen::SimplicialLDLT<SpMat> solver;
        solver.compute(H);
        
        if (solver.info() != Eigen::Success) {
            LOG_ERROR("[PGO-Manual] Cholesky decomposition failed");
            return false;
        }
        
        Eigen::VectorXd dx = solver.solve(b);
        
        if (solver.info() != Eigen::Success) {
            LOG_ERROR("[PGO-Manual] Solve failed");
            return false;
        }
        
        // Update poses (delta is in GTSAM [rot, trans] order)
        for (int i = 0; i < n_vars; ++i) {
            int keyframe_id = m_keyframe_ids[i];
            Vector6d delta = dx.segment<6>(i * 6);  // [rot, trans] order
            
            // Get current pose
            SE3d& T = m_poses[keyframe_id];
            Eigen::Matrix3d R_old = T.RotationMatrix();
            Eigen::Vector3d t_old = T.Translation();
            
            // Apply delta: T_new = T_old * Exp(delta)
            Eigen::Matrix3d dR;
            Eigen::Vector3d dt;
            SE3_Expmap(delta, dR, dt);
            
            Eigen::Matrix3d R_new = R_old * dR;
            Eigen::Vector3d t_new = R_old * dt + t_old;
            
            // Update pose
            Eigen::Matrix4d T_new = Eigen::Matrix4d::Identity();
            T_new.block<3,3>(0,0) = R_new;
            T_new.block<3,1>(0,3) = t_new;
            m_poses[keyframe_id] = SE3d::FromMatrix(T_new);
        }
        
        // Check convergence
        double dx_norm = dx.norm();
        LOG_DEBUG("[PGO-Manual] Iter {}: ||dx|| = {:.6f}", iter, dx_norm);
        
        if (dx_norm < convergence_threshold) {
            return true;
        }
    }
    
    return false;
}

void PoseGraphOptimizer::buildLinearSystem(SpMat& H, Eigen::VectorXd& b) {
    const int n_vars = static_cast<int>(m_keyframe_ids.size());
    
    std::vector<Triplet> triplets;
    triplets.reserve(m_priors.size() * 36 + m_betweens.size() * 144);
    
    b.setZero();
    
    // Process prior factors
    for (const auto& prior : m_priors) {
        int idx = prior.key;
        int keyframe_id = m_keyframe_ids[idx];
        
        Matrix6d J;
        Vector6d error = computePriorError(m_poses[keyframe_id], prior.measured, J);
        
        // Whitened Jacobian and error
        Matrix6d Jw = prior.sqrt_info * J;
        Vector6d ew = prior.sqrt_info * error;
        
        // H += J^T * J
        Matrix6d JtJ = Jw.transpose() * Jw;
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) {
                triplets.emplace_back(idx * 6 + i, idx * 6 + j, JtJ(i, j));
            }
        }
        
        // b += J^T * e (negative gradient for GN)
        b.segment<6>(idx * 6) -= Jw.transpose() * ew;
    }
    
    // Process between factors
    for (const auto& between : m_betweens) {
        int idx_from = between.key_from;
        int idx_to = between.key_to;
        int kf_from = m_keyframe_ids[idx_from];
        int kf_to = m_keyframe_ids[idx_to];
        
        Matrix6d J_from, J_to;
        Vector6d error = computeBetweenError(m_poses[kf_from], m_poses[kf_to], 
                                              between.measured, J_from, J_to);
        
        // Whitened
        Matrix6d Jw_from = between.sqrt_info * J_from;
        Matrix6d Jw_to = between.sqrt_info * J_to;
        Vector6d ew = between.sqrt_info * error;
        
        // H blocks
        Matrix6d H_ff = Jw_from.transpose() * Jw_from;
        Matrix6d H_tt = Jw_to.transpose() * Jw_to;
        Matrix6d H_ft = Jw_from.transpose() * Jw_to;
        Matrix6d H_tf = Jw_to.transpose() * Jw_from;
        
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) {
                triplets.emplace_back(idx_from * 6 + i, idx_from * 6 + j, H_ff(i, j));
                triplets.emplace_back(idx_to * 6 + i, idx_to * 6 + j, H_tt(i, j));
                triplets.emplace_back(idx_from * 6 + i, idx_to * 6 + j, H_ft(i, j));
                triplets.emplace_back(idx_to * 6 + i, idx_from * 6 + j, H_tf(i, j));
            }
        }
        
        // b (negative gradient)
        b.segment<6>(idx_from * 6) -= Jw_from.transpose() * ew;
        b.segment<6>(idx_to * 6) -= Jw_to.transpose() * ew;
    }
    
    H.setFromTriplets(triplets.begin(), triplets.end());
}

Vector6d PoseGraphOptimizer::computeBetweenError(const SE3d& T_from, const SE3d& T_to,
                                                        const SE3d& measured,
                                                        Matrix6d& J_from, Matrix6d& J_to) const {
    // Get R, t from poses
    Eigen::Matrix3d R_from = T_from.RotationMatrix();
    Eigen::Vector3d t_from = T_from.Translation();
    Eigen::Matrix3d R_to = T_to.RotationMatrix();
    Eigen::Vector3d t_to = T_to.Translation();
    Eigen::Matrix3d R_meas = measured.RotationMatrix();
    Eigen::Vector3d t_meas = measured.Translation();
    
    // hx = T_from^{-1} * T_to  (actual relative pose)
    Eigen::Matrix3d R_from_inv = R_from.transpose();
    Eigen::Matrix3d R_hx = R_from_inv * R_to;
    Eigen::Vector3d t_hx = R_from_inv * (t_to - t_from);
    
    // error_pose = measured^{-1} * hx
    Eigen::Matrix3d R_meas_inv = R_meas.transpose();
    Eigen::Matrix3d R_err = R_meas_inv * R_hx;
    Eigen::Vector3d t_err = R_meas_inv * (t_hx - t_meas);
    
    // error = Logmap(error_pose) in GTSAM order [rot, trans]
    Vector6d error = SE3_Logmap(R_err, t_err);
    
    // Jacobians (GTSAM style):
    // J_to = I (identity)
    // J_from = -Ad(hx^{-1}) where hx = T_from^{-1} * T_to
    Eigen::Matrix3d R_hx_inv = R_hx.transpose();
    Eigen::Vector3d t_hx_inv = -R_hx_inv * t_hx;
    Matrix6d Ad_hx_inv = SE3_AdjointMap(R_hx_inv, t_hx_inv);
    
    J_to = Matrix6d::Identity();
    J_from = -Ad_hx_inv;
    
    return error;
}

Vector6d PoseGraphOptimizer::computePriorError(const SE3d& T, const SE3d& measured,
                                                      Matrix6d& J) const {
    // Get R, t
    Eigen::Matrix3d R = T.RotationMatrix();
    Eigen::Vector3d t = T.Translation();
    Eigen::Matrix3d R_meas = measured.RotationMatrix();
    Eigen::Vector3d t_meas = measured.Translation();
    
    // error_pose = measured^{-1} * T
    Eigen::Matrix3d R_meas_inv = R_meas.transpose();
    Eigen::Matrix3d R_err = R_meas_inv * R;
    Eigen::Vector3d t_err = R_meas_inv * (t - t_meas);
    
    // error = Logmap(error_pose) in GTSAM order [rot, trans]
    Vector6d error = SE3_Logmap(R_err, t_err);
    
    // Jacobian = I (for prior, derivative is identity at first order)
    J = Matrix6d::Identity();
    
    return error;
}

Matrix6d PoseGraphOptimizer::adjointSE3(const SE3d& T) const {
    // Adjoint of SE(3) for [rho (trans), phi (rot)] ordering
    // Ad_T = | R    t^R |
    //        | 0    R   |
    // where t^ = skew(t)
    Eigen::Matrix3d R = T.RotationMatrix();
    Eigen::Vector3d t = T.Translation();
    
    Matrix6d Ad = Matrix6d::Zero();
    Ad.block<3, 3>(0, 0) = R;                    // top-left: R
    Ad.block<3, 3>(3, 3) = R;                    // bottom-right: R
    Ad.block<3, 3>(0, 3) = Hatd(t) * R;          // top-right: t^R (trans-rot coupling)
    // bottom-left stays zero
    
    return Ad;
}

Matrix6d PoseGraphOptimizer::rightJacobianSE3(const Vector6d& xi) const {
    const Eigen::Vector3d phi = xi.tail<3>();
    const double theta = phi.norm();
    
    if (theta < kEpsLie) {
        Matrix6d Jr = Matrix6d::Identity();
        Jr.block<3, 3>(0, 0) -= 0.5 * Hatd(phi);
        Jr.block<3, 3>(3, 3) -= 0.5 * Hatd(phi);
        return Jr;
    }
    
    // SO3 right Jacobian
    Eigen::Matrix3d phi_hat = Hatd(phi);
    double theta2 = theta * theta;
    double sin_theta = std::sin(theta);
    double cos_theta = std::cos(theta);
    
    Eigen::Matrix3d Jr_so3 = Eigen::Matrix3d::Identity() - 
        (1.0 - cos_theta) / theta2 * phi_hat +
        (theta - sin_theta) / (theta2 * theta) * phi_hat * phi_hat;
    
    Matrix6d Jr = Matrix6d::Identity();
    Jr.block<3, 3>(0, 0) = Jr_so3;
    Jr.block<3, 3>(3, 3) = Jr_so3;
    
    return Jr;
}

Matrix6d PoseGraphOptimizer::rightJacobianInverseSE3(const Vector6d& xi) const {
    const Eigen::Vector3d phi = xi.tail<3>();
    const double theta = phi.norm();
    
    if (theta < kEpsLie) {
        Matrix6d Jr_inv = Matrix6d::Identity();
        Jr_inv.block<3, 3>(0, 0) += 0.5 * Hatd(phi);
        Jr_inv.block<3, 3>(3, 3) += 0.5 * Hatd(phi);
        return Jr_inv;
    }
    
    // SO3 right Jacobian inverse
    Eigen::Matrix3d phi_hat = Hatd(phi);
    double theta2 = theta * theta;
    double sin_theta = std::sin(theta);
    double cos_theta = std::cos(theta);
    
    Eigen::Matrix3d Jr_inv_so3 = Eigen::Matrix3d::Identity() + 
        0.5 * phi_hat +
        (1.0 / theta2 - (1.0 + cos_theta) / (2.0 * theta * sin_theta)) * phi_hat * phi_hat;
    
    Matrix6d Jr_inv = Matrix6d::Identity();
    Jr_inv.block<3, 3>(0, 0) = Jr_inv_so3;
    Jr_inv.block<3, 3>(3, 3) = Jr_inv_so3;
    
    return Jr_inv;
}

SE3d PoseGraphOptimizer::toDouble(const SE3f& pose) const {
    Eigen::Matrix4d mat = pose.Matrix().cast<double>();
    return SE3d::FromMatrix(mat);
}

SE3f PoseGraphOptimizer::toFloat(const SE3d& pose) const {
    Eigen::Matrix4f mat = pose.Matrix().cast<float>();
    return SE3f::FromMatrix(mat);
}

Matrix6d PoseGraphOptimizer::makeInformationMatrix(double trans_noise, double rot_noise) const {
    Matrix6d info = Matrix6d::Zero();
    
    // Information = 1 / sigma^2
    double trans_info = 1.0 / (trans_noise * trans_noise);
    double rot_info = 1.0 / (rot_noise * rot_noise);
    
    // GTSAM order: [rot, rot, rot, trans, trans, trans]
    info(0, 0) = rot_info;
    info(1, 1) = rot_info;
    info(2, 2) = rot_info;
    info(3, 3) = trans_info;
    info(4, 4) = trans_info;
    info(5, 5) = trans_info;
    
    return info;
}

} // namespace optimization
} // namespace lidar_slam
