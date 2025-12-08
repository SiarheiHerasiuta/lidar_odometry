/**
 * @file      PoseGraphOptimizer.cpp
 * @brief     Implementation of GTSAM ISAM2-based pose graph optimization (Incremental).
 * @author    Seungwon Choi
 * @date      2025-10-18
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 * 
 * @note This implementation follows LIO-SAM's incremental ISAM2 pattern:
 *       - Each keyframe addition triggers an ISAM2 update
 *       - Loop closures trigger multiple additional updates for better convergence
 *       - Pending factors/values are cleared after each update
 */

#include "PoseGraphOptimizer.h"
#include <spdlog/spdlog.h>
#include <chrono>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ISAM2.h>

namespace lidar_odometry {
namespace optimization {

namespace {
    gtsam::Pose3 se3_to_pose3(const Sophus::SE3f& se3) {
        Eigen::Matrix4d mat = se3.matrix().cast<double>();
        return gtsam::Pose3(mat);
    }
    
    Sophus::SE3f pose3_to_se3(const gtsam::Pose3& pose3) {
        // Get rotation and translation from GTSAM Pose3
        Eigen::Matrix3d R = pose3.rotation().matrix();
        Eigen::Vector3d t = pose3.translation();
        
        // Orthogonalize rotation matrix using SVD (closest orthogonal matrix)
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d R_ortho = svd.matrixU() * svd.matrixV().transpose();
        
        // Ensure proper rotation (det = +1)
        if (R_ortho.determinant() < 0) {
            R_ortho = -R_ortho;
        }
        
        // Construct SE3f from orthogonalized rotation and translation
        Sophus::SO3f so3(R_ortho.cast<float>());
        return Sophus::SE3f(so3, t.cast<float>());
    }
    
    gtsam::Key make_key(int keyframe_id) {
        return static_cast<gtsam::Key>(keyframe_id);
    }
}

PoseGraphOptimizer::PoseGraphOptimizer()
    : m_pending_graph(std::make_unique<gtsam::NonlinearFactorGraph>())
    , m_pending_values(std::make_unique<gtsam::Values>())
    , m_loop_closure_count(0)
    , m_odometry_count(0)
    , m_is_initialized(false) {
    
    gtsam::ISAM2Params params;
    params.relinearizeThreshold = 0.1;
    params.relinearizeSkip = 1;
    m_isam2 = std::make_unique<gtsam::ISAM2>(params);
    
    spdlog::debug("[PGO-ISAM2] Initialized with incremental update pattern");
}

PoseGraphOptimizer::~PoseGraphOptimizer() = default;

bool PoseGraphOptimizer::add_first_keyframe(int keyframe_id, const SE3f& pose) {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (!m_keyframe_ids.empty()) {
        spdlog::warn("[PGO-ISAM2] add_first_keyframe called but graph is not empty");
        return false;
    }
    
    gtsam::Pose3 gtsam_pose = se3_to_pose3(pose);
    
    // Add prior factor (fix first keyframe)
    // LIO-SAM: (Vector(6) << 1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8) - rad^2, m^2
    // We use tight prior for first keyframe
    gtsam::Vector6 prior_sigmas;
    prior_sigmas << 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4;  // rot, rot, rot, trans, trans, trans
    auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas(prior_sigmas);
    
    m_pending_graph->addPrior(make_key(keyframe_id), gtsam_pose, prior_noise);
    m_pending_values->insert(make_key(keyframe_id), gtsam_pose);
    
    // Run ISAM2 update
    try {
        m_isam2->update(*m_pending_graph, *m_pending_values);
        m_isam2->update();  // Extra update for better linearization
        
        // Clear pending (LIO-SAM pattern)
        m_pending_graph->resize(0);
        m_pending_values->clear();
        
        // Track keyframe
        m_keyframe_ids.push_back(keyframe_id);
        m_keyframe_set.insert(keyframe_id);
        m_is_initialized = true;
        
        spdlog::info("[PGO-ISAM2] Added first keyframe {} with prior", keyframe_id);
        return true;
        
    } catch (const std::exception& e) {
        spdlog::error("[PGO-ISAM2] Failed to add first keyframe: {}", e.what());
        m_pending_graph->resize(0);
        m_pending_values->clear();
        return false;
    }
}

bool PoseGraphOptimizer::add_keyframe_with_odom(int prev_keyframe_id, int curr_keyframe_id,
                                                 const SE3f& curr_pose,
                                                 const SE3f& relative_pose,
                                                 double odom_trans_noise,
                                                 double odom_rot_noise) {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    // Check if curr keyframe already exists
    if (m_keyframe_set.find(curr_keyframe_id) != m_keyframe_set.end()) {
        spdlog::debug("[PGO-ISAM2] Keyframe {} already exists in graph", curr_keyframe_id);
        return true;  // Not an error, just skip
    }
    
    gtsam::Pose3 gtsam_curr_pose = se3_to_pose3(curr_pose);
    
    // Check if prev keyframe exists
    bool prev_exists = (m_keyframe_set.find(prev_keyframe_id) != m_keyframe_set.end());
    
    if (prev_exists) {
        // Normal case: add odometry constraint from previous keyframe
        gtsam::Pose3 gtsam_relative = se3_to_pose3(relative_pose);
        
        // Odometry noise model
        gtsam::Vector6 odom_sigmas;
        odom_sigmas << odom_rot_noise, odom_rot_noise, odom_rot_noise,
                       odom_trans_noise, odom_trans_noise, odom_trans_noise;
        auto odom_noise = gtsam::noiseModel::Diagonal::Sigmas(odom_sigmas);
        
        // Add odometry factor (BetweenFactor)
        m_pending_graph->emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
            make_key(prev_keyframe_id),
            make_key(curr_keyframe_id),
            gtsam_relative,
            odom_noise
        );
    } else {
        // Previous keyframe not in graph - add with loose prior instead
        // This can happen if there was a gap due to race conditions
        spdlog::warn("[PGO-ISAM2] Previous keyframe {} not found, adding {} with loose prior", 
                    prev_keyframe_id, curr_keyframe_id);
        
        // Loose prior (allows optimization to adjust)
        gtsam::Vector6 prior_sigmas;
        prior_sigmas << 0.1, 0.1, 0.1, 0.5, 0.5, 0.5;  // looser than first keyframe
        auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas(prior_sigmas);
        
        m_pending_graph->addPrior(make_key(curr_keyframe_id), gtsam_curr_pose, prior_noise);
    }
    
    // Add initial value for new keyframe
    m_pending_values->insert(make_key(curr_keyframe_id), gtsam_curr_pose);
    
    // Run ISAM2 update
    try {
        auto start = std::chrono::high_resolution_clock::now();
        
        m_isam2->update(*m_pending_graph, *m_pending_values);
        m_isam2->update();  // Extra update for better linearization
        
        // Clear pending (LIO-SAM pattern)
        m_pending_graph->resize(0);
        m_pending_values->clear();
        
        auto end = std::chrono::high_resolution_clock::now();
        double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        // Track keyframe
        m_keyframe_ids.push_back(curr_keyframe_id);
        m_keyframe_set.insert(curr_keyframe_id);
        m_odometry_count++;
        
        if (prev_exists) {
            spdlog::debug("[PGO-ISAM2] Added keyframe {} with odom from {} ({:.2f}ms)", 
                         curr_keyframe_id, prev_keyframe_id, duration_ms);
        } else {
            spdlog::info("[PGO-ISAM2] Added keyframe {} with loose prior ({:.2f}ms)", 
                        curr_keyframe_id, duration_ms);
        }
        return true;
        
    } catch (const std::exception& e) {
        spdlog::error("[PGO-ISAM2] Failed to add keyframe {}: {}", curr_keyframe_id, e.what());
        m_pending_graph->resize(0);
        m_pending_values->clear();
        return false;
    }
}

bool PoseGraphOptimizer::add_loop_and_optimize(int from_keyframe_id, int to_keyframe_id,
                                                const SE3f& relative_pose,
                                                double loop_trans_noise,
                                                double loop_rot_noise) {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    // Check if both keyframes exist
    if (m_keyframe_set.find(from_keyframe_id) == m_keyframe_set.end()) {
        spdlog::error("[PGO-ISAM2] Loop from-keyframe {} not found", from_keyframe_id);
        return false;
    }
    if (m_keyframe_set.find(to_keyframe_id) == m_keyframe_set.end()) {
        spdlog::error("[PGO-ISAM2] Loop to-keyframe {} not found", to_keyframe_id);
        return false;
    }
    
    gtsam::Pose3 gtsam_relative = se3_to_pose3(relative_pose);
    
    // Loop closure noise model (typically tighter than odometry)
    gtsam::Vector6 loop_sigmas;
    loop_sigmas << loop_rot_noise, loop_rot_noise, loop_rot_noise,
                   loop_trans_noise, loop_trans_noise, loop_trans_noise;
    auto loop_noise = gtsam::noiseModel::Diagonal::Sigmas(loop_sigmas);
    
    // Add loop closure factor
    m_pending_graph->emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
        make_key(from_keyframe_id),
        make_key(to_keyframe_id),
        gtsam_relative,
        loop_noise
    );
    
    // Run ISAM2 update with extra iterations (LIO-SAM pattern)
    try {
        // First update with new factor
        m_isam2->update(*m_pending_graph, *m_pending_values);
        m_isam2->update();
        
        // Extra updates for loop closure convergence (LIO-SAM does 5 extra updates)
        m_isam2->update();
        m_isam2->update();
        m_isam2->update();
        m_isam2->update();
        m_isam2->update();
        
        // Clear pending
        m_pending_graph->resize(0);
        m_pending_values->clear();
        
        m_loop_closure_count++;
        
        return true;
        
    } catch (const std::exception& e) {
        spdlog::error("[PGO-ISAM2] Loop closure optimization failed: {}", e.what());
        m_pending_graph->resize(0);
        m_pending_values->clear();
        return false;
    }
}

bool PoseGraphOptimizer::get_optimized_pose(int keyframe_id, SE3f& optimized_pose) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (!m_is_initialized) {
        return false;
    }
    
    if (m_keyframe_set.find(keyframe_id) == m_keyframe_set.end()) {
        return false;
    }
    
    try {
        gtsam::Values current_estimate = m_isam2->calculateEstimate();
        gtsam::Key key = make_key(keyframe_id);
        
        if (!current_estimate.exists(key)) {
            return false;
        }
        
        gtsam::Pose3 gtsam_pose = current_estimate.at<gtsam::Pose3>(key);
        optimized_pose = pose3_to_se3(gtsam_pose);
        return true;
        
    } catch (const std::exception& e) {
        spdlog::error("[PGO-ISAM2] Failed to get pose for keyframe {}: {}", keyframe_id, e.what());
        return false;
    }
}

std::map<int, PoseGraphOptimizer::SE3f> PoseGraphOptimizer::get_all_optimized_poses() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    std::map<int, SE3f> result;
    
    if (!m_is_initialized) {
        return result;
    }
    
    try {
        gtsam::Values current_estimate = m_isam2->calculateEstimate();
        
        for (int keyframe_id : m_keyframe_ids) {
            gtsam::Key key = make_key(keyframe_id);
            if (current_estimate.exists(key)) {
                gtsam::Pose3 gtsam_pose = current_estimate.at<gtsam::Pose3>(key);
                result[keyframe_id] = pose3_to_se3(gtsam_pose);
            }
        }
    } catch (const std::exception& e) {
        spdlog::error("[PGO-ISAM2] Failed to get all poses: {}", e.what());
    }
    
    return result;
}

void PoseGraphOptimizer::clear() {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    // Reset ISAM2 instance
    gtsam::ISAM2Params params;
    params.relinearizeThreshold = 0.1;
    params.relinearizeSkip = 1;
    m_isam2 = std::make_unique<gtsam::ISAM2>(params);
    
    // Clear pending
    m_pending_graph = std::make_unique<gtsam::NonlinearFactorGraph>();
    m_pending_values = std::make_unique<gtsam::Values>();
    
    // Clear tracking
    m_keyframe_ids.clear();
    m_keyframe_set.clear();
    
    m_loop_closure_count = 0;
    m_odometry_count = 0;
    m_is_initialized = false;
    
    spdlog::debug("[PGO-ISAM2] Cleared pose graph");
}

} // namespace optimization
} // namespace lidar_odometry
