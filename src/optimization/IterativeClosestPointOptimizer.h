/**
 * @file      IterativeClosestPointOptimizer.h
 * @brief     Two-frame ICP optimizer using Gauss-Newton
 * @author    Seungwon Choi
 * @date      2025-10-04
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */


#pragma once

#include "../util/Types.h"
#include "../database/LidarFrame.h"
#include "../map/VoxelMap.h"
#include "AdaptiveMEstimator.h"
#include "../util/ICPConfig.h"

#include <memory>
#include <vector>

namespace lidar_odometry {
namespace optimization {

// Import types from util namespace
using namespace lidar_odometry::util;

/**
 * @brief Two-frame correspondence data
 */
struct DualFrameCorrespondences {
    std::vector<Eigen::Vector3d> points_last; ///< Points in last frame (local coordinates)
    std::vector<Eigen::Vector3d> points_curr; ///< Points in curr frame (local coordinates) 
    std::vector<Eigen::Vector3d> normals_last; ///< Normal vectors at last frame points (world coordinates)
    std::vector<double> residuals;         ///< Raw residuals for robust estimation
    
    void clear() {
        points_last.clear();
        points_curr.clear();
        normals_last.clear();
        residuals.clear();
    }
    
    size_t size() const {
        return points_last.size();
    }
};

/**
 * @brief Simple two-frame ICP optimizer
 * 
 * This class implements point-to-plane ICP between exactly two frames.
 * Much simpler than MultiFrameOptimizer for debugging and basic use cases.
 */
class IterativeClosestPointOptimizer {
public:
    /**
     * @brief Constructor
     * @param config ICP configuration
     */
    explicit IterativeClosestPointOptimizer(const ICPConfig& config = ICPConfig());
    
    /**
     * @brief Constructor with AdaptiveMEstimator
     * @param config ICP configuration
     * @param adaptive_estimator Pointer to AdaptiveMEstimator for robust optimization
     */
    IterativeClosestPointOptimizer(const ICPConfig& config, 
                         std::shared_ptr<optimization::AdaptiveMEstimator> adaptive_estimator);
    
    /**
     * @brief Destructor
     */
    ~IterativeClosestPointOptimizer() = default;
    
    /**
     * @brief Optimize relative pose between two frames using VoxelMap
     * 
     * @param voxel_map Pointer to VoxelMap with precomputed surfels
     * @param curr_frame Current frame (optimized)
     * @param initial_transform Initial relative transform guess (curr relative to keyframe)
     * @param optimized_transform Output optimized relative transform
     * @return True if optimization succeeded
     */
    bool optimize(map::VoxelMap* voxel_map,
                 std::shared_ptr<database::LidarFrame> curr_frame,
                 const SE3f& initial_transform,
                 SE3f& optimized_transform);

    /**
     * @brief Optimize relative pose between two keyframes for loop closure
     * @param curr_keframe Current keyframe (has fresh kdtree built)
     * @param matched_keyframe Matched keyframe from loop closure detection
     * @param optimized_relative_transform Output optimized relative transform (curr to matched)
     * @return True if optimization succeeded
     */
    
    bool optimize_loop(std::shared_ptr<database::LidarFrame> curr_keframe,
                 std::shared_ptr<database::LidarFrame> matched_keyframe,
                 SE3f& optimized_relative_transform,
                 float& inlier_ratio);
    
    /**
     * @brief Get optimization statistics
     */
    struct OptimizationStats {
        size_t num_correspondences = 0;
        size_t num_iterations = 0;
        double initial_cost = 0.0;
        double final_cost = 0.0;
        double optimization_time_ms = 0.0;
        bool converged = false;
    };
    
    /**
     * @brief Get last optimization statistics
     */
    const OptimizationStats& get_last_stats() const { return m_last_stats; }
    
    /**
     * @brief Update configuration
     */
    void update_config(const ICPConfig& config) { m_config = config; }
    
    /**
     * @brief Get current configuration
     */
    const ICPConfig& get_config() const { return m_config; }

private:
    /**
     * @brief Find correspondences using VoxelMap surfels
     */
    size_t find_correspondences(map::VoxelMap* voxel_map,
                               std::shared_ptr<database::LidarFrame> curr_frame,
                               DualFrameCorrespondences& correspondences);
    
    
    /**
     * @brief Find correspondences between two keyframes for loop closure
     */
    size_t find_correspondences_loop(std::shared_ptr<database::LidarFrame> last_keyframe,
                                     std::shared_ptr<database::LidarFrame> curr_keyframe,
                                     DualFrameCorrespondences &correspondences);

    /**
     * @brief Extract point cloud for correspondence finding
     */
    PointCloudConstPtr get_frame_cloud(std::shared_ptr<database::LidarFrame> frame);
    
    /**
     * @brief Check if points are collinear
     */
    bool is_collinear(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2, 
                     const Eigen::Vector3d& p3, double threshold = 0.5);

    // Member variables
    ICPConfig m_config;
    OptimizationStats m_last_stats;
    std::shared_ptr<optimization::AdaptiveMEstimator> m_adaptive_estimator;
};

} // namespace processing
} // namespace lidar_odometry