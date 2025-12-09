/**
 * @file      ICPConfig.h
 * @brief     ICP configuration and related types for LiDAR odometry.
 * @author    Seungwon Choi
 * @date      2025-10-07
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#pragma once

#include <memory>
#include <vector>
#include "Types.h"
#include <Eigen/Dense>

namespace lidar_odometry {
namespace util {

// Type aliases for ICP - using our point cloud implementation
using ICPPointType = Point3D;
using ICPPointCloud = PointCloud;
using ICPPointCloudPtr = std::shared_ptr<ICPPointCloud>;
using ICPPointCloudConstPtr = std::shared_ptr<const ICPPointCloud>;
using ICPPose = SE3f;
using ICPVector3f = Eigen::Vector3f;

/**
 * @brief Configuration for ICP algorithm
 */
struct ICPConfig {
    // Convergence criteria
    int max_iterations = 50;
    double translation_tolerance = 1e-6;  // meters
    double rotation_tolerance = 1e-6;     // radians
    
    // Correspondence parameters
    double max_correspondence_distance = 1.0;  // meters
    int min_correspondence_points = 10;
    
    // Outlier rejection
    double outlier_rejection_ratio = 0.9;  // Keep top 90% of correspondences
    bool use_robust_loss = true;
    double robust_loss_delta = 0.1;  // Huber loss delta
    
    // Performance
    bool use_kdtree = true;
    int max_kdtree_neighbors = 1;
};

/**
 * @brief Point-to-plane correspondence for ICP
 */
struct ICPPointCorrespondence {
    ICPVector3f source_point;
    ICPVector3f target_point;
    ICPVector3f plane_normal;    // Normal vector of target plane
    double distance = 0.0;
    double weight = 1.0;
    bool is_valid = false;
    
    ICPPointCorrespondence() = default;
    ICPPointCorrespondence(const ICPVector3f& src, const ICPVector3f& tgt, double dist)
        : source_point(src), target_point(tgt), distance(dist), weight(1.0), is_valid(true) {}
    ICPPointCorrespondence(const ICPVector3f& src, const ICPVector3f& tgt, const ICPVector3f& normal, double dist)
        : source_point(src), target_point(tgt), plane_normal(normal), distance(dist), weight(1.0), is_valid(true) {}
};

using ICPCorrespondenceVector = std::vector<ICPPointCorrespondence>;

/**
 * @brief ICP statistics for monitoring convergence
 */
struct ICPStatistics {
    int iterations_used = 0;
    double final_cost = 0.0;
    double initial_cost = 0.0;
    size_t correspondences_count = 0;
    size_t inlier_count = 0;
    double match_ratio = 0.0;
    bool converged = false;
    
    void reset() {
        iterations_used = 0;
        final_cost = 0.0;
        initial_cost = 0.0;
        correspondences_count = 0;
        inlier_count = 0;
        match_ratio = 0.0;
        converged = false;
    }
};

} // namespace util
} // namespace lidar_odometry