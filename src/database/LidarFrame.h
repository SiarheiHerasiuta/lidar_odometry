/**
 * @file      LidarFrame.h
 * @brief     LiDAR frame data structure for odometry system.
 * @author    Seungwon Choi
 * @date      2025-09-24
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#pragma once

#include <memory>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "../util/MathUtils.h"
#include "../util/PointCloudUtils.h"

// Type definitions (declared here as needed)
namespace lidar_slam {
    // Import types from util namespace
    using namespace lidar_slam::util;
    
    // Import types from util namespace
    using namespace util;
    
    // KdTree type
    using KdTree = util::KdTree;
    using KdTreePtr = std::shared_ptr<KdTree>;
    
    // Feature types
    struct Correspondence {
        Vector3f source_point;     // Point in source frame
        Vector3f target_point;     // Point in target frame  
        Vector3f plane_normal;     // Normal vector for plane fitting
        float distance;            // Point-to-plane distance
        bool is_valid;             // Validity flag
        
        Correspondence() : distance(0.0f), is_valid(false) {}
    };
    
    using CorrespondenceVector = std::vector<Correspondence>;
}

namespace lidar_slam {
namespace database {

/**
 * @brief LiDAR Frame class for storing frame data and pose information
 * 
 * This class represents a single LiDAR frame with associated data:
 * - Raw point cloud data
 * - Pose information (current and relative)
 * - Preprocessing results (downsampled cloud, normals)
 * - Feature extraction results
 * - Optimization status
 */
class LidarFrame {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    /**
     * @brief Constructor
     * @param frame_id Unique frame identifier
     * @param timestamp Frame timestamp in seconds
     * @param raw_cloud Raw point cloud data
     */
    LidarFrame(int frame_id, 
               double timestamp,
               const PointCloudPtr& raw_cloud);
    
    /**
     * @brief Destructor
     */
    ~LidarFrame() = default;
    
    /**
     * @brief Copy constructor - creates deep copy of all data
     * @param other Frame to copy from
     */
    LidarFrame(const LidarFrame& other);
    
    // Copy and move constructors/operators
    LidarFrame& operator=(const LidarFrame&) = delete;
    LidarFrame(LidarFrame&&) = default;
    LidarFrame& operator=(LidarFrame&&) = default;
    
    // ===== Basic Getters =====
    int get_frame_id() const { return m_frame_id; }
    int get_keyframe_id() const { return m_keyframe_id; }
    bool is_keyframe() const { return m_keyframe_id >= 0; }
    double get_timestamp() const { return m_timestamp; }
    
    // ===== Pose Management =====
    
    /**
     * @brief Set current pose (world frame)
     * @param pose SE3 pose in world coordinates
     */
    void set_pose(const SE3f& pose);
    
    /**
     * @brief Get current pose (world frame)
     * Dynamic calculation: if keyframe, return m_pose; else return prev_keyframe->pose * m_relative_pose
     * @return SE3 pose in world coordinates
     */
    SE3f get_pose() const;
    
    /**
     * @brief Get stored pose (for keyframes only, returns actual stored value)
     * @return Reference to stored SE3 pose
     */
    const SE3f& get_stored_pose() const { return m_pose; }
    
    /**
     * @brief Set previous keyframe (for non-keyframes to compute pose dynamically)
     * @param prev_kf Shared pointer to previous keyframe
     */
    void set_previous_keyframe(std::shared_ptr<LidarFrame> prev_kf) { 
        m_previous_keyframe = prev_kf; 
    }
    
    /**
     * @brief Get previous keyframe
     * @return Shared pointer to previous keyframe
     */
    std::shared_ptr<LidarFrame> get_previous_keyframe() const { 
        return m_previous_keyframe.lock(); 
    }
    
    /**
     * @brief Set relative pose (to previous frame)
     * @param relative_pose SE3 relative transformation
     */
    void set_relative_pose(const SE3f& relative_pose);
    
    /**
     * @brief Get relative pose (to previous frame)
     * @return SE3 relative transformation
     */
    const SE3f& get_relative_pose() const { return m_relative_pose; }
    
    /**
     * @brief Set ground truth pose for evaluation
     * @param gt_pose Ground truth SE3 pose
     */
    void set_ground_truth_pose(const SE3f& gt_pose);
    
    /**
     * @brief Get ground truth pose
     * @return Ground truth SE3 pose
     */
    const SE3f& get_ground_truth_pose() const { return m_ground_truth_pose; }
    
    /**
     * @brief Set initial pose for this frame (e.g., from other sensors)
     * @param initial_pose Initial SE3 pose estimate
     */
    void set_initial_pose(const SE3f& initial_pose);
    
    /**
     * @brief Get initial pose for this frame
     * @return Initial SE3 pose estimate
     */
    const SE3f& get_initial_pose() const { return m_initial_pose; }

    // ===== Point Cloud Management =====
    
    /**
     * @brief Get raw point cloud
     * @return Shared pointer to raw point cloud
     */
    PointCloudPtr get_raw_cloud() { return m_raw_cloud; }
    PointCloudConstPtr get_raw_cloud() const { return m_raw_cloud; }
    
    /**
     * @brief Set processed point cloud (after filtering/downsampling)
     * @param processed_cloud Processed point cloud
     */
    void set_processed_cloud(const PointCloudPtr& processed_cloud);
    
    /**
     * @brief Get processed point cloud
     * @return Shared pointer to processed point cloud
     */
    PointCloudPtr get_processed_cloud() { return m_processed_cloud; }
    PointCloudConstPtr get_processed_cloud() const { return m_processed_cloud; }
    
    /**
     * @brief Set feature point cloud (extracted features)
     * @param feature_cloud Feature point cloud
     */
    void set_feature_cloud(const PointCloudPtr& feature_cloud);

    
    /**
     * @brief Set feature point cloud in global coordinates (lazy evaluation)
     * @param feature_cloud_global Feature point cloud in world coordinates
     */
    void set_feature_cloud_global(const PointCloudPtr& feature_cloud_global);

    


    /**
     * @brief Get feature point cloud
     * @return Shared pointer to feature point cloud
     */
    PointCloudPtr get_feature_cloud() { return m_feature_cloud; }
    PointCloudConstPtr get_feature_cloud() const { return m_feature_cloud; }
    PointCloudConstPtr get_feature_cloud_global() const { return m_feature_cloud_global; }
    
    /**
     * @brief Transform point cloud to world coordinates using current pose
     * @return Point cloud in world coordinates
     */
    PointCloudPtr get_world_cloud() const;
    
    /**
     * @brief Set local map for keyframe (features within local region at keyframe creation time)
     * @param local_map Local map point cloud in world coordinates
     */
    void set_local_map(const PointCloudPtr& local_map);
    
    /**
     * @brief Get local map for keyframe
     * @return Shared pointer to local map point cloud
     */
    PointCloudPtr get_local_map() { return m_local_map; }
    PointCloudConstPtr get_local_map() const { return m_local_map; }
    
    /**
     * @brief Clear local map to free memory
     */
    void clear_local_map();
    
    /**
     * @brief Clear heavy data for old keyframes (keep only feature_cloud and pose)
     * Clears: raw_cloud, processed_cloud, local_map, kdtrees, correspondences
     */
    void clear_heavy_data_for_old_keyframe();
    
    /**
     * @brief Clear all clouds for non-keyframe (keep only pose and relative_pose)
     * Clears: raw_cloud, processed_cloud, feature_cloud, correspondences, kdtrees
     */
    void clear_non_keyframe_data();
    
    // ===== KdTree Management =====
    
    /**
     * @brief Build KdTree for processed cloud (lazy initialization)
     */
    void build_kdtree();
    
    /**
     * @brief Get KdTree for nearest neighbor search
     * @return Shared pointer to KdTree (builds if not exists)
     */
    KdTreePtr get_kdtree();
    
    /**
     * @brief Build KdTree for local map (for keyframes)
     */
    void build_local_map_kdtree();
    
    /**
     * @brief Clear KdTree for local map to save memory
     */
    void clear_local_map_kdtree();
    
    /**
     * @brief Get KdTree for local map
     * @return Shared pointer to local map KdTree
     */
    KdTreePtr get_local_map_kdtree();
    
    // ===== Feature Management =====
    
    /**
     * @brief Set feature correspondences for ICP
     * @param correspondences Vector of point correspondences
     */
    void set_correspondences(const CorrespondenceVector& correspondences);
    
    /**
     * @brief Get feature correspondences
     * @return Vector of point correspondences
     */
    const CorrespondenceVector& get_correspondences() const { return m_correspondences; }
    
    /**
     * @brief Clear correspondences (e.g., after optimization)
     */
    void clear_correspondences();
    
    // ===== Optimization Status =====
    
    /**
     * @brief Set optimization status
     * @param is_fixed True if pose should not be optimized
     */
    void set_fixed(bool is_fixed) { m_is_fixed = is_fixed; }
    
    /**
     * @brief Check if frame is fixed in optimization
     * @return True if pose is fixed
     */
    bool is_fixed() const { return m_is_fixed; }
    
    /**
     * @brief Set keyframe ID and status
     * @param keyframe_id Keyframe identifier (-1 if not a keyframe)
     */
    void set_keyframe_id(int keyframe_id) { m_keyframe_id = keyframe_id; }
    
    /**
     * @brief Set keyframe status (deprecated - use set_keyframe_id instead)
     * @param is_keyframe True if this frame is a keyframe
     */
    void set_keyframe(bool is_keyframe) { 
        // For backward compatibility, assign next available ID or -1
        m_keyframe_id = is_keyframe ? 0 : -1; // This should be properly managed by Estimator
    }
    
    // ===== Statistics =====
    
    /**
     * @brief Get number of points in raw cloud
     * @return Number of points
     */
    size_t get_raw_point_count() const;
    
    /**
     * @brief Get number of points in processed cloud
     * @return Number of points
     */
    size_t get_processed_point_count() const;
    
    /**
     * @brief Get number of valid correspondences
     * @return Number of correspondences
     */
    size_t get_correspondence_count() const;
    
    /**
     * @brief Compute distance to another frame
     * @param other Other LiDAR frame
     * @return Euclidean distance between poses
     */
    float compute_distance_to(const LidarFrame& other) const;

private:
    // ===== Core Frame Data =====
    int m_frame_id;                    // Unique frame identifier
    int m_keyframe_id;                 // Keyframe identifier (-1 if not a keyframe)
    double m_timestamp;                // Frame timestamp
    
    // ===== Pose Data =====
    SE3f m_pose;                       // Current pose in world frame (for keyframes) or cached pose
    SE3f m_relative_pose;              // Relative pose to previous keyframe
    SE3f m_ground_truth_pose;          // Ground truth pose (for evaluation)
    SE3f m_initial_pose;               // Initial pose estimate (e.g., from other sensors)
    
    // ===== Frame Relationship =====
    std::weak_ptr<LidarFrame> m_previous_keyframe;  // Previous keyframe (for pose computation)
    
    // ===== Point Cloud Data =====
    PointCloudPtr m_raw_cloud;         // Original point cloud
    PointCloudPtr m_processed_cloud;   // Processed point cloud (filtered/downsampled)
    PointCloudPtr m_feature_cloud;     // Feature point cloud
    PointCloudPtr m_feature_cloud_global;       // Cached world coordinate cloud (lazy evaluation)
    PointCloudPtr m_local_map;         // Local map at keyframe creation time (world coordinates)
    
    // ===== Search Structure =====
    KdTreePtr m_kdtree;                // KdTree for nearest neighbor search
    bool m_kdtree_built;               // Flag to track KdTree status
    KdTreePtr m_local_map_kdtree;      // KdTree for local map (keyframes)
    bool m_local_map_kdtree_built;     // Flag to track local map KdTree status
    
    // ===== Feature Data =====
    CorrespondenceVector m_correspondences; // Point correspondences for ICP
    
    // ===== Status Flags =====
    bool m_is_fixed;                   // True if pose should not be optimized
    bool m_has_ground_truth;           // True if ground truth pose is available
};

// Convenience typedefs
using LidarFramePtr = std::shared_ptr<LidarFrame>;
using LidarFrameConstPtr = std::shared_ptr<const LidarFrame>;

} // namespace database
} // namespace lidar_slam
