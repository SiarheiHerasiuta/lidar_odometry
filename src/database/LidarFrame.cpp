/**
 * @file      LidarFrame.cpp
 * @brief     Implementation of LiDAR frame data structure.
 * @author    Seungwon Choi
 * @date      2025-09-24
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "LidarFrame.h"
#include "../util/PointCloudUtils.h"
#include <spdlog/spdlog.h>

namespace lidar_odometry {
namespace database {

LidarFrame::LidarFrame(int frame_id, 
                       double timestamp,
                       const PointCloudPtr& raw_cloud)
    : m_frame_id(frame_id)
    , m_keyframe_id(-1)  // Initialize as not a keyframe
    , m_timestamp(timestamp)
    , m_pose(SE3f())
    , m_relative_pose(SE3f())
    , m_ground_truth_pose(SE3f())
    , m_initial_pose(SE3f())  // Initialize as identity
    , m_raw_cloud(raw_cloud)
    , m_processed_cloud(nullptr)
    , m_feature_cloud(nullptr)
    , m_kdtree(nullptr)
    , m_kdtree_built(false)
    , m_local_map_kdtree(nullptr)
    , m_local_map_kdtree_built(false)
    , m_is_fixed(false)
    , m_has_ground_truth(false) {
    
    if (!m_raw_cloud) {
        spdlog::warn("[LidarFrame] Frame {} initialized with null raw cloud", m_frame_id);
        m_raw_cloud = std::make_shared<PointCloud>();
    }
    
    // Reserve space for correspondences
    m_correspondences.reserve(1000); // Reserve for typical correspondence count
}

LidarFrame::LidarFrame(const LidarFrame& other)
    : m_frame_id(other.m_frame_id)
    , m_keyframe_id(other.m_keyframe_id)
    , m_timestamp(other.m_timestamp)
    , m_pose(other.m_pose)
    , m_relative_pose(other.m_relative_pose)
    , m_ground_truth_pose(other.m_ground_truth_pose)
    , m_initial_pose(other.m_initial_pose)
    , m_kdtree_built(false)  // Don't copy kdtree, rebuild if needed
    , m_local_map_kdtree_built(false)
    , m_is_fixed(other.m_is_fixed)
    , m_has_ground_truth(other.m_has_ground_truth) {
    
    // Deep copy point clouds
    if (other.m_raw_cloud) {
        m_raw_cloud = std::make_shared<PointCloud>(*other.m_raw_cloud);
    } else {
        m_raw_cloud = nullptr;
    }
    
    if (other.m_processed_cloud) {
        m_processed_cloud = std::make_shared<PointCloud>(*other.m_processed_cloud);
    } else {
        m_processed_cloud = nullptr;
    }
    
    if (other.m_feature_cloud) {
        m_feature_cloud = std::make_shared<PointCloud>(*other.m_feature_cloud);
    } else {
        m_feature_cloud = nullptr;
    }
    
    if (other.m_feature_cloud_global) {
        m_feature_cloud_global = std::make_shared<PointCloud>(*other.m_feature_cloud_global);
    } else {
        m_feature_cloud_global = nullptr;
    }
    
    if (other.m_local_map) {
        m_local_map = std::make_shared<PointCloud>(*other.m_local_map);
    } else {
        m_local_map = nullptr;
    }
    
    // Copy correspondences
    m_correspondences = other.m_correspondences;
    
    // Rebuild KdTrees if they existed in the source
    if (other.m_kdtree_built && other.m_kdtree) {
        build_kdtree();
    } else {
        m_kdtree = nullptr;
        m_kdtree_built = false;
    }
    
    if (other.m_local_map_kdtree_built && other.m_local_map_kdtree) {
        build_local_map_kdtree();
    } else {
        m_local_map_kdtree = nullptr;
        m_local_map_kdtree_built = false;
    }
}

// ===== Pose Management =====

SE3f LidarFrame::get_pose() const {
    // If this is a keyframe, return the stored pose
    if (is_keyframe()) {
        return m_pose;
    }
    
    // If this is a regular frame, compute pose from previous keyframe
    auto prev_kf = m_previous_keyframe.lock();
    if (prev_kf) {
        // Pose = prev_keyframe_pose * relative_pose
        return prev_kf->get_pose() * m_relative_pose;
    }
    
    // Fallback: return stored pose (should not happen in normal operation)
    return m_pose;
}

void LidarFrame::set_pose(const SE3f& pose) {
    m_pose = pose;
    
    // Invalidate world cloud cache and KdTree when pose changes
    // (This would be handled by a caching mechanism in a full implementation)
    m_kdtree_built = false;
    m_kdtree.reset();
}

void LidarFrame::set_relative_pose(const SE3f& relative_pose) {
    m_relative_pose = relative_pose;
}

void LidarFrame::set_ground_truth_pose(const SE3f& gt_pose) {
    m_ground_truth_pose = gt_pose;
    m_has_ground_truth = true;
}

void LidarFrame::set_initial_pose(const SE3f& initial_pose) {
    m_initial_pose = initial_pose;
}

// ===== Point Cloud Management =====

void LidarFrame::set_processed_cloud(const PointCloudPtr& processed_cloud) {
    m_processed_cloud = processed_cloud;
    
    // Invalidate KdTree when processed cloud changes
    m_kdtree_built = false;
    m_kdtree.reset();
    
    if (!m_processed_cloud) {
        spdlog::warn("[LidarFrame] Frame {} processed cloud set to null", m_frame_id);
    }
}

void LidarFrame::set_feature_cloud(const PointCloudPtr& feature_cloud) {
    m_feature_cloud = feature_cloud;
    
    if (!m_feature_cloud) {
        spdlog::warn("[LidarFrame] Frame {} feature cloud set to null", m_frame_id);
    }

    // util::transform_point_cloud(m_feature_cloud, m_feature_cloud_global, m_pose.Matrix());

}

void LidarFrame::set_feature_cloud_global(const PointCloudPtr& feature_cloud_global) {
    m_feature_cloud_global = feature_cloud_global;
    
    if (!m_feature_cloud_global) {
        spdlog::warn("[LidarFrame] Frame {} global feature cloud set to null", m_frame_id);
    }
}

void LidarFrame::set_local_map(const PointCloudPtr& local_map) {
    m_local_map = local_map;
    
    if (!m_local_map) {
        spdlog::warn("[LidarFrame] Frame {} local map set to null", m_frame_id);
    } else {
        spdlog::debug("[LidarFrame] Frame {} local map set with {} points", m_frame_id, m_local_map->size());
    }
}

PointCloudPtr LidarFrame::get_world_cloud() const {
    PointCloudPtr cloud_to_transform = m_processed_cloud ? m_processed_cloud : m_raw_cloud;
    
    if (!cloud_to_transform || cloud_to_transform->empty()) {
        spdlog::warn("[LidarFrame] Frame {} has no valid cloud for world transformation", m_frame_id);
        return std::make_shared<PointCloud>();
    }
    
    // Transform cloud to world coordinates
    PointCloudPtr world_cloud = std::make_shared<PointCloud>();
    
    // Convert SE3f to Matrix4f for transformation
    Matrix4f transform_matrix = m_pose.Matrix();
    
    try {
        util::transform_point_cloud(cloud_to_transform, world_cloud, transform_matrix);
    } catch (const std::exception& e) {
        spdlog::error("[LidarFrame] Frame {} world cloud transformation failed: {}", m_frame_id, e.what());
        return std::make_shared<PointCloud>();
    }
    
    return world_cloud;
}

// ===== KdTree Management =====

void LidarFrame::build_kdtree() {
    PointCloudPtr cloud_for_kdtree = m_processed_cloud ? m_processed_cloud : m_raw_cloud;
    
    if (!cloud_for_kdtree || cloud_for_kdtree->empty()) {
        spdlog::warn("[LidarFrame] Frame {} cannot build KdTree: no valid cloud", m_frame_id);
        m_kdtree_built = false;
        return;
    }
    
    try {
        m_kdtree = std::make_shared<KdTree>();
        m_kdtree->setInputCloud(cloud_for_kdtree);
        m_kdtree_built = true;
        
        spdlog::debug("[LidarFrame] Frame {} KdTree built with {} points", 
                     m_frame_id, cloud_for_kdtree->size());
        
    } catch (const std::exception& e) {
        spdlog::error("[LidarFrame] Frame {} KdTree build failed: {}", m_frame_id, e.what());
        m_kdtree.reset();
        m_kdtree_built = false;
    }
}

KdTreePtr LidarFrame::get_kdtree() {
    if (!m_kdtree_built || !m_kdtree) {
        build_kdtree();
    }
    
    return m_kdtree;
}

// ===== Feature Management =====

void LidarFrame::set_correspondences(const CorrespondenceVector& correspondences) {
    m_correspondences = correspondences;
    
    spdlog::debug("[LidarFrame] Frame {} correspondences set: {} total, {} valid", 
                 m_frame_id, m_correspondences.size(), get_correspondence_count());
}

void LidarFrame::clear_correspondences() {
    m_correspondences.clear();
    
    spdlog::debug("[LidarFrame] Frame {} correspondences cleared", m_frame_id);
}

// ===== Statistics =====

size_t LidarFrame::get_raw_point_count() const {
    return m_raw_cloud ? m_raw_cloud->size() : 0;
}

size_t LidarFrame::get_processed_point_count() const {
    return m_processed_cloud ? m_processed_cloud->size() : 0;
}

size_t LidarFrame::get_correspondence_count() const {
    size_t valid_count = 0;
    for (const auto& corr : m_correspondences) {
        if (corr.is_valid) {
            valid_count++;
        }
    }
    return valid_count;
}

float LidarFrame::compute_distance_to(const LidarFrame& other) const {
    Vector3f translation_diff = m_pose.Translation() - other.m_pose.Translation();
    return translation_diff.norm();
}

void LidarFrame::build_local_map_kdtree() {


    if (!m_local_map || m_local_map->empty()) {
        spdlog::warn("[LidarFrame] Cannot build KdTree: local map is empty for frame {}", m_frame_id);
        m_local_map_kdtree_built = false;
        return;
    }

    m_local_map_kdtree = std::make_unique<util::KdTree>();

    m_local_map_kdtree->setInputCloud(m_local_map);
    m_local_map_kdtree_built = true;
    

}

void LidarFrame::clear_local_map_kdtree() {
    if (m_local_map_kdtree) {
        m_local_map_kdtree.reset();
        m_local_map_kdtree_built = false;
        spdlog::debug("[LidarFrame] Cleared KdTree for frame {}", m_frame_id);
    }
}

void LidarFrame::clear_local_map() {
    if (m_local_map) {
        m_local_map->clear();
        m_local_map.reset();
        spdlog::debug("[LidarFrame] Cleared local map for frame {}", m_frame_id);
    }
}

void LidarFrame::clear_heavy_data_for_old_keyframe() {
    // Keep: m_feature_cloud, m_pose, m_relative_pose, m_keyframe_id
    // Clear: everything else
    
    m_raw_cloud.reset();
    m_processed_cloud.reset();
    m_feature_cloud_global.reset();
    m_local_map.reset();
    
    m_kdtree.reset();
    m_kdtree_built = false;
    m_local_map_kdtree.reset();
    m_local_map_kdtree_built = false;
    
    m_correspondences.clear();
    m_correspondences.shrink_to_fit();
    
    spdlog::debug("[LidarFrame] Cleared heavy data for old keyframe {}", m_keyframe_id);
}

void LidarFrame::clear_non_keyframe_data() {
    // Keep: m_pose, m_relative_pose, m_frame_id, m_timestamp
    // Clear: all clouds and search structures
    
    m_raw_cloud.reset();
    m_processed_cloud.reset();
    m_feature_cloud.reset();
    m_feature_cloud_global.reset();
    m_local_map.reset();
    
    m_kdtree.reset();
    m_kdtree_built = false;
    m_local_map_kdtree.reset();
    m_local_map_kdtree_built = false;
    
    m_correspondences.clear();
    m_correspondences.shrink_to_fit();
    
    spdlog::debug("[LidarFrame] Cleared non-keyframe data for frame {}", m_frame_id);
}

KdTreePtr LidarFrame::get_local_map_kdtree() {
    if (m_local_map_kdtree_built && m_local_map_kdtree) {
        return m_local_map_kdtree;
    }
    return nullptr;
}

} // namespace database
} // namespace lidar_odometry
