/**
 * @file      VoxelMap.h
 * @brief     2-Level hierarchical voxel map with precomputed surfels
 * @author    Seungwon Choi
 * @date      2025-12-08
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 * 
 * @details
 * 2-Level Hierarchy:
 * - L0: Leaf voxels (voxel_size) - stores centroid only (no raw points)
 * - L1: Parent voxels (hierarchy_factor × voxel_size) - stores precomputed surfels
 * 
 * Surfel computation:
 * - Collects L0 centroids within L1 voxel
 * - Fits plane using PCA (SVD of covariance matrix)
 * - Caches normal, centroid, planarity score
 * - Updates only when L0 child count changes (incremental)
 */

#ifndef LIDAR_ODOMETRY_VOXEL_MAP_H
#define LIDAR_ODOMETRY_VOXEL_MAP_H

#include "../util/PointCloudUtils.h"
#include "../../thirdparty/unordered_dense/unordered_dense.h"
#include <vector>
#include <mutex>
#include <Eigen/Dense>

namespace lidar_odometry {
namespace map {

/**
 * @brief Voxel key for spatial hashing
 */
struct VoxelKey {
    int x, y, z;
    
    VoxelKey() : x(0), y(0), z(0) {}
    VoxelKey(int x_, int y_, int z_) : x(x_), y(y_), z(z_) {}
    
    bool operator==(const VoxelKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

/**
 * @brief Z-order (Morton code) hash function for VoxelKey
 */
struct VoxelKeyHash {
private:
    static inline uint64_t ExpandBits(int32_t v) {
        uint64_t x = static_cast<uint64_t>(v + (1 << 20)) & 0x1fffff;
        x = (x | (x << 32)) & 0x1f00000000ffffULL;
        x = (x | (x << 16)) & 0x1f0000ff0000ffULL;
        x = (x | (x << 8))  & 0x100f00f00f00f00fULL;
        x = (x | (x << 4))  & 0x10c30c30c30c30c3ULL;
        x = (x | (x << 2))  & 0x1249249249249249ULL;
        return x;
    }
    
public:
    std::size_t operator()(const VoxelKey& key) const {
        uint64_t morton = ExpandBits(key.x) | (ExpandBits(key.y) << 1) | (ExpandBits(key.z) << 2);
        return static_cast<std::size_t>(morton);
    }
};

/**
 * @brief 2-Level hierarchical voxel map with precomputed surfels
 */
class VoxelMap {
public:
    using PointCloud = lidar_odometry::util::PointCloud;
    using PointCloudPtr = lidar_odometry::util::PointCloudPtr;
    using PointCloudConstPtr = lidar_odometry::util::PointCloudConstPtr;
    using Point3D = lidar_odometry::util::Point3D;

    explicit VoxelMap(float voxel_size = 0.5f);
    
    void SetVoxelSize(float size);
    void SetMaxHitCount(int max_count) { m_max_hit_count = max_count; }
    void SetInitHitCount(int count) { m_init_hit_count = count; }
    void SetHierarchyFactor(int factor);
    void SetPlanarityThreshold(float threshold) { m_planarity_threshold = threshold; }
    
    float GetVoxelSize() const { return m_voxel_size; }
    int GetHierarchyFactor() const { return m_hierarchy_factor; }
    size_t GetVoxelCount() const { return m_voxels_L0.size(); }
    size_t GetL1VoxelCount() const { return m_voxels_L1.size(); }
    bool empty() const { return m_voxels_L0.empty(); }
    
    /**
     * @brief Get number of L1 voxels with valid surfels
     */
    size_t GetSurfelCount() const {
        size_t count = 0;
        for (const auto& [key, node] : m_voxels_L1) {
            if (node.has_surfel) count++;
        }
        return count;
    }
    
    /**
     * @brief Clear all voxels
     */
    void Clear();
    
    /**
     * @brief Update voxel map with new point cloud
     * @param new_cloud New points in world coordinates
     * @param sensor_position Current sensor position
     * @param max_distance Maximum distance to keep voxels
     * @param is_keyframe Whether this is a keyframe (triggers surfel update)
     */
    void UpdateVoxelMap(const PointCloudConstPtr& new_cloud,
                        const Eigen::Vector3d& sensor_position,
                        double max_distance,
                        bool is_keyframe);
    
    /**
     * @brief Apply rigid transformation and rehash all voxels (for PGO correction)
     * @param T_correction Correction transformation matrix
     */
    void ApplyTransformAndRehash(const Eigen::Matrix4f& T_correction);
    
    /**
     * @brief Get surfel at a query point (O(1) lookup)
     * @param point Query point in world coordinates
     * @param normal Output: surfel normal
     * @param centroid Output: surfel centroid
     * @return True if valid surfel found
     */
    bool GetSurfelAtPoint(const Eigen::Vector3f& point,
                          Eigen::Vector3f& normal,
                          Eigen::Vector3f& centroid) const;
    
    /**
     * @brief Get point cloud of all L0 centroids (for visualization)
     */
    PointCloudPtr GetPointCloud() const;
    
    /**
     * @brief Get all L1 surfels for visualization
     * @return Vector of (centroid, normal, planarity_score) tuples
     */
    std::vector<std::tuple<Eigen::Vector3f, Eigen::Vector3f, float>> GetL1Surfels() const;
    
private:
    VoxelKey PointToVoxelKey(const Eigen::Vector3f& point, int level = 0) const;
    VoxelKey GetParentKey(const VoxelKey& key) const;
    void AddPoint(const Eigen::Vector3f& point);
    void RegisterToParent(const VoxelKey& key_L0);
    void UnregisterFromParent(const VoxelKey& key_L0);
    Eigen::Vector3f VoxelKeyToCenter(const VoxelKey& key) const;
    void RecomputeAllSurfels();
    
    // Parameters
    float m_voxel_size;
    int m_max_hit_count = 10;
    int m_init_hit_count = 1;
    int m_hierarchy_factor = 3;  // L1 = 3×3×3 L0 voxels
    float m_planarity_threshold = 0.1f;  // Default: 0.1 (less strict)
    
    // Level 0: Leaf voxels (centroid only, no raw points)
    struct VoxelNode_L0 {
        Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
        int hit_count = 1;
        int point_count = 0;
        
        VoxelNode_L0() = default;
    };
    ankerl::unordered_dense::map<VoxelKey, VoxelNode_L0, VoxelKeyHash> m_voxels_L0;
    
    // Level 1: Parent voxels with precomputed surfels
    struct VoxelNode_L1 {
        ankerl::unordered_dense::set<VoxelKey, VoxelKeyHash> occupied_children;
        
        // Surfel data
        bool has_surfel = false;
        Eigen::Vector3f surfel_normal = Eigen::Vector3f::Zero();
        Eigen::Vector3f surfel_centroid = Eigen::Vector3f::Zero();
        float planarity_score = 1.0f;
        int last_child_count = 0;
        
        VoxelNode_L1() = default;
    };
    ankerl::unordered_dense::map<VoxelKey, VoxelNode_L1, VoxelKeyHash> m_voxels_L1;
    
    mutable std::recursive_mutex m_mutex;
};

} // namespace map
} // namespace lidar_odometry

#endif // LIDAR_ODOMETRY_VOXEL_MAP_H
