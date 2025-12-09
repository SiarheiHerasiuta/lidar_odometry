/**
 * @file      FastVoxelGrid.h
 * @brief     Fast voxel grid filter using Morton code and Robin Hood hashing
 * @author    Seungwon Choi
 * @date      2025-12-08
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#pragma once

#include "../util/TypeUtils.h"
#include "../../thirdparty/unordered_dense/unordered_dense.h"

#include <cstdint>
#include <cmath>

namespace lidar_odometry {
namespace map {

/**
 * @brief Fast voxel grid filter using Z-order Morton code and Robin Hood hashing
 * 
 * Key optimizations:
 * - Stride-based point skipping for initial downsampling
 * - Morton code (Z-order curve) for spatial locality
 * - Robin Hood hashing (unordered_dense) for cache-friendly O(1) lookup
 * - Single-pass filtering with centroid accumulation
 */
class FastVoxelGrid {
public:
    /**
     * @brief Constructor
     * @param voxel_size Size of each voxel cube
     */
    explicit FastVoxelGrid(float voxel_size = 0.5f)
        : m_voxel_size(voxel_size)
        , m_inv_voxel_size(1.0f / voxel_size) {
    }
    
    /**
     * @brief Set voxel size
     * @param voxel_size Size of each voxel cube
     */
    void setVoxelSize(float voxel_size) {
        m_voxel_size = voxel_size;
        m_inv_voxel_size = 1.0f / voxel_size;
    }
    
    /**
     * @brief Get current voxel size
     * @return Voxel size
     */
    float getVoxelSize() const { return m_voxel_size; }
    
    /**
     * @brief Filter point cloud with stride and voxel grid
     * @param input Input point cloud
     * @param output Output filtered point cloud (cleared and filled)
     * @param stride Process every N-th point (1 = all points)
     */
    void filter(const util::PointCloud& input, util::PointCloud& output, int stride = 1) {
        output.clear();
        
        if (input.empty() || stride < 1) {
            return;
        }
        
        // Clear voxel map for new filtering
        m_voxel_map.clear();
        
        // Reserve approximate size
        size_t estimated_size = input.size() / (stride * 8);  // Rough estimate
        m_voxel_map.reserve(estimated_size);
        
        // First pass: accumulate points into voxels with stride
        for (size_t i = 0; i < input.size(); i += stride) {
            const auto& pt = input[i];
            
            // Skip invalid points
            if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
                continue;
            }
            
            // Compute Morton code key
            uint64_t key = computeMortonKey(pt.x, pt.y, pt.z);
            
            // Accumulate into voxel
            auto& voxel = m_voxel_map[key];
            voxel.sum_x += pt.x;
            voxel.sum_y += pt.y;
            voxel.sum_z += pt.z;
            voxel.count++;
        }
        
        // Second pass: extract centroids
        output.reserve(m_voxel_map.size());
        
        for (const auto& [key, voxel] : m_voxel_map) {
            util::PointType pt;
            float inv_count = 1.0f / static_cast<float>(voxel.count);
            pt.x = voxel.sum_x * inv_count;
            pt.y = voxel.sum_y * inv_count;
            pt.z = voxel.sum_z * inv_count;
            output.push_back(pt);
        }
    }
    
    /**
     * @brief Get number of voxels after last filtering
     * @return Number of voxels
     */
    size_t getVoxelCount() const { return m_voxel_map.size(); }

private:
    /**
     * @brief Voxel data for centroid calculation
     */
    struct VoxelData {
        float sum_x = 0.0f;
        float sum_y = 0.0f;
        float sum_z = 0.0f;
        uint32_t count = 0;
    };
    
    /**
     * @brief Expand 21-bit integer for Morton code interleaving
     * @param v Input value (21 bits used)
     * @return Expanded 63-bit value with gaps for interleaving
     */
    static inline uint64_t expandBits(uint64_t v) {
        // Expand 21 bits to 63 bits with 2-bit gaps
        v = v & 0x1FFFFF;  // Keep only 21 bits
        v = (v | (v << 32)) & 0x1F00000000FFFF;
        v = (v | (v << 16)) & 0x1F0000FF0000FF;
        v = (v | (v << 8))  & 0x100F00F00F00F00F;
        v = (v | (v << 4))  & 0x10C30C30C30C30C3;
        v = (v | (v << 2))  & 0x1249249249249249;
        return v;
    }
    
    /**
     * @brief Compute Morton code (Z-order) key from 3D coordinates
     * @param x X coordinate
     * @param y Y coordinate
     * @param z Z coordinate
     * @return 64-bit Morton code key
     */
    inline uint64_t computeMortonKey(float x, float y, float z) const {
        // Convert to voxel indices (with offset to handle negative coordinates)
        // Using 21 bits per dimension, we can represent ±1M voxels
        constexpr int64_t OFFSET = (1 << 20);  // 2^20 offset for negative handling
        
        int64_t ix = static_cast<int64_t>(std::floor(x * m_inv_voxel_size)) + OFFSET;
        int64_t iy = static_cast<int64_t>(std::floor(y * m_inv_voxel_size)) + OFFSET;
        int64_t iz = static_cast<int64_t>(std::floor(z * m_inv_voxel_size)) + OFFSET;
        
        // Clamp to valid range (0 to 2^21 - 1)
        ix = std::max<int64_t>(0, std::min<int64_t>(ix, (1 << 21) - 1));
        iy = std::max<int64_t>(0, std::min<int64_t>(iy, (1 << 21) - 1));
        iz = std::max<int64_t>(0, std::min<int64_t>(iz, (1 << 21) - 1));
        
        // Interleave bits: z gets MSB, then y, then x
        return expandBits(static_cast<uint64_t>(ix)) |
               (expandBits(static_cast<uint64_t>(iy)) << 1) |
               (expandBits(static_cast<uint64_t>(iz)) << 2);
    }
    
    float m_voxel_size;
    float m_inv_voxel_size;
    
    // Robin Hood hash map for O(1) average lookup
    ankerl::unordered_dense::map<uint64_t, VoxelData> m_voxel_map;
};

} // namespace map
} // namespace lidar_odometry
