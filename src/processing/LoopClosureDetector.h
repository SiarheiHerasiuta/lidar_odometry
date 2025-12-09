/**
 * @file      LoopClosureDetector.h
 * @brief     Loop closure detection using LiDAR Iris features
 * @author    Seungwon Choi
 * @date      2025-10-10
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#pragma once

#include "../database/LidarFrame.h"
#include "../util/MathUtils.h"
#include "../util/PointCloudUtils.h"
#include "../util/PointCloudUtils.h"
#include "../../thirdparty/LidarIris/LidarIris.h"

#include <memory>
#include <vector>
#include <mutex>
#include "util/LogUtils.h"

namespace lidar_slam {
namespace processing {

/**
 * @brief Loop closure candidate structure
 */
struct LoopCandidate {
    size_t query_keyframe_id;      ///< Current keyframe ID
    size_t match_keyframe_id;      ///< Matched keyframe ID
    float similarity_score;        ///< LiDAR Iris similarity score (lower is better)
    int bias;                      ///< Rotational bias from LiDAR Iris
    bool is_valid;                 ///< Whether this candidate passed validation
    
    LoopCandidate() 
        : query_keyframe_id(0), match_keyframe_id(0), 
          similarity_score(999.0f), bias(0), is_valid(false) {}
    
    LoopCandidate(size_t query_id, size_t match_id, float score, int rot_bias)
        : query_keyframe_id(query_id), match_keyframe_id(match_id),
          similarity_score(score), bias(rot_bias), is_valid(true) {}
};

/**
 * @brief Loop closure detection configuration
 */
struct LoopClosureConfig {
    bool enable_loop_detection = true;       ///< Enable/disable loop detection
    float similarity_threshold = 0.3f;       ///< LiDAR Iris similarity threshold
    int min_keyframe_gap = 50;               ///< Minimum gap between keyframes for loop closure
    float max_search_distance = 10.0f;      ///< Maximum distance (meters) to search for loop candidates
    bool enable_debug_output = false;       ///< Enable debug logging
    
    // LiDAR Iris parameters will be automatically calculated from point cloud max_range
};

/**
 * @brief Loop closure detector using LiDAR Iris features
 */
class LoopClosureDetector {
public:
    /**
     * @brief Constructor
     * @param config Loop closure detection configuration
     */
    explicit LoopClosureDetector(const LoopClosureConfig& config = LoopClosureConfig());
    
    /**
     * @brief Destructor
     */
    ~LoopClosureDetector();
    
    /**
     * @brief Add keyframe and extract LiDAR Iris feature
     * @param keyframe New keyframe to add
     * @return true if feature extraction successful
     */
    bool add_keyframe(std::shared_ptr<database::LidarFrame> keyframe);
    
    /**
     * @brief Detect loop closure candidates for current keyframe
     * @param current_keyframe Current keyframe to query
     * @return Vector of loop closure candidates
     */
    std::vector<LoopCandidate> detect_loop_closures(std::shared_ptr<database::LidarFrame> current_keyframe);
    
    /**
     * @brief Get number of stored keyframes
     * @return Number of keyframes in database
     */
    size_t get_keyframe_count() const { return m_keyframe_ids.size(); }
    
    /**
     * @brief Update configuration
     * @param config New configuration
     */
    void update_config(const LoopClosureConfig& config);
    
    /**
     * @brief Get current configuration
     * @return Current configuration
     */
    const LoopClosureConfig& get_config() const { return m_config; }
    
    /**
     * @brief Clear all stored features and keyframes
     */
    void clear();

private:
    /**
     * @brief Convert LidarFrame point cloud to SimplePointCloud format for LiDAR Iris
     * @param lidar_frame Input LiDAR frame
     * @return SimplePointCloud for LiDAR Iris
     */
    SimplePointCloud convert_to_simple_cloud(std::shared_ptr<database::LidarFrame> lidar_frame);
    
    /**
     * @brief Extract LiDAR Iris feature from SimplePointCloud
     * @param point_cloud Input point cloud
     * @return LiDAR Iris feature descriptor
     */
    LidarIris::FeatureDesc extract_iris_feature(const SimplePointCloud& point_cloud);
    
    LoopClosureConfig m_config;                                    ///< Configuration
    std::unique_ptr<LidarIris> m_iris;                            ///< LiDAR Iris detector
    std::vector<LidarIris::FeatureDesc> m_feature_database;       ///< Feature database
    std::vector<size_t> m_keyframe_ids;                           ///< Keyframe IDs corresponding to features
    std::vector<Eigen::Vector3f> m_keyframe_positions;            ///< Keyframe positions for distance filtering
    
    // Pending keyframes for lazy feature extraction (store data, not references)
    struct PendingKeyframeData {
        SimplePointCloud cloud;
        size_t keyframe_id;
        Eigen::Vector3f position;
    };
    std::vector<PendingKeyframeData> m_pending_keyframes;  ///< Keyframe data pending feature extraction
    mutable std::mutex m_pending_mutex;                    ///< Mutex for thread-safe pending keyframe access
    
    // Statistics
    size_t m_total_queries = 0;                                   ///< Total number of queries
    size_t m_total_candidates = 0;                                ///< Total number of candidates found
};

} // namespace processing
} // namespace lidar_slam