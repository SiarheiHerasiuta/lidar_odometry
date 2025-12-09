/**
 * @file      LoopClosureDetector.cpp
 * @brief     Loop closure detection using LiDAR Iris features
 * @author    Seungwon Choi
 * @date      2025-10-10
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "LoopClosureDetector.h"
#include <algorithm>
#include <chrono>

namespace lidar_slam {
namespace processing {

LoopClosureDetector::LoopClosureDetector(const LoopClosureConfig& config)
    : m_config(config) {
    
    // Initialize LiDAR Iris detector with standard parameters
    // Note: range filtering will be done in the point cloud processing stage
    m_iris = std::make_unique<LidarIris>(
        4,    // nscale: number of filter scales
        18,   // minWaveLength: minimum wavelength
        2.1f, // mult: wavelength multiplier
        0.75f,// sigmaOnf: bandwidth parameter
        2     // matchNum: both forward and reverse directions
    );
    
    LOG_INFO("[LoopClosureDetector] Initialized with similarity_threshold={:.3f}, min_gap={}, max_distance={:.1f}m",
                 m_config.similarity_threshold, m_config.min_keyframe_gap, m_config.max_search_distance);
}

LoopClosureDetector::~LoopClosureDetector() {
    LOG_INFO("[LoopClosureDetector] Statistics: {} queries, {} candidates found",
                 m_total_queries, m_total_candidates);
}

bool LoopClosureDetector::add_keyframe(std::shared_ptr<database::LidarFrame> keyframe) {
    if (!keyframe) {
        LOG_WARN("[LoopClosureDetector] Null keyframe provided");
        return false;
    }
    
    // Convert point cloud immediately (before it gets cleared) and store data
    auto simple_cloud = convert_to_simple_cloud(keyframe);
    if (simple_cloud.empty()) {
        LOG_WARN("[LoopClosureDetector] Empty point cloud for keyframe {}", keyframe->get_keyframe_id());
        return false;
    }
    
    PendingKeyframeData data;
    data.cloud = std::move(simple_cloud);
    data.keyframe_id = keyframe->get_keyframe_id();
    data.position = keyframe->get_pose().Translation();
    
    {
        std::lock_guard<std::mutex> lock(m_pending_mutex);
        m_pending_keyframes.push_back(std::move(data));
    }
    
    if (m_config.enable_debug_output) {
        LOG_DEBUG("[LoopClosureDetector] Queued keyframe {} for lazy feature extraction (pending: {})",
                     keyframe->get_keyframe_id(), m_pending_keyframes.size());
    }
    
    return true;
}

std::vector<LoopCandidate> LoopClosureDetector::detect_loop_closures(
    std::shared_ptr<database::LidarFrame> current_keyframe) {
    
    std::vector<LoopCandidate> candidates;
    
    if (!m_config.enable_loop_detection) {
        return candidates;
    }
    
    if (!current_keyframe) {
        LOG_WARN("[LoopClosureDetector] Null current keyframe provided");
        return candidates;
    }
    
    m_total_queries++;
    
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Process pending keyframes first (lazy feature extraction)
        std::vector<PendingKeyframeData> pending_copy;
        {
            std::lock_guard<std::mutex> lock(m_pending_mutex);
            pending_copy = std::move(m_pending_keyframes);
            m_pending_keyframes.clear();
        }
        
        for (const auto& pending_data : pending_copy) {
            if (!pending_data.cloud.empty()) {
                auto feature = extract_iris_feature(pending_data.cloud);
                m_feature_database.push_back(feature);
                m_keyframe_ids.push_back(pending_data.keyframe_id);
                m_keyframe_positions.push_back(pending_data.position);
            }
        }
        
        // Convert current keyframe to SimplePointCloud format
        auto simple_cloud = convert_to_simple_cloud(current_keyframe);
        
        if (simple_cloud.empty()) {
            LOG_WARN("[LoopClosureDetector] Empty point cloud for current keyframe {}", 
                        current_keyframe->get_keyframe_id());
            return candidates;
        }
        
        // Extract feature for current keyframe
        auto current_feature = extract_iris_feature(simple_cloud);
        
        // Search for loop closure candidates
        size_t current_id = current_keyframe->get_keyframe_id();  // Use keyframe ID, not frame ID
        Eigen::Vector3f current_position = current_keyframe->get_pose().Translation();
        float min_similarity = 999.0f;
        std::vector<std::pair<float, size_t>> similarity_scores;
        
        for (size_t i = 0; i < m_feature_database.size(); ++i) {
            size_t candidate_id = m_keyframe_ids[i];
            

            // Check minimum keyframe gap
            if (static_cast<int>(current_id) - static_cast<int>(candidate_id) < m_config.min_keyframe_gap) {
                continue;
            }
            
            // Check distance constraint
            const Eigen::Vector3f& candidate_position = m_keyframe_positions[i];
            float distance = (current_position - candidate_position).norm();
            if (distance > m_config.max_search_distance) {

                continue;
            }
            
            // Compare features
            int bias = 0;
            float similarity = m_iris->Compare(current_feature, m_feature_database[i], &bias);


            
            similarity_scores.push_back({similarity, i});
            min_similarity = std::min(min_similarity, similarity);
        }
        
        // Sort by similarity (lower is better)
        std::sort(similarity_scores.begin(), similarity_scores.end());
        
        // Select only the best candidate that meets threshold
        for (const auto& score_pair : similarity_scores) {
            float similarity = score_pair.first;
            size_t db_index = score_pair.second;
            
            if (similarity > m_config.similarity_threshold) {
                break; // No valid candidates
            }
            
            // Get rotational bias
            int bias = 0;
            m_iris->Compare(current_feature, m_feature_database[db_index], &bias);
            
            LoopCandidate candidate(current_id, m_keyframe_ids[db_index], similarity, bias);
            candidates.push_back(candidate);
            break; // Only take the best candidate
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        
        m_total_candidates += candidates.size();
        
        if (!candidates.empty()) {
            LOG_DEBUG("[LoopClosureDetector] Found {} loop candidates for keyframe {} (search time: {}ms)",
                        candidates.size(), current_id, duration);
            
            for (const auto& candidate : candidates) {
                LOG_DEBUG("  -> Candidate: {} <-> {} (distance: {:.4f}, bias: {})",
                           candidate.query_keyframe_id, candidate.match_keyframe_id,
                           candidate.similarity_score, candidate.bias);
            }
        } else if (m_config.enable_debug_output) {
            LOG_DEBUG("[LoopClosureDetector] No loop candidates found for keyframe {} (min_distance: {:.4f}, search time: {}ms)",
                         current_id, min_similarity, duration);
        }
        
    } catch (const std::exception& e) {
        LOG_ERROR("[LoopClosureDetector] Exception detecting loops for keyframe {}: {}", 
                     current_keyframe->get_keyframe_id(), e.what());
    }
    
    return candidates;
}

void LoopClosureDetector::update_config(const LoopClosureConfig& config) {
    m_config = config;
    LOG_DEBUG("[LoopClosureDetector] Configuration updated: threshold={:.3f}, min_gap={}",
                 m_config.similarity_threshold, m_config.min_keyframe_gap);
}

void LoopClosureDetector::clear() {
    m_feature_database.clear();
    m_keyframe_ids.clear();
    m_total_queries = 0;
    m_total_candidates = 0;
    LOG_INFO("[LoopClosureDetector] Database cleared");
}

SimplePointCloud LoopClosureDetector::convert_to_simple_cloud(
    std::shared_ptr<database::LidarFrame> lidar_frame) {
    
    SimplePointCloud simple_cloud;
    
    // Use feature cloud in LOCAL (sensor) coordinates for LiDAR Iris
    // LiDAR Iris requires sensor-centric point cloud for consistent BEV representation
    auto feature_cloud = lidar_frame->get_feature_cloud();
    if (!feature_cloud || feature_cloud->empty()) {
        LOG_WARN("[LoopClosureDetector] No feature cloud available for keyframe {}", 
                    lidar_frame->get_keyframe_id());
        return simple_cloud;
    }
    
    simple_cloud.reserve(feature_cloud->size());
    for (const auto& point : *feature_cloud) {
        simple_cloud.emplace_back(point.x, point.y, point.z);
    }
    
    return simple_cloud;
}

LidarIris::FeatureDesc LoopClosureDetector::extract_iris_feature(
    const SimplePointCloud& point_cloud) {
    
    // Generate LiDAR Iris image directly from SimplePointCloud
    cv::Mat1b iris_image = LidarIris::GetIris(point_cloud);
    
    // Extract feature descriptor
    LidarIris::FeatureDesc feature = m_iris->GetFeature(iris_image);
    
    return feature;
}

} // namespace processing
} // namespace lidar_slam