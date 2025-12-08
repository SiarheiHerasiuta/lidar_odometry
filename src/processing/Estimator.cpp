/**
 * @file      Estimator.cpp
 * @brief     Implementation of LiDAR odometry estimator.
 * @author    Seungwon Choi
 * @date      2025-09-24
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "Estimator.h"
#include "../util/MathUtils.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>

namespace lidar_odometry {
namespace processing {

Estimator::Estimator(const util::SystemConfig& config)
    : m_config(config)
    , m_initialized(false)
    , m_T_wl_current()
    , m_velocity()
    , m_next_keyframe_id(0)
    , m_debug_pre_icp_cloud(new PointCloud())
    , m_debug_post_icp_cloud(new PointCloud())
    , m_last_successful_loop_keyframe_id(-1)  // Initialize to -1 (no successful loop closure yet)
    , m_last_keyframe_pose()
    , m_total_optimization_iterations(0)
    , m_total_optimization_time_ms(0.0)
    , m_optimization_call_count(0)
{
    // Initialize pose graph optimizer (Ceres only)
    m_pose_graph_optimizer = std::make_shared<optimization::PoseGraphOptimizer>();
    
    // Create AdaptiveMEstimator with PKO configuration only
    m_adaptive_estimator = std::make_shared<optimization::AdaptiveMEstimator>(
        config.use_adaptive_m_estimator,
        config.loss_type,
        config.min_scale_factor,
        config.max_scale_factor,
        config.num_alpha_segments,
        config.truncated_threshold,
        config.gmm_components,
        config.gmm_sample_size,
        config.pko_kernel_type
    );
    
    // Initialize IterativeClosestPointOptimizer with AdaptiveMEstimator and configuration
    ICPConfig dual_frame_config;
    dual_frame_config.max_iterations = config.max_iterations;
    dual_frame_config.translation_tolerance = config.translation_threshold;
    dual_frame_config.rotation_tolerance = config.rotation_threshold;
    dual_frame_config.max_correspondence_distance = config.max_correspondence_distance;
    dual_frame_config.outlier_rejection_ratio = 0.9;
    dual_frame_config.use_robust_loss = true;
    dual_frame_config.robust_loss_delta = 0.1;
    
    m_icp_optimizer = std::make_shared<optimization::IterativeClosestPointOptimizer>(dual_frame_config, m_adaptive_estimator);
    
    // Initialize fast voxel grid filter (Morton code + Robin Hood hashing)
    m_fast_voxel_grid = std::make_unique<map::FastVoxelGrid>(config.voxel_size);
    
    // Initialize legacy voxel filter for map downsampling
    m_voxel_filter = std::make_unique<util::VoxelGrid>();
    m_voxel_filter->setLeafSize(config.voxel_size);
    
    // Initialize feature extractor (kept for compatibility, may be removed later)
    FeatureExtractorConfig feature_config;
    feature_config.voxel_size = config.feature_voxel_size;
    feature_config.max_neighbor_distance = config.max_neighbor_distance;
    m_feature_extractor = std::make_unique<FeatureExtractor>(feature_config);
    
    spdlog::info("[Estimator] Using FastVoxelGrid with stride={}, voxel_size={}", 
                 config.point_stride, config.voxel_size);
    
    // Initialize loop closure detector
    LoopClosureConfig loop_config;
    loop_config.enable_loop_detection = config.loop_enable_loop_detection;
    loop_config.similarity_threshold = config.loop_similarity_threshold;
    loop_config.min_keyframe_gap = config.loop_min_keyframe_gap;
    loop_config.max_search_distance = config.loop_max_search_distance;
    loop_config.enable_debug_output = config.loop_enable_debug_output;
    // Iris parameters are now automatically calculated
    m_loop_detector = std::make_unique<LoopClosureDetector>(loop_config);
    
    // Start background thread for loop detection and PGO
    m_thread_running = true;
    m_loop_pgo_thread = std::thread(&Estimator::loop_pgo_thread_function, this);
    spdlog::info("[Estimator] Background loop+PGO thread started");
}

Estimator::~Estimator() {
    // Stop background thread
    spdlog::info("[Estimator] Stopping background loop+PGO thread...");
    m_thread_running = false;
    m_query_cv.notify_all();  // Wake up thread if waiting
    
    if (m_loop_pgo_thread.joinable()) {
        m_loop_pgo_thread.join();
        spdlog::info("[Estimator] Background thread stopped successfully");
    }
}

bool Estimator::process_frame(std::shared_ptr<database::LidarFrame> current_frame) {
    auto start_time = std::chrono::high_resolution_clock::now();
    TimingStats timing;

    if (!current_frame || !current_frame->get_raw_cloud()) {
        spdlog::warn("[Estimator] Invalid frame or point cloud");
        return false;
    }
    
    // Step 0: Check and apply pending PGO result from background thread (non-blocking)
    apply_pending_pgo_result_if_available();
    
    // Step 1: Preprocess frame (downsample + feature extraction)
    auto preprocess_start = std::chrono::high_resolution_clock::now();
    if (!preprocess_frame(current_frame)) {
        spdlog::error("[Estimator] Frame preprocessing failed");
        return false;
    }
    auto preprocess_end = std::chrono::high_resolution_clock::now();
    timing.preprocessing_ms = std::chrono::duration<double, std::milli>(preprocess_end - preprocess_start).count();
    
    if (!m_initialized) {
        initialize_first_frame(current_frame);
        return true;
    }

    // Get feature cloud from frame
    auto feature_cloud = current_frame->get_feature_cloud();

    // Step 3: Use last keyframe for optimization
    if (!m_last_keyframe) {
        spdlog::warn("[Estimator] No keyframe available, using velocity model only");
        return true;
    }
    
    // Step 4: optimization::IterativeClosestPointOptimizer between current frame and last keyframe
    auto icp_start = std::chrono::high_resolution_clock::now();
    // Calculate initial guess from velocity model: transform from keyframe to current velocity estimate
    SE3f T_keyframe_current_guess = m_previous_frame->get_pose() * m_velocity;
    SE3f T_keyframe_current = estimate_motion_dual_frame(current_frame, m_last_keyframe, T_keyframe_current_guess); 
    auto icp_end = std::chrono::high_resolution_clock::now();
    timing.icp_ms = std::chrono::duration<double, std::milli>(icp_end - icp_start).count();
    
    // Convert result to world coordinate (keyframe pose is already in world coordinates)
    SE3f optimized_pose = T_keyframe_current;
    
    // Store post-optimization cloud in world coordinates for visualization
    auto map_update_start = std::chrono::high_resolution_clock::now();
    
    auto transform_start = std::chrono::high_resolution_clock::now();
    PointCloudPtr post_opt_cloud_world(new PointCloud());
    Eigen::Matrix4f T_wl_final = optimized_pose.matrix();
    util::transform_point_cloud(feature_cloud, post_opt_cloud_world, T_wl_final);
    auto transform_end = std::chrono::high_resolution_clock::now();
    double transform_ms = std::chrono::duration<double, std::milli>(transform_end - transform_start).count();

    current_frame->set_feature_cloud_global(post_opt_cloud_world); // Cache world coordinate features
    
    m_T_wl_current = optimized_pose;
    
    // Step 5: Update velocity model
    m_velocity = m_previous_frame->get_pose().inverse() * m_T_wl_current;


    // Update frame pose and trajectory
    current_frame->set_pose(m_T_wl_current);
    m_trajectory.push_back(m_T_wl_current);
    
    // Set previous keyframe reference for dynamic pose calculation
    auto pose_calc_start = std::chrono::high_resolution_clock::now();
    if (m_last_keyframe) {
        current_frame->set_previous_keyframe(m_last_keyframe);
        // Calculate and store relative pose from keyframe to current frame
        SE3f relative_pose = m_last_keyframe->get_stored_pose().inverse() * m_T_wl_current;
        current_frame->set_relative_pose(relative_pose);
    }
    auto pose_calc_end = std::chrono::high_resolution_clock::now();
    double pose_calc_ms = std::chrono::duration<double, std::milli>(pose_calc_end - pose_calc_start).count();
    
    // Step 6: Check for keyframe creation
    auto keyframe_start = std::chrono::high_resolution_clock::now();
    bool keyframe_created = false;
    if (should_create_keyframe(m_T_wl_current)) {
        // Transform feature cloud to world coordinates for keyframe storage
        create_keyframe(current_frame);
        keyframe_created = true;
    }
    auto keyframe_end = std::chrono::high_resolution_clock::now();
    double keyframe_ms = std::chrono::duration<double, std::milli>(keyframe_end - keyframe_start).count();
    
    auto map_update_end = std::chrono::high_resolution_clock::now();
    timing.map_update_ms = std::chrono::duration<double, std::milli>(map_update_end - map_update_start).count();
    
    // Print detailed map update timing
    spdlog::debug("[MapUpdate] Total: {:.2f}ms | Transform: {:.2f}ms | PoseCalc: {:.2f}ms | Keyframe({}): {:.2f}ms",
                  timing.map_update_ms, transform_ms, pose_calc_ms, keyframe_created ? "YES" : "NO", keyframe_ms);
    
    // Update for next iteration - clean up old frame first
    m_old_frame = m_previous_frame;  // Save old frame
    m_previous_frame = current_frame;  // Update to new frame
    
    // Clean up old frame (non-keyframe) - keep only pose info
    if (m_old_frame && !m_old_frame->is_keyframe()) {
        m_old_frame->clear_non_keyframe_data();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    timing.total_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    // Store timing and print statistics every 100 frames
    m_timing_history.push_back(timing);
    m_frame_count++;
    
    if (m_frame_count % 100 == 0) {
        print_timing_statistics();
    }
    
    spdlog::debug("[Estimator] Frame {} processed in {:.1f}ms (Preprocess: {:.1f}ms, ICP: {:.1f}ms, MapUpdate: {:.1f}ms)", 
                 m_frame_count, timing.total_ms, timing.preprocessing_ms, timing.icp_ms, timing.map_update_ms);
    
    return true;
}

void Estimator::initialize_first_frame(std::shared_ptr<database::LidarFrame> frame) {
    // Get initial pose from frame (could be set by other sensors)
    m_T_wl_current = frame->get_initial_pose();
    m_velocity = SE3f();      // Identity velocity
    frame->set_pose(m_T_wl_current);
    m_trajectory.push_back(m_T_wl_current);
    
    // First frame has no previous keyframe (it will become the first keyframe)
    frame->set_previous_keyframe(nullptr);
    frame->set_relative_pose(SE3f());  // Identity relative pose

    // Get feature cloud from preprocessed frame
    auto feature_cloud = frame->get_feature_cloud();
    if (!feature_cloud || feature_cloud->empty()) {
        spdlog::error("[Estimator] No feature cloud in first frame");
        m_initialized = true;
        return;
    }

    // Transform features to world coordinates using current pose
    PointCloudPtr feature_cloud_world(new PointCloud());
    Eigen::Matrix4f T_wl = m_T_wl_current.matrix();
    util::transform_point_cloud(feature_cloud, feature_cloud_world, T_wl);
    
    // Set global feature cloud in frame
    frame->set_feature_cloud_global(feature_cloud_world);
    
    // Set as keyframe and initialize local map
    create_keyframe(frame);
    
    m_previous_frame = frame;
    m_last_keyframe_pose = m_T_wl_current;
    
    m_initialized = true;
}

SE3f Estimator::estimate_motion_dual_frame(std::shared_ptr<database::LidarFrame> current_frame,
                                          std::shared_ptr<database::LidarFrame> keyframe,
                                          const SE3f& initial_guess) {
    if (!current_frame || !keyframe) {
        spdlog::warn("[Estimator] Invalid frames for dual frame optimization");
        return initial_guess;
    }
    
    auto current_features = current_frame->get_feature_cloud();
    auto keyframe_local_map = keyframe->get_local_map();
    
    if (!current_features || !keyframe_local_map || current_features->empty() || keyframe_local_map->empty()) {
        spdlog::warn("[Estimator] Invalid feature clouds for dual frame optimization");
        return initial_guess;
    }
    
    // Convert SE3f to Sophus::SE3f for optimization::IterativeClosestPointOptimizer
    Sophus::SE3f initial_transform_sophus(initial_guess.rotationMatrix(), initial_guess.translation());
    Sophus::SE3f optimized_transform_sophus;

    // Debug log for initial guess (only in debug mode)
    spdlog::debug("Check Initial Guess: Translation ({:.3f}, {:.3f}, {:.3f}), Rotation ({:.3f}, {:.3f}, {:.3f})",
                 initial_guess.translation().x(), initial_guess.translation().y(), initial_guess.translation().z(),
                 initial_guess.so3().log().x(), initial_guess.so3().log().y(), initial_guess.so3().log().z());
    
    // Perform dual frame optimization (keyframe as source, current as target)
    bool success = m_icp_optimizer->optimize(
        keyframe,                    // source frame (keyframe)
        current_frame,              // target frame (current)
        initial_transform_sophus,   // initial relative transform
        optimized_transform_sophus  // optimized relative transform (output)
    );
    
    if (!success) {
        spdlog::warn("[Estimator] Dual frame optimization failed, using initial guess");
        return initial_guess;
    }
    
    // Convert back to SE3f
    SE3f T_keyframe_current(optimized_transform_sophus.rotationMatrix(), optimized_transform_sophus.translation());
    
    // Collect optimization statistics
    m_total_optimization_iterations += 10; // TODO: Get actual iterations from optimization::IterativeClosestPointOptimizer
    m_total_optimization_time_ms += 1.0;   // TODO: Get actual time from optimization::IterativeClosestPointOptimizer
    m_optimization_call_count++;
    
    spdlog::debug("[Estimator] Dual frame optimization completed successfully");
    
    return T_keyframe_current;
}

std::shared_ptr<database::LidarFrame> Estimator::select_best_keyframe(const SE3f& current_pose) {
    if (m_keyframes.empty()) {
        return nullptr;
    }
    
    // Use the most recent (latest) keyframe for temporal consistency
    std::shared_ptr<database::LidarFrame> best_keyframe = nullptr;
    
    // Find the most recent keyframe with a valid local map
    for (auto it = m_keyframes.rbegin(); it != m_keyframes.rend(); ++it) {
        if (*it && (*it)->get_local_map() && !(*it)->get_local_map()->empty()) {
            best_keyframe = *it;
            break;  // Use the most recent valid keyframe
        }
    }
    
    if (best_keyframe) {
        Vector3f translation_diff = current_pose.translation() - best_keyframe->get_pose().translation();
        double distance = translation_diff.norm();
        spdlog::debug("[Estimator] Selected most recent keyframe at distance {:.2f}m", distance);
    } else {
        spdlog::debug("[Estimator] No suitable keyframe found");
    }
    
    return best_keyframe;
}

bool Estimator::should_create_keyframe(const SE3f& current_pose) {
   
    if (m_keyframes.empty()) {
        return true;
    }
    
    // Calculate distance and rotation from last keyframe
    Vector3f translation_diff = current_pose.translation() - m_last_keyframe_pose.translation();
    double distance = translation_diff.norm();
    
    Sophus::SO3f rotation_diff = m_last_keyframe_pose.so3().inverse() * current_pose.so3();
    double rotation_angle = rotation_diff.log().norm();

    // Debug log for keyframe check
    spdlog::debug("[Estimator] Keyframe check: Δt={:.2f}m, Δr={:.2f}° (thresholds: {:.2f}m, {:.2f}°)", 
                  distance, rotation_angle * 180.0 / M_PI,
                  m_config.keyframe_distance_threshold, m_config.keyframe_rotation_threshold);
    
    return (distance > m_config.keyframe_distance_threshold || rotation_angle > m_config.keyframe_rotation_threshold);
}

void Estimator::create_keyframe(std::shared_ptr<database::LidarFrame> frame)
{
    auto kf_total_start = std::chrono::high_resolution_clock::now();
    
    // Set keyframe ID
    frame->set_keyframe_id(m_next_keyframe_id++);
    
    // Calculate and store relative pose from previous keyframe
    if (!m_keyframes.empty()) {
        auto previous_keyframe = m_keyframes.back();
        SE3f prev_pose = previous_keyframe->get_pose();
        SE3f curr_pose = frame->get_pose();
        
        // Compute relative pose: T_prev_curr = T_prev^-1 * T_curr
        SE3f relative_pose_raw = prev_pose.inverse() * curr_pose;
        
        // Normalize rotation matrix for numerical stability
        Eigen::Matrix3f rotation_matrix = relative_pose_raw.rotationMatrix();
        Eigen::Matrix3f normalized_rotation = util::MathUtils::normalize_rotation_matrix(rotation_matrix);
        SE3f relative_pose(normalized_rotation, relative_pose_raw.translation());
        
        frame->set_relative_pose(relative_pose);
        
        spdlog::debug("[Estimator] Set relative pose for keyframe {}: t_norm={:.3f}, r_norm={:.3f}°", 
                     frame->get_keyframe_id(), 
                     relative_pose.translation().norm(),
                     relative_pose.so3().log().norm() * 180.0f / M_PI);
        
        // Odometry constraint will be added to PGO later, right before optimization
    } else {
        // First keyframe: set identity relative pose
        frame->set_relative_pose(SE3f());
        spdlog::debug("[Estimator] First keyframe: set identity relative pose");
    }
    
    // Add to keyframes list (thread-safe)
    {
        std::lock_guard<std::mutex> lock(m_keyframes_mutex);
        m_keyframes.push_back(frame);
    }

    // Check if frame has global feature cloud
    auto global_feature_cloud = frame->get_feature_cloud_global();
    if (!global_feature_cloud) {
        spdlog::warn("[Estimator] Frame has no global feature cloud, using local feature cloud");
        // For first frame, transform local features to global (identity transform)
        auto local_feature_cloud = frame->get_feature_cloud();
        if (local_feature_cloud && !local_feature_cloud->empty()) {
            PointCloudPtr global_cloud = std::make_shared<PointCloud>();
            *global_cloud = *local_feature_cloud;  // Copy for first frame
            frame->set_feature_cloud_global(global_cloud);
            global_feature_cloud = global_cloud;
        } else {
            spdlog::error("[Estimator] Frame has no feature clouds at all!");
            return;
        }
    }

    // Build local map by accumulating last keyframe's local map + current global features
    auto accumulate_start = std::chrono::high_resolution_clock::now();
    PointCloudPtr accumulated_map = std::make_shared<util::PointCloud>();
    
    // Add last keyframe's local map if it exists
    if (m_last_keyframe && m_last_keyframe->get_local_map()) {
        *accumulated_map += *m_last_keyframe->get_local_map();
        spdlog::debug("[Estimator] Added {} points from last keyframe's local map", 
                      m_last_keyframe->get_local_map()->size());
    }
    
    // Add current keyframe's global features
    if (global_feature_cloud) {
        *accumulated_map += *global_feature_cloud;
        spdlog::debug("[Estimator] Added {} points from current global features", 
                      global_feature_cloud->size());
    } else {
        spdlog::error("[Estimator] Null global feature cloud!");
        return;
    }
    auto accumulate_end = std::chrono::high_resolution_clock::now();
    double accumulate_ms = std::chrono::duration<double, std::milli>(accumulate_end - accumulate_start).count();
    
    // Downsample the accumulated map
    auto downsample_start = std::chrono::high_resolution_clock::now();
    util::VoxelGrid map_voxel_filter;
    float map_voxel_size = static_cast<float>(m_config.map_voxel_size);
    map_voxel_filter.setLeafSize(map_voxel_size);
    map_voxel_filter.setInputCloud(accumulated_map);
    
    auto downsampled_map = std::make_shared<util::PointCloud>();
    map_voxel_filter.filter(*downsampled_map);
    auto downsample_end = std::chrono::high_resolution_clock::now();
    double downsample_ms = std::chrono::duration<double, std::milli>(downsample_end - downsample_start).count();
    
    spdlog::debug("[Estimator] Downsampled accumulated map: {} -> {} points", 
                  accumulated_map->size(), downsampled_map->size());
    
    // Apply radius-based filtering around current pose (LiDAR circular pattern)
    auto radius_filter_start = std::chrono::high_resolution_clock::now();
    Eigen::Vector3f current_position = frame->get_pose().translation();
    float filter_radius = static_cast<float>(m_config.max_range * 1.2);
    
    auto filtered_local_map = std::make_shared<util::PointCloud>();
    filtered_local_map->reserve(downsampled_map->size());
    
    // Filter points within radius from current position
    for (const auto& point : *downsampled_map) {
        Eigen::Vector3f point_pos(point.x, point.y, point.z);
        float distance = (point_pos - current_position).norm();
        
        if (distance <= filter_radius) {
            filtered_local_map->push_back(point);
        }
    }
    auto radius_filter_end = std::chrono::high_resolution_clock::now();
    double radius_filter_ms = std::chrono::duration<double, std::milli>(radius_filter_end - radius_filter_start).count();
    
    // Store filtered local map in keyframe for optimization::IterativeClosestPointOptimizer
    frame->set_local_map(filtered_local_map);
    
    // Build KdTree for the local map at keyframe creation
    auto kdtree_start = std::chrono::high_resolution_clock::now();
    frame->build_local_map_kdtree();
    auto kdtree_end = std::chrono::high_resolution_clock::now();
    double kdtree_ms = std::chrono::duration<double, std::milli>(kdtree_end - kdtree_start).count();
    
    // Clean up previous last keyframe's heavy data (keep only feature_cloud for visualization)
    if (m_last_keyframe && m_last_keyframe != frame) {
        m_last_keyframe->clear_heavy_data_for_old_keyframe();
        spdlog::debug("[Estimator] Cleared heavy data for previous keyframe {}", m_last_keyframe->get_keyframe_id());
    }
    
    // Update last keyframe reference for optimization
    m_last_keyframe = frame;

    m_last_keyframe_pose = m_last_keyframe->get_pose();
    
    spdlog::debug("[Estimator] Keyframe created: input={} -> local_map={} points", 
                  global_feature_cloud->size(), filtered_local_map->size());
    
    // Add keyframe to loop detector database and query queue for async processing
    if (m_loop_detector && m_config.loop_enable_loop_detection) {
        // Always add keyframe to database
        m_loop_detector->add_keyframe(frame);
        
        // Check cooldown: only add to query queue if enough keyframes have passed
        int current_keyframe_id = frame->get_keyframe_id();
        int keyframes_since_last_loop = current_keyframe_id - m_last_successful_loop_keyframe_id;
        bool allow_detection = (keyframes_since_last_loop >= m_config.loop_min_keyframe_gap);
        
        if (allow_detection) {
            // Add keyframe ID to query queue for background thread processing
            {
                std::lock_guard<std::mutex> lock(m_query_mutex);
                m_loop_query_queue.push_back(current_keyframe_id);
                spdlog::debug("[Estimator] Added KF {} to loop query queue (queue size: {})", 
                             current_keyframe_id, m_loop_query_queue.size());
            }
            m_query_cv.notify_one();  // Wake up background thread
        } else {
            spdlog::debug("[Estimator] Loop detection skipped: only {} keyframes since last loop (need {})",
                         keyframes_since_last_loop, m_config.loop_min_keyframe_gap);
        }
    }
    
    auto kf_total_end = std::chrono::high_resolution_clock::now();
    double kf_total_ms = std::chrono::duration<double, std::milli>(kf_total_end - kf_total_start).count();
    
    // Print detailed keyframe creation timing
    spdlog::info("[Keyframe {}] Total: {:.2f}ms | Accumulate: {:.2f}ms | Downsample: {:.2f}ms | RadiusFilter: {:.2f}ms | KdTree: {:.2f}ms | MapSize: {} pts",
                 frame->get_keyframe_id(), kf_total_ms, accumulate_ms, downsample_ms, radius_filter_ms, kdtree_ms, filtered_local_map->size());
}


void Estimator::update_config(const util::SystemConfig& config) {
    m_config = config;
    
    // Update voxel filter
    m_voxel_filter->setLeafSize(m_config.voxel_size);
}

const util::SystemConfig& Estimator::get_config() const {
    return m_config;
}

PointCloudConstPtr Estimator::get_local_map() const {

    if(m_last_keyframe) {
        return m_last_keyframe->get_local_map();
    }  
    else
    {
        return nullptr;
    }
}


void Estimator::get_debug_clouds(PointCloudConstPtr& pre_icp_cloud, PointCloudConstPtr& post_icp_cloud) const {
    pre_icp_cloud = m_debug_pre_icp_cloud;
    post_icp_cloud = m_debug_post_icp_cloud;
}

bool Estimator::preprocess_frame(std::shared_ptr<database::LidarFrame> frame) {
    auto raw_cloud = frame->get_raw_cloud();
    if (!raw_cloud || raw_cloud->empty()) {
        spdlog::error("[Estimator] Invalid raw cloud");
        return false;
    }
    
    // Step 1: Fast voxel grid downsampling with stride (Morton code + Robin Hood hashing)
    PointCloudPtr downsampled_cloud = std::make_shared<PointCloud>();
    m_fast_voxel_grid->filter(*raw_cloud, *downsampled_cloud, m_config.point_stride);
    
    if (downsampled_cloud->empty()) {
        spdlog::error("[Estimator] Downsampled cloud is empty");
        return false;
    }
    
    // Step 2: Use downsampled cloud directly as feature cloud (no separate feature extraction)
    // Feature extraction is removed - ICP will compute normals from correspondences
    
    // Step 3: Set processed clouds in the frame
    frame->set_processed_cloud(downsampled_cloud);
    frame->set_feature_cloud(downsampled_cloud);  // Use same cloud as features
    
    spdlog::debug("[Estimator] Preprocessing: {} -> {} points (stride={}, voxels={})", 
                  raw_cloud->size(), downsampled_cloud->size(), 
                  m_config.point_stride, m_fast_voxel_grid->getVoxelCount());
    
    return true;
}

void Estimator::get_optimization_statistics(double& avg_iterations, double& avg_time_ms) const {
    if (m_optimization_call_count > 0) {
        avg_iterations = static_cast<double>(m_total_optimization_iterations) / m_optimization_call_count;
        avg_time_ms = m_total_optimization_time_ms / m_optimization_call_count;
    } else {
        avg_iterations = 0.0;
        avg_time_ms = 0.0;
    }
}

const SE3f& Estimator::get_current_pose() const {
    return m_T_wl_current;
}

size_t Estimator::get_keyframe_count() const {
    return m_keyframes.size();
}

std::shared_ptr<database::LidarFrame> Estimator::get_keyframe(size_t index) const {
    if (index >= m_keyframes.size()) {
        return nullptr;
    }
    return m_keyframes[index];
}

void Estimator::enable_loop_closure(bool enable) {
    if (m_loop_detector) {
        LoopClosureConfig config = m_loop_detector->get_config();
        config.enable_loop_detection = enable;
        m_loop_detector->update_config(config);
        spdlog::info("[Estimator] Loop closure detection {}", enable ? "enabled" : "disabled");
    }
}

void Estimator::set_loop_closure_config(const LoopClosureConfig& config) {
    if (m_loop_detector) {
        m_loop_detector->update_config(config);
    }
}

size_t Estimator::get_loop_closure_count() const {
    // For now, return 0 since we haven't implemented PGO yet
    // This will be updated when we add pose graph optimization
    return 0;
}

std::map<int, Eigen::Matrix4f> Estimator::get_optimized_trajectory() const {
    std::map<int, Eigen::Matrix4f> result;
    for (const auto& [id, pose] : m_optimized_poses) {
        result[id] = pose.matrix();
    }
    return result;
}

void Estimator::process_loop_closures(std::shared_ptr<database::LidarFrame> current_keyframe, 
                                     const std::vector<LoopCandidate>& loop_candidates) {
    
    if (loop_candidates.empty()) {
        return;
    }
    
    // Use only the best candidate (first one, already sorted by similarity score)
    const auto& candidate = loop_candidates[0];
    
    spdlog::info("[Estimator] Processing best loop closure candidate for ICP optimization");
    
    // Find the matched keyframe in our database
    std::shared_ptr<database::LidarFrame> matched_keyframe = nullptr;
    
    for (const auto& kf : m_keyframes) {
        if (static_cast<size_t>(kf->get_keyframe_id()) == candidate.match_keyframe_id) {
            matched_keyframe = kf;
            break;
        }
    }
    
    if (!matched_keyframe) {
        spdlog::warn("[Estimator] Could not find matched keyframe {} in database", candidate.match_keyframe_id);
        return;
    }

    // Get feature clouds from both keyframes
    auto local_feature_matched = matched_keyframe->get_feature_cloud();
    auto local_feature_current = current_keyframe->get_feature_cloud();

    if (!local_feature_matched || !local_feature_current ||
        local_feature_matched->empty() || local_feature_current->empty()) {
        spdlog::warn("[Estimator] Empty feature clouds for loop {} <-> {}", 
                    candidate.query_keyframe_id, candidate.match_keyframe_id);
        return;
    }

    Sophus::SE3f T_current_l2l;
    float inlier_ratio = 0.0f;

    bool icp_success = m_icp_optimizer->optimize_loop(
        current_keyframe,             // source frame (has fresh kdtree built)
        matched_keyframe,             // target frame (will use local map as features)
        T_current_l2l,                // optimized relative transform (output)
        inlier_ratio                  // inlier ratio (output)
    );

    if (!icp_success) {
        spdlog::warn("[Estimator] Loop closure ICP failed for {} <-> {}", 
                    candidate.query_keyframe_id, candidate.match_keyframe_id);
        return;
    }
    
    // Validate loop closure using inlier ratio
    const float min_inlier_ratio = 0.3f;  // Minimum 30% inliers required
    if (inlier_ratio < min_inlier_ratio) {
        spdlog::warn("[Estimator] Loop closure rejected: inlier ratio {:.2f}% < {:.2f}% for {} <-> {}", 
                    inlier_ratio * 100.0f, min_inlier_ratio * 100.0f,
                    candidate.query_keyframe_id, candidate.match_keyframe_id);
        return;
    }

    // T_current_l2l is the ICP correction: T_original^-1 * T_corrected
    // ICP optimizes curr_keyframe pose, returns correction transform
    
    // Get current poses (with drift)
    Sophus::SE3f T_world_current = current_keyframe->get_pose();
    Sophus::SE3f T_world_matched = matched_keyframe->get_pose();
    
    // Apply ICP correction: T_corrected = T_correction * T_original
    // ICP returns: T_correction = T_original^-1 * T_optimized
    // So: T_corrected = (T_original^-1 * T_optimized) * T_original = T_optimized
    Sophus::SE3f T_current_corrected = T_world_current * T_current_l2l;
    
    // Calculate pose difference for logging (how much correction ICP suggests)
    SE3f pose_diff = T_world_current.inverse() * T_current_corrected;
    float translation_diff = pose_diff.translation().norm();
    float rotation_diff = pose_diff.so3().log().norm() * 180.0f / M_PI;
    
    spdlog::info("[Estimator] Loop closure ICP success {} <-> {}: Δt={:.3f}m, Δr={:.2f}°, inliers={:.1f}%",
                candidate.query_keyframe_id, candidate.match_keyframe_id,
                translation_diff, rotation_diff, inlier_ratio * 100.0f);
    
    // Compute relative pose constraint: from matched to current
    // Using GTSAM's between() logic: poseFrom.between(poseTo) = poseFrom^-1 * poseTo
    Sophus::SE3f T_matched_to_current = T_world_matched.inverse() * T_current_corrected;
    
    // Check if PGO is enabled
    if (!m_config.pgo_enable_pgo) {
        spdlog::info("[Estimator] PGO disabled, skipping pose graph optimization");
        return;
    }
    
    // Build pose graph from scratch with all keyframes and odometry constraints
    spdlog::info("[PGO] Using Ceres optimizer");
    
    // Store pre-PGO poses for visualization (before optimization)
    std::map<int, SE3f> pre_pgo_poses;
    for (const auto& kf : m_keyframes) {
        // Use get_stored_pose() to get the actual stored value, not dynamic calculation
        pre_pgo_poses[kf->get_keyframe_id()] = kf->get_stored_pose();
    }
    
    // Clear previous graph
    m_pose_graph_optimizer->clear();
    
    // Add all keyframes and odometry constraints
    spdlog::info("[Estimator] Building pose graph with {} keyframes", m_keyframes.size());
    
    // Add all keyframes as optimization variables
    for (size_t i = 0; i < m_keyframes.size(); ++i) {
        auto& kf = m_keyframes[i];
        
        if (i == 0) {
            // First keyframe: add with prior factor (constant)
            m_pose_graph_optimizer->add_keyframe_pose(
                kf->get_keyframe_id(),
                pre_pgo_poses[kf->get_keyframe_id()],
                true);
        } else {
            // Non-first keyframe: add as variable
            m_pose_graph_optimizer->add_keyframe_pose(
                kf->get_keyframe_id(),
                pre_pgo_poses[kf->get_keyframe_id()],
                false);
            
            // Add odometry constraint from previous keyframe
            auto& prev_kf = m_keyframes[i-1];
            SE3f relative_pose = kf->get_relative_pose();
            
            double odom_trans_noise = m_config.pgo_odometry_translation_noise;
            double odom_rot_noise = m_config.pgo_odometry_rotation_noise;
            double trans_weight = 1.0 / (odom_trans_noise * odom_trans_noise);
            double rot_weight = 1.0 / (odom_rot_noise * odom_rot_noise);
            m_pose_graph_optimizer->add_odometry_constraint(
                prev_kf->get_keyframe_id(),
                kf->get_keyframe_id(),
                relative_pose,
                trans_weight,
                rot_weight
            );
        }
    }
    
    // Store and add loop closure constraint
    LoopConstraint loop_constraint;
    loop_constraint.from_keyframe_id = matched_keyframe->get_keyframe_id();
    loop_constraint.to_keyframe_id = current_keyframe->get_keyframe_id();
    loop_constraint.relative_pose = T_matched_to_current;
    loop_constraint.translation_noise = m_config.pgo_loop_translation_noise;
    loop_constraint.rotation_noise = m_config.pgo_loop_rotation_noise;
    
    // Accumulate all loop constraints
    m_loop_constraints.push_back(loop_constraint);
    spdlog::info("[Estimator] Stored loop closure constraint: {} -> {} (total loops: {})",
                loop_constraint.from_keyframe_id, loop_constraint.to_keyframe_id, 
                m_loop_constraints.size());
    
    // Add ALL accumulated loop constraints to pose graph
    spdlog::info("[Estimator] Adding {} stored loop constraints to PGO", m_loop_constraints.size());
    for (const auto& lc : m_loop_constraints) {
        double trans_weight = 1.0 / (lc.translation_noise * lc.translation_noise);
        double rot_weight = 1.0 / (lc.rotation_noise * lc.rotation_noise);
        spdlog::debug("[Estimator]   Loop {}->{}: trans_weight={}, rot_weight={}", 
                     lc.from_keyframe_id, lc.to_keyframe_id, trans_weight, rot_weight);
        m_pose_graph_optimizer->add_loop_closure_constraint(
            lc.from_keyframe_id,
            lc.to_keyframe_id,
            lc.relative_pose,
            trans_weight,
            rot_weight
        );
    }
    
    // Perform pose graph optimization
    spdlog::info("[Estimator] Running Ceres pose graph optimization with {} loop closures...", m_loop_constraints.size());
    
    bool opt_success = m_pose_graph_optimizer->optimize();
        
        if (opt_success) {
            // Print pose corrections without applying them
            auto optimized_poses = m_pose_graph_optimizer->get_all_optimized_poses();
            
            spdlog::info("[Estimator] ========== PGO Results (NOT APPLIED) ==========");
            spdlog::info("[Estimator] Total keyframes optimized: {}", optimized_poses.size());
            spdlog::info("[Estimator]");
            spdlog::info("[Estimator] Per-Keyframe Pose Corrections:");
            spdlog::info("[Estimator] KF_ID | Δt (m)  | Δr (deg) | Position Before -> After");
            spdlog::info("[Estimator] ------|---------|----------|------------------------");
            
            float max_translation_diff = 0.0f;
            float max_rotation_diff = 0.0f;
            float avg_translation_diff = 0.0f;
            float avg_rotation_diff = 0.0f;
            int count = 0;
            
            for (size_t i = 0; i < m_keyframes.size(); i++) {
                int kf_id = m_keyframes[i]->get_keyframe_id();
                auto it = optimized_poses.find(kf_id);
                
                if (it != optimized_poses.end()) {
                    SE3f old_pose = m_keyframes[i]->get_pose();
                    SE3f new_pose = it->second;
                    
                    float translation_diff = (new_pose.translation() - old_pose.translation()).norm();
                    float rotation_diff = (new_pose.so3().log() - old_pose.so3().log()).norm() * 180.0f / M_PI;
                    
                    // Print each keyframe's correction (debug level)
                    Eigen::Vector3f pos_before = old_pose.translation();
                    Eigen::Vector3f pos_after = new_pose.translation();
                    spdlog::debug("[Estimator] {:5d} | {:7.4f} | {:8.3f} | ({:6.2f},{:6.2f},{:6.2f}) -> ({:6.2f},{:6.2f},{:6.2f})",
                                kf_id, 
                                translation_diff, 
                                rotation_diff,
                                pos_before.x(), pos_before.y(), pos_before.z(),
                                pos_after.x(), pos_after.y(), pos_after.z());
                    
                    max_translation_diff = std::max(max_translation_diff, translation_diff);
                    max_rotation_diff = std::max(max_rotation_diff, rotation_diff);
                    avg_translation_diff += translation_diff;
                    avg_rotation_diff += rotation_diff;
                    count++;
                    
                    // Calculate odometry constraint cost
                    float odom_cost_before = 0.0f;
                    float odom_cost_after = 0.0f;
                    
                    if (i > 0) {
                        SE3f prev_old = m_keyframes[i-1]->get_pose();
                        SE3f prev_new = optimized_poses[m_keyframes[i-1]->get_keyframe_id()];
                        SE3f relative = prev_old.inverse() * old_pose;
                        
                        SE3f error_before = relative.inverse() * prev_old.inverse() * old_pose;
                        SE3f error_after = relative.inverse() * prev_new.inverse() * new_pose;
                        
                        Eigen::Matrix<float, 6, 1> log_error_before = error_before.log();
                        Eigen::Matrix<float, 6, 1> log_error_after = error_after.log();
                        
                        float t_weight = 1.0f / (0.1f * 0.1f);
                        float r_weight = 1.0f / (0.1f * 0.1f);
                        
                        odom_cost_before = t_weight * log_error_before.head<3>().squaredNorm() + 
                                          r_weight * log_error_before.tail<3>().squaredNorm();
                        odom_cost_after = t_weight * log_error_after.head<3>().squaredNorm() + 
                                         r_weight * log_error_after.tail<3>().squaredNorm();
                    }
                    
                    // Calculate loop closure constraint cost
                    float loop_cost_before = 0.0f;
                    float loop_cost_after = 0.0f;
                    
                    if ((matched_keyframe && kf_id == matched_keyframe->get_keyframe_id()) ||
                        (current_keyframe && kf_id == current_keyframe->get_keyframe_id())) {
                        
                        SE3f matched_old = matched_keyframe->get_pose();
                        SE3f matched_new = optimized_poses[matched_keyframe->get_keyframe_id()];
                        SE3f current_old = current_keyframe->get_pose();
                        SE3f current_new = optimized_poses[current_keyframe->get_keyframe_id()];
                        
                        SE3f error_before = T_matched_to_current.inverse() * matched_old.inverse() * current_old;
                        SE3f error_after = T_matched_to_current.inverse() * matched_new.inverse() * current_new;
                        
                        Eigen::Matrix<float, 6, 1> log_error_before = error_before.log();
                        Eigen::Matrix<float, 6, 1> log_error_after = error_after.log();
                        
                        float loop_t_weight = 1.0f / (0.01f * 0.01f);
                        float loop_r_weight = 1.0f / (0.01f * 0.01f);
                        
                        loop_cost_before = loop_t_weight * log_error_before.head<3>().squaredNorm() + 
                                          loop_r_weight * log_error_before.tail<3>().squaredNorm();
                        loop_cost_after = loop_t_weight * log_error_after.head<3>().squaredNorm() + 
                                         loop_r_weight * log_error_after.tail<3>().squaredNorm();
                    }
                    
                    float total_cost_before = odom_cost_before + loop_cost_before;
                    float total_cost_after = odom_cost_after + loop_cost_after;
                    
                }
            }
            
            if (count > 0) {
                avg_translation_diff /= count;
                avg_rotation_diff /= count;
                
                spdlog::info("[Estimator] ========== PGO Statistics =========="); 
                spdlog::info("[Estimator] Average correction: Δt={:.3f}m, Δr={:.2f}°", avg_translation_diff, avg_rotation_diff);
                spdlog::info("[Estimator] Maximum correction: Δt={:.3f}m, Δr={:.2f}°", max_translation_diff, max_rotation_diff);
            }
            
            spdlog::info("[Estimator] =========================================");
            
            // Store optimized poses for visualization
            m_optimized_poses = optimized_poses;
            
            // Apply pose graph optimization results to all keyframes
            spdlog::info("[Estimator] Applying PGO corrections to all keyframes...");
            apply_pose_graph_optimization();
            
            // Update cooldown
            m_last_successful_loop_keyframe_id = current_keyframe->get_keyframe_id();
            spdlog::info("[Estimator] Loop closure cooldown activated: next detection after keyframe {}",
                        m_last_successful_loop_keyframe_id + m_config.loop_min_keyframe_gap);
        } else {
            spdlog::error("[Estimator] Pose graph optimization failed!");
        }
}

void Estimator::apply_pose_graph_optimization() {
    // Get all optimized poses from Ceres pose graph optimizer
    auto optimized_poses = m_pose_graph_optimizer->get_all_optimized_poses();
    
    if (optimized_poses.empty()) {
        spdlog::warn("[Estimator] No optimized poses available from pose graph!");
        return;
    }
    
    spdlog::info("[Estimator] Applying PGO results to {} keyframes", optimized_poses.size());

    auto last_keyframe = m_keyframes.back();

    Sophus::SE3f last_keyframe_pose_before_opt = last_keyframe->get_pose();
    Sophus::SE3f last_keyframe_pose_after_opt = optimized_poses[last_keyframe->get_keyframe_id()];
    Sophus::SE3f total_correction = last_keyframe_pose_after_opt * last_keyframe_pose_before_opt.inverse();
    
    // Update all keyframe poses (absolute poses only)
    // NOTE: We do NOT recalculate relative poses after PGO!
    // The PGO optimization already considers relative pose constraints,
    // so recalculating and normalizing them would distort the optimization result.
    for (auto& keyframe : m_keyframes) {
        int kf_id = keyframe->get_keyframe_id();
        
        auto it = optimized_poses.find(kf_id);
        if (it == optimized_poses.end()) {
            spdlog::warn("[Estimator] No optimized pose for keyframe {}", kf_id);
            continue;
        }
        
        SE3f old_pose = keyframe->get_pose();
        SE3f new_pose = it->second;
        
        float translation_diff = (new_pose.translation() - old_pose.translation()).norm();
        float rotation_diff = (new_pose.so3().log() - old_pose.so3().log()).norm() * 180.0f / M_PI;
        
        spdlog::debug("[Estimator] Keyframe {}: Δt={:.3f}m, Δr={:.2f}°", 
                     kf_id, translation_diff, rotation_diff);
        
        // Update keyframe pose (absolute pose)
        keyframe->set_pose(new_pose);
    }
    
    // Relative poses are kept as-is (not recalculated from absolute poses)
    // This preserves the optimization result from PGO

    // transform last keyframe's local map
    auto last_local_map = last_keyframe->get_local_map();

    util::PointCloudPtr transformed_local_map = std::make_shared<util::PointCloud>();

    util::transform_point_cloud(
        last_local_map,
        transformed_local_map,
        total_correction.matrix()
    );

    // Update last keyframe's local map
    last_keyframe->set_local_map(transformed_local_map);
    last_keyframe->build_local_map_kdtree();
}

void Estimator::loop_pgo_thread_function() {
    spdlog::info("[Background] Loop+PGO thread started");
    
    while (m_thread_running) {
        int query_kf_id = -1;
        std::shared_ptr<database::LidarFrame> query_keyframe = nullptr;
        
        // Wait for query or termination signal
        {
            std::unique_lock<std::mutex> lock(m_query_mutex);
            m_query_cv.wait(lock, [this] { 
                return !m_loop_query_queue.empty() || !m_thread_running; 
            });
            
            if (!m_thread_running) break;
            
            // If PGO is in progress, skip processing (wait for next wake-up)
            if (m_pgo_in_progress) {
                spdlog::debug("[Background] PGO in progress, skipping queries");
                continue;
            }
            
            // Get most recent query and clear the rest (for real-time performance)
            query_kf_id = m_loop_query_queue.back();
            m_loop_query_queue.clear();
            spdlog::debug("[Background] Processing loop query for KF {}", query_kf_id);
        }
        
        // Find the keyframe (read-only access with lock)
        {
            std::lock_guard<std::mutex> lock(m_keyframes_mutex);
            for (const auto& kf : m_keyframes) {
                if (kf->get_keyframe_id() == query_kf_id) {
                    query_keyframe = kf;
                    break;
                }
            }
        }
        
        if (!query_keyframe) {
            spdlog::warn("[Background] Keyframe {} not found", query_kf_id);
            continue;
        }
        
        // Detect loop closure candidates
        auto loop_candidates = m_loop_detector->detect_loop_closures(query_keyframe);
        
        if (loop_candidates.empty()) {
            spdlog::debug("[Background] No loop candidates found for KF {}", query_kf_id);
            continue;
        }
        
        // Loop detected! Start PGO
        m_pgo_in_progress = true;
        
        // Process loop closure (ICP optimization + PGO)
        bool pgo_success = run_pgo_for_loop(query_keyframe, loop_candidates);
        
        m_pgo_in_progress = false;
        
        if (pgo_success) {
            
            // Clear accumulated queries during PGO (they're outdated now)
            {
                std::lock_guard<std::mutex> lock(m_query_mutex);
                m_loop_query_queue.clear();
            }
        }
    }
    
    spdlog::info("[Background] Loop+PGO thread stopped");
}

bool Estimator::run_pgo_for_loop(
    std::shared_ptr<database::LidarFrame> current_keyframe,
    const std::vector<LoopCandidate>& loop_candidates) 
{
    // Use only the best candidate (first one, already sorted by similarity score)
    const auto& candidate = loop_candidates[0];
    
    // Find the matched keyframe
    std::shared_ptr<database::LidarFrame> matched_keyframe = nullptr;
    {
        std::lock_guard<std::mutex> lock(m_keyframes_mutex);
        for (const auto& kf : m_keyframes) {
            if (static_cast<size_t>(kf->get_keyframe_id()) == candidate.match_keyframe_id) {
                matched_keyframe = kf;
                break;
            }
        }
    }

    if (!matched_keyframe)
    {
        spdlog::warn("[Background] Could not find matched keyframe {}",
                     candidate.match_keyframe_id);
        return false;
    }

    // Get feature clouds from both keyframes
    auto local_feature_matched = matched_keyframe->get_feature_cloud();
    auto local_feature_current = current_keyframe->get_feature_cloud();

    if (!local_feature_matched || !local_feature_current ||
        local_feature_matched->empty() || local_feature_current->empty())
    {
        spdlog::warn("[Background] Empty feature clouds for loop {} <-> {}",
                     candidate.query_keyframe_id, candidate.match_keyframe_id);
        return false;
    }

    // Perform ICP optimization for loop closure
    Sophus::SE3f T_current_l2l;
    float inlier_ratio = 0.0f;

    bool icp_success = m_icp_optimizer->optimize_loop(
        current_keyframe,
        matched_keyframe,
        T_current_l2l,
        inlier_ratio
    );

    if (!icp_success) {
        spdlog::warn("[Background] Loop ICP failed {} <-> {}", 
                    candidate.query_keyframe_id, candidate.match_keyframe_id);
        return false;
    }
    
    // Validate loop closure using inlier ratio
    const float min_inlier_ratio = 0.3f;
    if (inlier_ratio < min_inlier_ratio) {
        spdlog::warn("[Background] Loop rejected: {:.1f}% inliers < {:.1f}%", 
                    inlier_ratio * 100.0f, min_inlier_ratio * 100.0f);
        return false;
    }

    // Get current poses
    Sophus::SE3f T_world_current = current_keyframe->get_pose();
    Sophus::SE3f T_world_matched = matched_keyframe->get_pose();
    
    // Apply ICP correction
    Sophus::SE3f T_current_corrected = T_world_current * T_current_l2l;
    
    // Calculate pose difference
    SE3f pose_diff = T_world_current.inverse() * T_current_corrected;
    float translation_diff = pose_diff.translation().norm();
    float rotation_diff = pose_diff.so3().log().norm() * 180.0f / M_PI;
    
    spdlog::info("[Background] Loop detected {} <-> {}: Δt={:.2f}m, Δr={:.2f}°, {:.1f}% inliers",
                candidate.query_keyframe_id, candidate.match_keyframe_id,
                translation_diff, rotation_diff, inlier_ratio * 100.0f);
    
    // Compute relative pose constraint
    Sophus::SE3f T_matched_to_current = T_world_matched.inverse() * T_current_corrected;
    
    // Check if PGO is enabled
    if (!m_config.pgo_enable_pgo) {
        spdlog::info("[Background] PGO disabled");
        return false;
    }
    
    // Take snapshot of keyframes (poses) for PGO
    std::vector<int> kf_ids;
    std::vector<SE3f> kf_poses;
    std::vector<SE3f> kf_relatives;
    
    {
        std::lock_guard<std::mutex> lock(m_keyframes_mutex);
        kf_ids.reserve(m_keyframes.size());
        kf_poses.reserve(m_keyframes.size());
        kf_relatives.reserve(m_keyframes.size());
        
        for (const auto& kf : m_keyframes) {
            kf_ids.push_back(kf->get_keyframe_id());
            kf_poses.push_back(kf->get_pose());
            kf_relatives.push_back(kf->get_relative_pose());
        }
    }
    
    // Clear and rebuild pose graph
    m_pose_graph_optimizer->clear();
    
    // Add all keyframe poses
    for (size_t i = 0; i < kf_ids.size(); ++i) {
        bool is_fixed = (i == 0);  // Fix first keyframe
        m_pose_graph_optimizer->add_keyframe_pose(kf_ids[i], kf_poses[i], is_fixed);
    }
    
    // Add odometry constraints
    double odometry_translation_weight = 1.0 / (m_config.pgo_odometry_translation_noise * 
                                                 m_config.pgo_odometry_translation_noise);
    double odometry_rotation_weight = 1.0 / (m_config.pgo_odometry_rotation_noise * 
                                             m_config.pgo_odometry_rotation_noise);
    
    for (size_t i = 1; i < kf_ids.size(); ++i) {
        m_pose_graph_optimizer->add_odometry_constraint(
            kf_ids[i-1], kf_ids[i],
            kf_relatives[i],
            odometry_translation_weight,
            odometry_rotation_weight
        );
    }
    
    // Add loop closure constraint
    double loop_translation_weight = 1.0 / (m_config.pgo_loop_translation_noise * 
                                            m_config.pgo_loop_translation_noise);
    double loop_rotation_weight = 1.0 / (m_config.pgo_loop_rotation_noise * 
                                         m_config.pgo_loop_rotation_noise);
    
    m_pose_graph_optimizer->add_loop_closure_constraint(
        matched_keyframe->get_keyframe_id(),
        current_keyframe->get_keyframe_id(),
        T_matched_to_current,
        loop_translation_weight,
        loop_rotation_weight
    );
    
    // Store loop constraint for future use
    LoopConstraint loop_constraint;
    loop_constraint.from_keyframe_id = matched_keyframe->get_keyframe_id();
    loop_constraint.to_keyframe_id = current_keyframe->get_keyframe_id();
    loop_constraint.relative_pose = T_matched_to_current;
    loop_constraint.translation_noise = m_config.pgo_loop_translation_noise;
    loop_constraint.rotation_noise = m_config.pgo_loop_rotation_noise;
    m_loop_constraints.push_back(loop_constraint);
    
    // Optimize pose graph
    if (!m_pose_graph_optimizer->optimize()) {
        spdlog::error("[Background] PGO failed!");
        return false;
    }
    
    // Get optimized poses and calculate statistics
    auto optimized_poses = m_pose_graph_optimizer->get_all_optimized_poses();
    
    float avg_trans_diff = 0.0f;
    float avg_rot_diff = 0.0f;
    float max_trans_diff = 0.0f;
    float max_rot_diff = 0.0f;
    
    for (size_t i = 0; i < kf_ids.size(); ++i) {
        auto it = optimized_poses.find(kf_ids[i]);
        if (it != optimized_poses.end()) {
            SE3f old_pose = kf_poses[i];
            SE3f new_pose = it->second;
            
            float trans_diff = (new_pose.translation() - old_pose.translation()).norm();
            float rot_diff = (new_pose.so3().log() - old_pose.so3().log()).norm() * 180.0f / M_PI;
            
            avg_trans_diff += trans_diff;
            avg_rot_diff += rot_diff;
            max_trans_diff = std::max(max_trans_diff, trans_diff);
            max_rot_diff = std::max(max_rot_diff, rot_diff);
        }
    }
    
    avg_trans_diff /= kf_ids.size();
    avg_rot_diff /= kf_ids.size();
    
    spdlog::info("[Background] PGO completed: {} KFs, avg Δ({:.3f}m, {:.2f}°), max Δ({:.3f}m, {:.2f}°)",
                 kf_ids.size(), avg_trans_diff, avg_rot_diff, max_trans_diff, max_rot_diff);
    
    // Calculate correction transform for last keyframe
    int last_kf_id = kf_ids.back();
    SE3f last_kf_pose_before = kf_poses.back();
    SE3f last_kf_pose_after = optimized_poses[last_kf_id];
    SE3f last_kf_correction = last_kf_pose_after * last_kf_pose_before.inverse();
    
    // Prepare PGO result for main thread
    PGOResult result;
    result.last_optimized_kf_id = last_kf_id;
    result.optimized_poses = std::move(optimized_poses);
    result.last_kf_correction = last_kf_correction;
    result.timestamp = std::chrono::steady_clock::now();
    
    // Put result in queue
    {
        std::lock_guard<std::mutex> lock(m_result_mutex);
        m_pending_result = std::move(result);
    }
    
    return true;
}

void Estimator::apply_pending_pgo_result_if_available() {
    // Check if there's a pending result (non-blocking)
    std::optional<PGOResult> result;
    {
        std::lock_guard<std::mutex> lock(m_result_mutex);
        if (m_pending_result.has_value()) {
            result = std::move(m_pending_result);
            m_pending_result.reset();
        }
    }
    
    if (!result) return;  // No pending result
    
    // Apply PGO result
    int last_optimized_id = result->last_optimized_kf_id;
    
    spdlog::info("[Main] Applying PGO result ({} keyframes optimized)", 
                 result->optimized_poses.size());
    
    // Step 1: Update poses for keyframes included in PGO
    {
        std::lock_guard<std::mutex> lock(m_keyframes_mutex);
        for (auto& kf : m_keyframes) {
            int kf_id = kf->get_keyframe_id();
            
            if (kf_id <= last_optimized_id) {
                // Keyframe was included in PGO - update with optimized pose
                auto it = result->optimized_poses.find(kf_id);
                if (it != result->optimized_poses.end()) {
                    SE3f old_pose = kf->get_pose();
                    SE3f new_pose = it->second;
                    
                    float trans_diff = (new_pose.translation() - old_pose.translation()).norm();
                    float rot_diff = (new_pose.so3().log() - old_pose.so3().log()).norm() * 180.0f / M_PI;
                    
                    kf->set_pose(new_pose);
                }
            } else {
                // Keyframe was added after PGO started - will be updated by propagation
                break;
            }
        }
    }
    
    // Step 2: Propagate poses to keyframes added after PGO
    propagate_poses_after_pgo(last_optimized_id);
    
    // Step 3: Transform current keyframe's map
    transform_current_keyframe_map(result->last_kf_correction);
    
    // Update cooldown
    m_last_successful_loop_keyframe_id = last_optimized_id;
    
    spdlog::info("[Main] PGO applied, cooldown until KF {}",
                m_last_successful_loop_keyframe_id + m_config.loop_min_keyframe_gap);
}

void Estimator::propagate_poses_after_pgo(int last_optimized_kf_id) {
    std::lock_guard<std::mutex> lock(m_keyframes_mutex);
    
    // Find the last optimized keyframe
    std::shared_ptr<database::LidarFrame> last_optimized_kf = nullptr;
    SE3f accumulated_pose;
    bool found_start = false;
    
    for (auto& kf : m_keyframes) {
        if (kf->get_keyframe_id() == last_optimized_kf_id) {
            last_optimized_kf = kf;
            accumulated_pose = kf->get_pose();
            found_start = true;
            continue;
        }
        
        if (!found_start) continue;
        
        // Propagate pose using relative transform: new_pose = prev_pose * relative
        SE3f relative = kf->get_relative_pose();
        accumulated_pose = accumulated_pose * relative;
        
        kf->set_pose(accumulated_pose);
    }
    
    if (!found_start) {
        spdlog::warn("[Main] Could not find last optimized KF {} for propagation", 
                     last_optimized_kf_id);
    }
}

void Estimator::transform_current_keyframe_map(const SE3f& correction) {
    std::lock_guard<std::mutex> lock(m_keyframes_mutex);
    
    if (m_keyframes.empty()) return;
    
    // Transform only the most recent keyframe's map
    auto current_kf = m_keyframes.back();
    auto local_map = current_kf->get_local_map();
    
    if (!local_map || local_map->empty()) {
        spdlog::debug("[Main] Current keyframe has no local map to transform");
        return;
    }
    
    util::PointCloudPtr transformed_map = std::make_shared<util::PointCloud>();
    util::transform_point_cloud(local_map, transformed_map, correction.matrix());
    
    current_kf->set_local_map(transformed_map);
    current_kf->build_local_map_kdtree();
}

bool Estimator::save_map_to_ply(const std::string& output_path, float voxel_size) {
    std::lock_guard<std::mutex> lock(m_keyframes_mutex);
    
    if (m_keyframes.empty()) {
        spdlog::warn("[Estimator] No keyframes to save");
        return false;
    }
    
    spdlog::info("[Estimator] Building final map from {} keyframes...", m_keyframes.size());
    
    // Accumulate all keyframe feature clouds in world coordinates
    util::PointCloudPtr accumulated_map = std::make_shared<util::PointCloud>();
    
    for (const auto& kf : m_keyframes) {
        auto feature_cloud = kf->get_feature_cloud();
        if (!feature_cloud || feature_cloud->empty()) {
            continue;
        }
        
        // Transform feature cloud to world coordinates
        SE3f pose = kf->get_pose();
        util::PointCloudPtr transformed_cloud = std::make_shared<util::PointCloud>();
        util::transform_point_cloud(feature_cloud, transformed_cloud, pose.matrix());
        
        // Accumulate
        *accumulated_map += *transformed_cloud;
    }
    
    if (accumulated_map->empty()) {
        spdlog::error("[Estimator] No points in accumulated map");
        return false;
    }
    
    spdlog::info("[Estimator] Accumulated map: {} points", accumulated_map->size());
    
    // Downsample if voxel_size > 0
    util::PointCloudPtr final_map = accumulated_map;
    if (voxel_size > 0.0f) {
        util::VoxelGrid voxel_filter;
        voxel_filter.setLeafSize(voxel_size);
        voxel_filter.setInputCloud(accumulated_map);
        
        final_map = std::make_shared<util::PointCloud>();
        voxel_filter.filter(*final_map);
        
        spdlog::info("[Estimator] Downsampled map: {} -> {} points (voxel_size={})", 
                     accumulated_map->size(), final_map->size(), voxel_size);
    }
    
    // Save as PLY binary format
    if (!util::save_point_cloud_ply(output_path, final_map)) {
        spdlog::error("[Estimator] Failed to save map to {}", output_path);
        return false;
    }
    
    spdlog::info("[Estimator] Saved final map to {} ({} points)", output_path, final_map->size());
    return true;
}

void Estimator::print_timing_statistics() const {
    if (m_timing_history.empty()) {
        return;
    }
    
    // Calculate statistics for last 100 frames (or all if less)
    size_t start_idx = m_timing_history.size() > 100 ? m_timing_history.size() - 100 : 0;
    size_t count = m_timing_history.size() - start_idx;
    
    double sum_preprocess = 0.0, sum_icp = 0.0, sum_map = 0.0, sum_total = 0.0;
    double max_preprocess = 0.0, max_icp = 0.0, max_map = 0.0, max_total = 0.0;
    double min_preprocess = 1e9, min_icp = 1e9, min_map = 1e9, min_total = 1e9;
    
    for (size_t i = start_idx; i < m_timing_history.size(); ++i) {
        const auto& t = m_timing_history[i];
        
        sum_preprocess += t.preprocessing_ms;
        sum_icp += t.icp_ms;
        sum_map += t.map_update_ms;
        sum_total += t.total_ms;
        
        max_preprocess = std::max(max_preprocess, t.preprocessing_ms);
        max_icp = std::max(max_icp, t.icp_ms);
        max_map = std::max(max_map, t.map_update_ms);
        max_total = std::max(max_total, t.total_ms);
        
        min_preprocess = std::min(min_preprocess, t.preprocessing_ms);
        min_icp = std::min(min_icp, t.icp_ms);
        min_map = std::min(min_map, t.map_update_ms);
        min_total = std::min(min_total, t.total_ms);
    }
    
    double avg_preprocess = sum_preprocess / count;
    double avg_icp = sum_icp / count;
    double avg_map = sum_map / count;
    double avg_total = sum_total / count;
    
    spdlog::info("============================================================");
    spdlog::info("[Timing Stats] Frame {} (last {} frames)", m_frame_count, count);
    spdlog::info("------------------------------------------------------------");
    spdlog::info("              |   Avg (ms)  |   Min (ms)  |   Max (ms)  ");
    spdlog::info("------------------------------------------------------------");
    spdlog::info(" Preprocess   | {:>10.2f}  | {:>10.2f}  | {:>10.2f}  ", avg_preprocess, min_preprocess, max_preprocess);
    spdlog::info(" ICP          | {:>10.2f}  | {:>10.2f}  | {:>10.2f}  ", avg_icp, min_icp, max_icp);
    spdlog::info(" Map Update   | {:>10.2f}  | {:>10.2f}  | {:>10.2f}  ", avg_map, min_map, max_map);
    spdlog::info("------------------------------------------------------------");
    spdlog::info(" Total        | {:>10.2f}  | {:>10.2f}  | {:>10.2f}  ", avg_total, min_total, max_total);
    spdlog::info("============================================================");
}

} // namespace processing
} // namespace lidar_odometry
