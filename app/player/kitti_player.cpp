/**
 * @file      kitti_player.cpp
 * @brief     KITTI dataset player implementation
 * @author    Seungwon Choi
 * @date      2025-09-25
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "kitti_player.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <thread>
#include <numeric>
#include <filesystem>
#include <set>
#include <map>

#include <spdlog/spdlog.h>

#include <util/Config.h>
#include <util/PointCloudUtils.h>  // Added our point cloud utilities
#include <util/Types.h>
#include <processing/Estimator.h>
#include <database/LidarFrame.h>
#include <viewer/PangolinViewer.h>

namespace lidar_odometry {
namespace app {

KittiPlayerResult KittiPlayer::run(const KittiPlayerConfig& config) {
    KittiPlayerResult result;
    
    try {
        // 1. Load configuration
        util::ConfigManager::instance().load_from_file(config.config_path);
        const auto& system_config = util::ConfigManager::instance().get_config();
        spdlog::info("[KittiPlayer] Successfully loaded configuration from: {}", config.config_path);
        
        // 2. Load dataset
        auto point_cloud_data = load_point_cloud_list(config.dataset_path, 
                                                     config.start_frame, 
                                                     config.end_frame, 
                                                     config.frame_skip);
        if (point_cloud_data.empty()) {
            result.error_message = "No point cloud files found in dataset";
            return result;
        }
        
        spdlog::info("[KittiPlayer] Loaded {} point cloud files", point_cloud_data.size());
        
        // 3. Load ground truth if available (disabled)
        std::vector<GroundTruthPose> gt_poses;
        // GT loading disabled - use EVO tool for evaluation instead
        
        // 4. Initialize systems
        auto viewer = initialize_viewer(config);
        initialize_estimator(system_config);
        
        // 5. Process frames
        FrameContext context;
        context.step_mode = config.step_mode;
        context.auto_play = !config.step_mode;
        
        // Fill ground truth poses in context (disabled)
        // GT processing disabled - trajectory saved for EVO evaluation
        
        // spdlog::info("[KittiPlayer] Processing frames {} to {} (step mode: {})", 0, point_cloud_data.size(), config.step_mode ? "enabled" : "disabled");
        
        context.current_idx = 0;
        while (context.current_idx < point_cloud_data.size()) {
            // Handle viewer controls first
            if (viewer && !handle_viewer_controls(*viewer, context)) {
                break;
            }
            
            bool should_process_frame = false;
            
            // Check processing conditions based on mode
            if (context.auto_play) {
                should_process_frame = true;
            } else {
                if (context.advance_frame) {
                    should_process_frame = true;
                    context.advance_frame = false;
                } else {
                    if (viewer) {
                        // Just sleep briefly as rendering runs in thread
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(30));
                    continue;
                }
            }
            
            if (should_process_frame) {
                // Set current frame information in context
                context.frame_index = context.current_idx;
                context.timestamp = static_cast<double>(context.current_idx) * 0.1; // 10Hz assumption
                
                // Load point cloud
                auto point_cloud = load_point_cloud(config.dataset_path, 
                                                   point_cloud_data[context.current_idx].filename);
                
                if (!point_cloud || point_cloud->empty()) {
                    spdlog::warn("[KittiPlayer] Skipping frame {} due to empty point cloud", 
                                context.current_idx);
                    ++context.current_idx;
                    continue;
                }
                // std::cout<<"\n";
                // spdlog::info("[KittiPlayer] Processing frame {} with {} points\n", context.current_idx, point_cloud->size());
                
                // Process single frame
                auto frame_start = std::chrono::high_resolution_clock::now();
                process_single_frame(point_cloud, context);
                auto frame_end = std::chrono::high_resolution_clock::now();
                
                auto frame_duration = std::chrono::duration_cast<std::chrono::microseconds>(frame_end - frame_start);
                double total_time_ms = frame_duration.count() / 1000.0;
                result.frame_processing_times.push_back(total_time_ms);
                
                // Handle ground truth alignment (disabled)
                // GT alignment disabled
                
                // Update viewer
                if (viewer) {
                    update_viewer(*viewer, context, point_cloud);
                }
                
                
                ++context.current_idx;
                ++context.processed_frames;
                
                // // Sleep in auto mode for real-time visualization
                // if (context.auto_play) {
                //     double sleep_time_ms = 100 - total_time_ms;
                //     if (sleep_time_ms > 0) {
                //         std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(sleep_time_ms)));
                //     }
                // }
            }
        }
        
        // 6. Save results
        if (config.enable_statistics) {
            std::string output_base = config.dataset_path + "/results";
            std::filesystem::create_directories(output_base);
            
            // Get SystemConfig for output directory and sequence info
            const auto& sys_config = util::config();
            
            // Generate filename with sequence and method: e.g., "07_PKO.txt"
            std::string seq_method_filename = sys_config.kitti_sequence + "_" + sys_config.scale_method + ".txt";
            
            // Create output directory: Result/07/
            std::string seq_output_dir = sys_config.output_directory + "/" + sys_config.kitti_sequence;
            std::filesystem::create_directories(seq_output_dir);
            
            // Save trajectory in sequence directory: Result/07/07_PKO.txt
            std::string kitti_output_path = seq_output_dir + "/" + seq_method_filename;
            
            save_trajectory_kitti_format(context, kitti_output_path);
            
            // GT evaluation disabled - use EVO tool for evaluation instead

            spdlog::info("[KittiPlayer] Saved trajectory to {}", kitti_output_path);
        }
        
        // 7. Calculate final statistics
        result.success = true;
        result.processed_frames = context.processed_frames;
        if (!result.frame_processing_times.empty()) {
            result.average_processing_time_ms = std::accumulate(
                result.frame_processing_times.begin(), 
                result.frame_processing_times.end(), 0.0) / result.frame_processing_times.size();
        }
        
        spdlog::info("[KittiPlayer] Successfully processed {} frames", result.processed_frames);
        
        // Get ICP statistics from estimator
        double avg_icp_iterations, avg_icp_time_ms;
        m_estimator->get_optimization_statistics(avg_icp_iterations, avg_icp_time_ms);
        
        // Generate output path for display
        std::string display_output_path = "";
        if (config.enable_statistics) {
            const auto& sys_config = util::config();
            std::string seq_method_filename = sys_config.kitti_sequence + "_" + sys_config.scale_method + ".txt";
            display_output_path = sys_config.output_directory + "/" + seq_method_filename;
        }
        
        // Display final statistics summary
        if (config.enable_console_statistics && result.success) {
            spdlog::info("════════════════════════════════════════════════════════════════════");
            spdlog::info("                          TIME STATISTICS                           ");
            spdlog::info("════════════════════════════════════════════════════════════════════");
            spdlog::info(" Total Frames Processed: {}", result.processed_frames);
            spdlog::info(" Average Processing Time: {:.2f}ms", result.average_processing_time_ms);
            double fps = 1000.0 / result.average_processing_time_ms;
            spdlog::info(" Average Frame Rate: {:.1f}fps", fps);
            
           
        }
        
        // Save final map if enabled
        const auto& sys_config = util::config();
        if (sys_config.output_save_map && m_estimator) {
            // Create output directory: Result/07/
            std::string seq_output_dir = sys_config.output_directory + "/" + sys_config.kitti_sequence;
            std::filesystem::create_directories(seq_output_dir);
            
            // Save map in sequence directory: Result/07/map.ply
            std::string map_output_path = seq_output_dir + "/map.ply";
            spdlog::info("[KittiPlayer] Saving final map to {}...", map_output_path);
            if (m_estimator->save_map_to_ply(map_output_path, sys_config.output_map_voxel_size)) {
                spdlog::info("[KittiPlayer] Map saved successfully!");
            } else {
                spdlog::error("[KittiPlayer] Failed to save map");
            }
        }
        
        // Wait for viewer finish if enabled
        if (viewer) {
            spdlog::info("[KittiPlayer] Processing completed! Close viewer to exit.");
            while (!viewer->should_close()) {
                // Render runs in thread, no need for explicit call
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
            }
        }
        
    } catch (const std::exception& e) {
        result.error_message = e.what();
        spdlog::error("[KittiPlayer] Exception occurred: {}", e.what());
    }
    
    return result;
}

KittiPlayerResult KittiPlayer::run_from_yaml(const std::string& config_path) {
    KittiPlayerResult result;
    
    try {
        // Load YAML configuration
        util::ConfigManager::instance().load_from_file(config_path);
        const auto& system_config = util::ConfigManager::instance().get_config();
        
        // Create KittiPlayerConfig from YAML settings
        KittiPlayerConfig config;
        config.config_path = config_path;
        
        // Auto-construct dataset path based on sequence
        config.dataset_path = system_config.data_directory + "/" + system_config.kitti_sequence;
        
        // Auto-construct ground truth path if enabled (disabled)
        // GT path construction disabled - use EVO tool for evaluation
        
        // Set other configuration from YAML
        config.enable_viewer = system_config.player_enable_viewer;
        config.enable_statistics = system_config.player_enable_statistics;
        config.enable_console_statistics = system_config.player_enable_console_statistics;
        config.step_mode = system_config.player_step_mode;
        config.start_frame = system_config.kitti_start_frame;
        config.end_frame = system_config.kitti_end_frame;
        config.frame_skip = system_config.kitti_frame_skip;
        config.viewer_width = system_config.viewer_width;
        config.viewer_height = system_config.viewer_height;
        
        spdlog::info("[KittiPlayer] Configuration from YAML:");
        spdlog::info("  Dataset path: {}", config.dataset_path);
        spdlog::info("  Sequence: {}", system_config.kitti_sequence);
        spdlog::info("  Frame range: {} - {} (skip: {})", config.start_frame, 
                    config.end_frame == -1 ? "all" : std::to_string(config.end_frame), config.frame_skip);
        spdlog::info("  Viewer enabled: {}", config.enable_viewer);
        spdlog::info("  Step mode: {}", config.step_mode);
        
        // Run with constructed configuration
        return run(config);
        
    } catch (const std::exception& e) {
        result.error_message = e.what();
        spdlog::error("[KittiPlayer] Exception in run_from_yaml: {}", e.what());
    }
    
    return result;
}

std::vector<PointCloudData> KittiPlayer::load_point_cloud_list(const std::string& dataset_path,
                                                               int start_frame,
                                                               int end_frame,
                                                               int frame_skip) {
    std::vector<PointCloudData> point_cloud_data;
    
    // Get velodyne directory path
    std::string velodyne_path = dataset_path;
    if (velodyne_path.back() != '/') {
        velodyne_path += "/";
    }
    velodyne_path += "velodyne";
    
    // Get all bin files
    auto bin_files = get_bin_files(velodyne_path);
    if (bin_files.empty()) {
        spdlog::error("[KittiPlayer] No .bin files found in: {}", velodyne_path);
        return point_cloud_data;
    }
    
    // Apply frame range and skip
    int actual_end = (end_frame == -1) ? static_cast<int>(bin_files.size()) : std::min(end_frame, static_cast<int>(bin_files.size()));
    
    for (int i = start_frame; i < actual_end; i += frame_skip) {
        if (i >= static_cast<int>(bin_files.size())) break;
        
        PointCloudData data;
        data.frame_id = i;
        data.filename = bin_files[i];
        data.timestamp = static_cast<long long>(i) * 100000000LL; // 10Hz假设
        point_cloud_data.push_back(data);
    }
    
    spdlog::info("[KittiPlayer] Found {} point cloud files (range: {}-{}, skip: {})", 
                point_cloud_data.size(), start_frame, actual_end - 1, frame_skip);
    
    return point_cloud_data;
}

util::PointCloud::Ptr KittiPlayer::load_point_cloud(const std::string& dataset_path, 
                                                                  const std::string& filename) {
    std::string velodyne_path = dataset_path;
    if (velodyne_path.back() != '/') {
        velodyne_path += "/";
    }
    velodyne_path += "velodyne/" + filename;
    
    auto point_cloud = util::load_kitti_binary(velodyne_path);
    
    if (!point_cloud || point_cloud->empty()) {
        spdlog::error("[KittiPlayer] Failed to load point cloud: {}", velodyne_path);
        return std::make_shared<util::PointCloud>();
    }
    
    return point_cloud;
}

std::shared_ptr<viewer::PangolinViewer> KittiPlayer::initialize_viewer(const KittiPlayerConfig& config) {
    if (!config.enable_viewer) {
        return nullptr;
    }
    
    auto viewer = std::make_shared<viewer::PangolinViewer>();
    if (viewer->initialize(config.viewer_width, config.viewer_height)) {
        spdlog::info("[KittiPlayer] Viewer initialized successfully");
        
        // Wait for viewer to be ready
        while (!viewer->is_ready()) {
            // Render runs in thread, no need for explicit call
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
        }
        spdlog::info("[KittiPlayer] Viewer is ready!");
        return viewer;
    } else {
        spdlog::warn("[KittiPlayer] Failed to initialize viewer");
        return nullptr;
    }
}

void KittiPlayer::initialize_estimator(const util::SystemConfig& config) {
    // Create estimator directly with SystemConfig
    m_estimator = std::make_unique<processing::Estimator>(config);
    spdlog::info("[KittiPlayer] Estimator initialized");
}

double KittiPlayer::process_single_frame(std::shared_ptr<lidar_odometry::util::PointCloud> point_cloud,
                                        FrameContext& context) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Process frame through LiDAR odometry
    // Create LidarFrame from point cloud
    auto lidar_frame = std::make_shared<database::LidarFrame>(context.frame_index, context.timestamp, point_cloud);
    
    bool result = m_estimator->process_frame(lidar_frame);
    
    // Store the processed frame in context
    context.current_lidar_frame = lidar_frame;
    
    // Store frame for dynamic pose evaluation (use get_pose() later)
    context.processed_frames_list.push_back(lidar_frame);
    
    // Store estimated pose (for backward compatibility)
    if (result) {
        // Get pose from the processed frame and convert SE3f to Matrix4f
        const auto& se3_pose = lidar_frame->get_pose();
        Eigen::Matrix4f pose_matrix = se3_pose.matrix();
        context.estimated_poses.push_back(pose_matrix);
    } else {
        spdlog::warn("[KittiPlayer] Frame processing failed for frame {}", context.current_idx);
        // Use last pose or identity if first frame
        if (!context.estimated_poses.empty()) {
            context.estimated_poses.push_back(context.estimated_poses.back());
        } else {
            context.estimated_poses.push_back(Eigen::Matrix4f::Identity());
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    return duration.count() / 1000.0; // Return milliseconds
}

void KittiPlayer::align_with_ground_truth(FrameContext& context, const Eigen::Matrix4f& gt_pose) {
    if (!context.transform_initialized && !context.estimated_poses.empty()) {
        // Calculate alignment transformation from first poses
        Eigen::Matrix4f estimated_first = context.estimated_poses[0];
        context.gt_to_estimated_transform = gt_pose.inverse()* estimated_first;
        context.transform_initialized = true;
        
        spdlog::info("[KittiPlayer] Trajectory alignment initialized");
    }
}

void KittiPlayer::update_viewer(viewer::PangolinViewer& viewer,
                               const FrameContext& context,
                               util::PointCloudPtr point_cloud) {
    if (context.estimated_poses.empty()) return;
    
    // Use the processed LidarFrame from context instead of creating a new one
    if (context.current_lidar_frame) {
        viewer.update_current_frame(context.current_lidar_frame);
        // Add frame to trajectory for dynamic pose updates
        viewer.add_trajectory_frame(context.current_lidar_frame);
    } else {
        spdlog::warn("[KittiPlayer] No processed LidarFrame available in context");
        // Fallback: create new frame with estimated pose
        Eigen::Matrix4f current_pose = context.estimated_poses.back();
        auto lidar_frame = std::make_shared<database::LidarFrame>(context.frame_index, context.timestamp, point_cloud);
        Sophus::SE3f se3_pose(current_pose);
        lidar_frame->set_pose(se3_pose);
        viewer.update_current_frame(lidar_frame);
        viewer.add_trajectory_frame(lidar_frame);
    }
    
    // Update map points from estimator
    if (m_estimator) {
        auto local_map = m_estimator->get_local_map();
        if (local_map && !local_map->empty()) {
            viewer.update_map_points(local_map);
        }
        
        // Check for new keyframes and add to viewer
        size_t current_keyframe_count = m_estimator->get_keyframe_count();
        if (current_keyframe_count > m_last_keyframe_count) {
            // Add new keyframes to viewer
            for (size_t i = m_last_keyframe_count; i < current_keyframe_count; ++i) {
                auto keyframe = m_estimator->get_keyframe(i);
                if (keyframe) {
                    viewer.add_keyframe(keyframe);
                    // spdlog::info("[KittiPlayer] Added keyframe {} to viewer", keyframe->get_frame_id());
                }
            }
            m_last_keyframe_count = current_keyframe_count;
        }
        
        // Update last keyframe for local map visualization
        auto last_keyframe = m_estimator->get_keyframe(m_estimator->get_keyframe_count() - 1);
        if (last_keyframe) {
            viewer.update_last_keyframe(last_keyframe);
        }
        
        // Update VoxelMap pointer for surfel visualization
        viewer.update_voxel_map(m_estimator->get_voxel_map());
        
        // Update ICP debug clouds if available
        PointCloudConstPtr pre_icp_cloud, post_icp_cloud;
        m_estimator->get_debug_clouds(pre_icp_cloud, post_icp_cloud);
        if (pre_icp_cloud && post_icp_cloud) {
            viewer.update_icp_debug_clouds(pre_icp_cloud, post_icp_cloud);
        }
        
        // Update optimized trajectory for debugging (GREEN line)
        auto optimized_trajectory = m_estimator->get_optimized_trajectory();
        if (!optimized_trajectory.empty()) {
            viewer.update_optimized_trajectory(optimized_trajectory);
        }
    }
    
    // Update trajectory
    std::vector<Eigen::Vector3f> trajectory_positions;
    for (const auto& pose : context.estimated_poses) {
        trajectory_positions.push_back(pose.block<3,1>(0,3));
    }
    // Trajectory is updated via add_trajectory_pose above
    
    // Update ground truth trajectory (disabled)
    // GT visualization disabled
    
    // Tracking statistics and render - methods not available in current viewer
    // int total_points = static_cast<int>(point_cloud->size());
    // viewer.update_tracking_stats(context.processed_frames + 1, total_points, 
    //                            total_points, total_points, 100.0f, 0.0f);
    
    // viewer.render(); // Rendering handled by thread
}

bool KittiPlayer::handle_viewer_controls(viewer::PangolinViewer& viewer, FrameContext& context) {
    // Check for exit conditions
    if (viewer.should_close()) {
        spdlog::info("[KittiPlayer] User requested exit");
        return false;
    }
    
    // Process keyboard input and update context
    bool dummy_auto_play = context.auto_play;
    bool dummy_step_mode = context.step_mode;
    viewer.process_keyboard_input(dummy_auto_play, dummy_step_mode, context.advance_frame);
    
    // Update context based on viewer state
    context.auto_play = dummy_auto_play;
    context.step_mode = dummy_step_mode;
    
    return true;
}

void KittiPlayer::save_trajectory_kitti_format(const FrameContext& context,
                                              const std::string& output_path) {
    std::ofstream file(output_path);
    if (!file.is_open()) {
        spdlog::error("[KittiPlayer] Cannot create output file: {}", output_path);
        return;
    }
    
    // Use processed_frames_list and get_pose() for dynamic post-PGO poses
    for (const auto& frame : context.processed_frames_list) {
        Eigen::Matrix4f pose = frame->get_pose().matrix();
        file << pose_to_kitti_string(pose) << std::endl;
    }
    
    file.close();
    spdlog::info("[KittiPlayer] Saved trajectory in KITTI format: {}", output_path);
}

void KittiPlayer::save_trajectory_tum_format(const FrameContext& context,
                                            const std::string& output_path) {
    std::ofstream file(output_path);
    if (!file.is_open()) {
        spdlog::error("[KittiPlayer] Cannot create output file: {}", output_path);
        return;
    }
    
    // Use processed_frames_list and get_pose() for dynamic post-PGO poses
    for (size_t i = 0; i < context.processed_frames_list.size(); ++i) {
        Eigen::Matrix4f pose = context.processed_frames_list[i]->get_pose().matrix();
        
        Eigen::Vector3f translation = pose.block<3,1>(0,3);
        Eigen::Matrix3f rotation = pose.block<3,3>(0,0);
        Eigen::Quaternionf quat(rotation);
        
        double timestamp = static_cast<double>(i) * 0.1; // 10Hz
        
        file << std::fixed << std::setprecision(6) << timestamp << " "
             << std::setprecision(8)
             << translation.x() << " " << translation.y() << " " << translation.z() << " "
             << quat.x() << " " << quat.y() << " " << quat.z() << " " << quat.w() << std::endl;
    }
    
    file.close();
    spdlog::info("[KittiPlayer] Saved trajectory in TUM format: {}", output_path);
}

KittiPlayerResult::ErrorStats KittiPlayer::analyze_trajectory_errors(const FrameContext& context) {
    KittiPlayerResult::ErrorStats stats;
    
    if (context.gt_poses.empty() || context.processed_frames_list.empty()) {
        return stats;
    }
    
    // KITTI evaluation lengths in meters
    std::vector<float> lengths = {100, 200, 300, 400, 500, 600, 700, 800};
    size_t min_size = std::min(context.gt_poses.size(), context.processed_frames_list.size());
    
    std::vector<double> translation_errors_percent;
    std::vector<double> rotation_errors_deg_per_100m;
    std::vector<double> ate_errors;
    
    // Create poses dict-style for compatibility with KITTI evaluation
    // Use get_pose() to get dynamic poses after PGO
    std::map<int, Eigen::Matrix4f> poses_gt, poses_result;
    for (size_t i = 0; i < min_size; ++i) {
        poses_gt[i] = context.gt_poses[i];
        poses_result[i] = context.processed_frames_list[i]->get_pose().matrix();
    }
    
    // KITTI-style first frame alignment
    Eigen::Matrix4f gt_0 = poses_gt[0];
    Eigen::Matrix4f pred_0 = poses_result[0];
    for (auto& pair : poses_gt) {
        poses_gt[pair.first] = gt_0.inverse() * pair.second;
    }
    for (auto& pair : poses_result) {
        poses_result[pair.first] = pred_0.inverse() * pair.second;
    }
    
    // Optional: Apply scale optimization (like KITTI official evaluation)
    // Calculate scale factor
    double scale_factor = 1.0;
    std::vector<double> gt_distances, est_distances;
    for (size_t i = 1; i < min_size; ++i) {
        Eigen::Vector3f gt_pos_prev = poses_gt[i-1].block<3,1>(0,3);
        Eigen::Vector3f gt_pos_curr = poses_gt[i].block<3,1>(0,3);
        Eigen::Vector3f est_pos_prev = poses_result[i-1].block<3,1>(0,3);
        Eigen::Vector3f est_pos_curr = poses_result[i].block<3,1>(0,3);
        
        gt_distances.push_back((gt_pos_curr - gt_pos_prev).norm());
        est_distances.push_back((est_pos_curr - est_pos_prev).norm());
    }
    
    if (!gt_distances.empty() && !est_distances.empty()) {
        double sum_gt_est = 0.0, sum_est_est = 0.0;
        for (size_t i = 0; i < gt_distances.size(); ++i) {
            sum_gt_est += gt_distances[i] * est_distances[i];
            sum_est_est += est_distances[i] * est_distances[i];
        }
        if (sum_est_est > 1e-10) {
            scale_factor = sum_gt_est / sum_est_est;
        }
    }
    
    // Apply scale factor to estimated poses
    for (auto& pair : poses_result) {
        pair.second.block<3,1>(0,3) *= scale_factor;
    }
    
    // Calculate trajectory distances (KITTI style)
    std::vector<double> trajectory_distances = {0.0};
    for (size_t i = 1; i < poses_gt.size(); ++i) {
        Eigen::Vector3f pos_prev = poses_gt[i-1].block<3,1>(0,3);
        Eigen::Vector3f pos_curr = poses_gt[i].block<3,1>(0,3);
        double dist = (pos_curr - pos_prev).norm();
        trajectory_distances.push_back(trajectory_distances.back() + dist);
    }
    
    // KITTI evaluation: segment-based relative pose error  
    int step_size = 10; // Evaluate every 10 frames
    std::vector<std::vector<double>> sequence_errors; // [first_frame, rot_err, trans_err, length, speed]
    
    for (int first_frame = 0; first_frame < (int)min_size; first_frame += step_size) {
        for (float length : lengths) {
            // Find last frame for this segment length
            int last_frame = -1;
            for (size_t i = first_frame; i < trajectory_distances.size(); ++i) {
                if (trajectory_distances[i] > (trajectory_distances[first_frame] + length)) {
                    last_frame = i;
                    break;
                }
            }
            
            // Skip if segment not long enough or frames not available
            if (last_frame == -1 || last_frame >= (int)min_size) {
                continue;
            }
            
            // Compute rotational and translational errors (KITTI style)
            Eigen::Matrix4f pose_delta_gt = poses_gt[first_frame].inverse() * poses_gt[last_frame];
            Eigen::Matrix4f pose_delta_result = poses_result[first_frame].inverse() * poses_result[last_frame];
            Eigen::Matrix4f pose_error = pose_delta_result.inverse() * pose_delta_gt;
            
            // Calculate actual traveled distance (path length, not Euclidean)
            double actual_path_length = 0.0;
            for (int j = first_frame; j < last_frame; ++j) {
                Eigen::Vector3f pos_prev = poses_gt[j].block<3,1>(0,3);
                Eigen::Vector3f pos_curr = poses_gt[j+1].block<3,1>(0,3);
                actual_path_length += (pos_curr - pos_prev).norm();
            }
            
            // Rotation error (KITTI style)
            Eigen::Matrix3f R_error = pose_error.block<3,3>(0,0);
            float trace = R_error(0,0) + R_error(1,1) + R_error(2,2);
            float d = 0.5f * (trace - 1.0f);
            d = std::max(-1.0f, std::min(1.0f, d)); // Clamp to valid range
            float rotation_error_rad = std::acos(d);
            
            // Translation error (KITTI style)
            Eigen::Vector3f t_error = pose_error.block<3,1>(0,3);
            float translation_error_m = t_error.norm();
            
            // KITTI percentage calculation: error relative to actual path length
            double rotation_error_per_length = rotation_error_rad / actual_path_length;
            double translation_error_per_length = translation_error_m / actual_path_length;
            
            // Convert to KITTI format
            double rotation_error_deg_per_100m = (rotation_error_per_length * 180.0 / M_PI) * 100.0;
            double translation_error_percent = translation_error_per_length * 100.0;
            
            rotation_errors_deg_per_100m.push_back(rotation_error_deg_per_100m);
            translation_errors_percent.push_back(translation_error_percent);
            
            // Store for detailed analysis
            double speed = actual_path_length / (0.1 * (last_frame - first_frame + 1)); // Assume 10Hz
            sequence_errors.push_back({(double)first_frame, rotation_error_per_length, 
                                     translation_error_per_length, actual_path_length, speed});
        }
    }
    
    // Calculate ATE (Absolute Trajectory Error) for reference
    for (size_t i = 0; i < min_size; ++i) {
        // For ATE, use aligned poses directly
        Eigen::Vector3f gt_pos = poses_gt[i].block<3,1>(0,3);
        Eigen::Vector3f est_pos = poses_result[i].block<3,1>(0,3);
        double ate = (gt_pos - est_pos).norm();
        ate_errors.push_back(ate);
    }
    
    if (!translation_errors_percent.empty()) {
        stats.available = true;
        stats.total_frames = context.estimated_poses.size();
        stats.gt_poses_count = context.gt_poses.size();
        stats.total_frame_pairs = translation_errors_percent.size();
        
        // Calculate KITTI-style statistics (averages over all segment lengths)
        stats.translation_mean = std::accumulate(translation_errors_percent.begin(), translation_errors_percent.end(), 0.0) / translation_errors_percent.size();
        stats.rotation_mean = std::accumulate(rotation_errors_deg_per_100m.begin(), rotation_errors_deg_per_100m.end(), 0.0) / rotation_errors_deg_per_100m.size();
        
        // For compatibility with existing code, set RMSE to mean values
        stats.translation_rmse = stats.translation_mean;
        stats.rotation_rmse = stats.rotation_mean;
        
        // Calculate ATE statistics for reference
        if (!ate_errors.empty()) {
            stats.ate_mean = std::accumulate(ate_errors.begin(), ate_errors.end(), 0.0) / ate_errors.size();
            stats.ate_rmse = std::sqrt(std::accumulate(ate_errors.begin(), ate_errors.end(), 0.0,
                [](double sum, double err) { return sum + err * err; }) / ate_errors.size());
            
            std::vector<double> ate_sorted = ate_errors;
            std::sort(ate_sorted.begin(), ate_sorted.end());
            stats.ate_median = ate_sorted[ate_sorted.size() / 2];
            stats.ate_min = *std::min_element(ate_errors.begin(), ate_errors.end());
            stats.ate_max = *std::max_element(ate_errors.begin(), ate_errors.end());
        }
        
        // Debug information
        spdlog::info("[KITTI Eval Debug] Scale factor applied: {:.6f}", scale_factor);
        spdlog::info("[KITTI Eval Debug] Total segments evaluated: {}", translation_errors_percent.size());
        if (!translation_errors_percent.empty()) {
            double min_trans_err = *std::min_element(translation_errors_percent.begin(), translation_errors_percent.end());
            double max_trans_err = *std::max_element(translation_errors_percent.begin(), translation_errors_percent.end());
            spdlog::info("[KITTI Eval Debug] Translation error range: {:.2f}% - {:.2f}%", min_trans_err, max_trans_err);
        }
    }
    
    return stats;
}

KittiPlayerResult::VelocityStats KittiPlayer::analyze_velocity_statistics(const FrameContext& context) {
    KittiPlayerResult::VelocityStats stats;
    
    if (context.estimated_poses.size() < 2) {
        return stats;
    }
    
    std::vector<double> linear_velocities;
    std::vector<double> angular_velocities;
    
    double dt = 0.1; // 10Hz assumption
    
    for (size_t i = 1; i < context.estimated_poses.size(); ++i) {
        const auto& pose_prev = context.estimated_poses[i-1];
        const auto& pose_curr = context.estimated_poses[i];
        
        // Linear velocity
        Eigen::Vector3f pos_prev = pose_prev.block<3,1>(0,3);
        Eigen::Vector3f pos_curr = pose_curr.block<3,1>(0,3);
        double linear_vel = (pos_curr - pos_prev).norm() / dt;
        
        // Angular velocity
        Eigen::Matrix3f R_prev = pose_prev.block<3,3>(0,0);
        Eigen::Matrix3f R_curr = pose_curr.block<3,3>(0,0);
        Eigen::Matrix3f R_rel = R_prev.transpose() * R_curr;
        Eigen::AngleAxisf angle_axis(R_rel);
        double angular_vel = std::abs(angle_axis.angle()) / dt;
        
        linear_velocities.push_back(linear_vel);
        angular_velocities.push_back(angular_vel);
    }
    
    if (!linear_velocities.empty()) {
        stats.available = true;
        
        std::vector<double> linear_sorted = linear_velocities;
        std::vector<double> angular_sorted = angular_velocities;
        std::sort(linear_sorted.begin(), linear_sorted.end());
        std::sort(angular_sorted.begin(), angular_sorted.end());
        
        stats.linear_vel_mean = std::accumulate(linear_velocities.begin(), linear_velocities.end(), 0.0) / linear_velocities.size();
        stats.linear_vel_median = linear_sorted[linear_sorted.size() / 2];
        stats.linear_vel_min = *std::min_element(linear_velocities.begin(), linear_velocities.end());
        stats.linear_vel_max = *std::max_element(linear_velocities.begin(), linear_velocities.end());
        
        stats.angular_vel_mean = std::accumulate(angular_velocities.begin(), angular_velocities.end(), 0.0) / angular_velocities.size();
        stats.angular_vel_median = angular_sorted[angular_sorted.size() / 2];
        stats.angular_vel_min = *std::min_element(angular_velocities.begin(), angular_velocities.end());
        stats.angular_vel_max = *std::max_element(angular_velocities.begin(), angular_velocities.end());
    }
    
    return stats;
}

void KittiPlayer::save_statistics(const KittiPlayerResult& result,
                                 const std::string& output_path) {
    std::ofstream file(output_path);
    if (!file.is_open()) {
        spdlog::error("[KittiPlayer] Cannot create statistics file: {}", output_path);
        return;
    }
    
    file << "════════════════════════════════════════════════════════════════════\n";
    file << "                          KITTI STATISTICS                          \n";
    file << "════════════════════════════════════════════════════════════════════\n\n";
    
    // Timing statistics
    file << "                          TIMING ANALYSIS                           \n";
    file << "════════════════════════════════════════════════════════════════════\n";
    file << " Total Frames Processed: " << result.processed_frames << "\n";
    file << " Average Processing Time: " << std::fixed << std::setprecision(2) 
         << result.average_processing_time_ms << "ms\n";
    double fps = 1000.0 / result.average_processing_time_ms;
    file << " Average Frame Rate: " << std::fixed << std::setprecision(1) << fps << "fps\n\n";
    
    // Error statistics
    if (result.error_stats.available) {
        file << "               KITTI TRAJECTORY EVALUATION              \n";
        file << "════════════════════════════════════════════════════════════════════\n";
        file << " Total Frames: " << result.error_stats.total_frames << "\n";
        file << " GT Poses: " << result.error_stats.gt_poses_count << "\n";
        file << " Evaluated Segments: " << result.error_stats.total_frame_pairs << "\n\n";
        
        file << "                     KITTI EVALUATION METRICS                    \n";
        file << " Trans. err. (%)    :" << std::setw(10) << std::fixed << std::setprecision(2) 
             << result.error_stats.translation_mean << "%\n";
        file << " Rot. err. (°/100m) :" << std::setw(10) << std::fixed << std::setprecision(2) 
             << result.error_stats.rotation_mean << "°/100m\n\n";
        
        file << "                     ABSOLUTE TRAJECTORY ERROR (ATE) - Reference                    \n";
        file << " RMSE      :" << std::setw(10) << std::fixed << std::setprecision(4) 
             << result.error_stats.ate_rmse << "m\n";
        file << " Mean      :" << std::setw(10) << std::fixed << std::setprecision(4) 
             << result.error_stats.ate_mean << "m\n";
        file << " Median    :" << std::setw(10) << std::fixed << std::setprecision(4) 
             << result.error_stats.ate_median << "m\n";
        file << " Min       :" << std::setw(10) << std::fixed << std::setprecision(4) 
             << result.error_stats.ate_min << "m\n";
        file << " Max       :" << std::setw(10) << std::fixed << std::setprecision(4) 
             << result.error_stats.ate_max << "m\n\n";
    }
    
    // Velocity statistics
    if (result.velocity_stats.available) {
        file << "                          VELOCITY ANALYSIS                         \n";
        file << "════════════════════════════════════════════════════════════════════\n";
        file << "                        LINEAR VELOCITY (m/s)                       \n";
        file << " Mean      :" << std::setw(10) << std::fixed << std::setprecision(4) 
             << result.velocity_stats.linear_vel_mean << "m/s\n";
        file << " Median    :" << std::setw(10) << std::fixed << std::setprecision(4) 
             << result.velocity_stats.linear_vel_median << "m/s\n";
        file << " Min       :" << std::setw(10) << std::fixed << std::setprecision(4) 
             << result.velocity_stats.linear_vel_min << "m/s\n";
        file << " Max       :" << std::setw(10) << std::fixed << std::setprecision(4) 
             << result.velocity_stats.linear_vel_max << "m/s\n\n";
        
        file << "                       ANGULAR VELOCITY (rad/s)                     \n";
        file << " Mean      :" << std::setw(10) << std::fixed << std::setprecision(4) 
             << result.velocity_stats.angular_vel_mean << "rad/s\n";
        file << " Median    :" << std::setw(10) << std::fixed << std::setprecision(4) 
             << result.velocity_stats.angular_vel_median << "rad/s\n";
        file << " Min       :" << std::setw(10) << std::fixed << std::setprecision(4) 
             << result.velocity_stats.angular_vel_min << "rad/s\n";
        file << " Max       :" << std::setw(10) << std::fixed << std::setprecision(4) 
             << result.velocity_stats.angular_vel_max << "rad/s\n\n";
    }
    
    file << "════════════════════════════════════════════════════════════════════\n";
    file.close();
    
    spdlog::info("[KittiPlayer] Saved statistics to: {}", output_path);
}

std::vector<std::string> KittiPlayer::get_bin_files(const std::string& directory_path) {
    std::vector<std::string> bin_files;
    
    try {
        for (const auto& entry : std::filesystem::directory_iterator(directory_path)) {
            if (entry.is_regular_file() && entry.path().extension() == ".bin") {
                bin_files.push_back(entry.path().filename().string());
            }
        }
        
        // Sort numerically
        std::sort(bin_files.begin(), bin_files.end());
        
    } catch (const std::filesystem::filesystem_error& e) {
        spdlog::error("[KittiPlayer] Filesystem error: {}", e.what());
    }
    
    return bin_files;
}

Eigen::Matrix4f KittiPlayer::parse_kitti_pose(const std::string& line) {
    std::istringstream iss(line);
    std::vector<float> values;
    float value;
    
    while (iss >> value) {
        values.push_back(value);
    }
    
    if (values.size() != 12) {
        spdlog::warn("[KittiPlayer] Invalid KITTI pose line, using identity");
        return Eigen::Matrix4f::Identity();
    }
    
    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    pose(0, 0) = values[0];  pose(0, 1) = values[1];  pose(0, 2) = values[2];  pose(0, 3) = values[3];
    pose(1, 0) = values[4];  pose(1, 1) = values[5];  pose(1, 2) = values[6];  pose(1, 3) = values[7];
    pose(2, 0) = values[8];  pose(2, 1) = values[9];  pose(2, 2) = values[10]; pose(2, 3) = values[11];
    
    return pose;
}

std::string KittiPlayer::pose_to_kitti_string(const Eigen::Matrix4f& pose) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(9);
    
    // Apply LiDAR to camera coordinate transformation
    // KITTI LiDAR to camera transformation matrix
    Eigen::Matrix4f T_lidar_to_cam;
    T_lidar_to_cam << 0.0f, -1.0f,  0.0f, 0.0f,
                      0.0f,  0.0f, -1.0f, 0.0f,
                      1.0f,  0.0f,  0.0f, 0.0f,
                      0.0f,  0.0f,  0.0f, 1.0f;
    
    Eigen::Matrix4f T_cam_to_lidar = T_lidar_to_cam.inverse();
    Eigen::Matrix4f converted_pose = T_lidar_to_cam * pose * T_cam_to_lidar;
    
    oss << converted_pose(0, 0) << " " << converted_pose(0, 1) << " " << converted_pose(0, 2) << " " << converted_pose(0, 3) << " "
        << converted_pose(1, 0) << " " << converted_pose(1, 1) << " " << converted_pose(1, 2) << " " << converted_pose(1, 3) << " "
        << converted_pose(2, 0) << " " << converted_pose(2, 1) << " " << converted_pose(2, 2) << " " << converted_pose(2, 3);
    
    return oss.str();
}

KittiPlayer::TrajectoryErrors KittiPlayer::calculate_trajectory_errors(const std::string& trajectory_file) {
    TrajectoryErrors errors;
    
    try {
        // Get ground truth path
        const auto& sys_config = util::config();
        std::string gt_file = sys_config.ground_truth_directory + "/" + sys_config.kitti_sequence + ".txt";
        
        // Load trajectory poses
        std::vector<Eigen::Matrix4f> estimated_poses;
        std::vector<Eigen::Matrix4f> gt_poses;
        
        // Load estimated poses
        std::ifstream est_file(trajectory_file);
        if (!est_file.is_open()) {
            throw std::runtime_error("Cannot open trajectory file: " + trajectory_file);
        }
        
        std::string line;
        while (std::getline(est_file, line)) {
            if (!line.empty()) {
                estimated_poses.push_back(parse_kitti_pose(line));
            }
        }
        est_file.close();
        
        // Load ground truth poses
        std::ifstream gt_file_stream(gt_file);
        if (!gt_file_stream.is_open()) {
            throw std::runtime_error("Cannot open ground truth file: " + gt_file);
        }
        
        while (std::getline(gt_file_stream, line)) {
            if (!line.empty()) {
                gt_poses.push_back(parse_kitti_pose(line));
            }
        }
        gt_file_stream.close();
        
        if (estimated_poses.empty() || gt_poses.empty()) {
            throw std::runtime_error("Empty trajectory data");
        }
        
        size_t min_size = std::min(estimated_poses.size(), gt_poses.size());
        
        // Align to first frame
        if (min_size > 0) {
            Eigen::Matrix4f est_0_inv = estimated_poses[0].inverse();
            Eigen::Matrix4f gt_0_inv = gt_poses[0].inverse();
            
            for (size_t i = 0; i < min_size; ++i) {
                estimated_poses[i] = est_0_inv * estimated_poses[i];
                gt_poses[i] = gt_0_inv * gt_poses[i];
            }
        }
        
        // Calculate simple trajectory errors (simplified KITTI evaluation)
        std::vector<double> trans_errors;
        std::vector<double> rot_errors;
        
        // Use fixed segments for evaluation (similar to KITTI evaluation)
        std::vector<int> lengths = {100, 200, 300, 400, 500, 600, 700, 800};
        int step_size = 10;
        
        for (size_t first_frame = 0; first_frame < min_size; first_frame += step_size) {
            for (int length : lengths) {
                // Find last frame at required distance
                double dist = 0.0;
                size_t last_frame = first_frame;
                
                for (size_t i = first_frame + 1; i < min_size; ++i) {
                    Eigen::Vector3f diff = gt_poses[i].block<3,1>(0,3) - gt_poses[i-1].block<3,1>(0,3);
                    dist += diff.norm();
                    if (dist >= length) {
                        last_frame = i;
                        break;
                    }
                }
                
                if (last_frame <= first_frame || last_frame >= min_size) continue;
                
                // Calculate relative transformations
                Eigen::Matrix4f gt_delta = gt_poses[first_frame].inverse() * gt_poses[last_frame];
                Eigen::Matrix4f est_delta = estimated_poses[first_frame].inverse() * estimated_poses[last_frame];
                Eigen::Matrix4f error_pose = est_delta.inverse() * gt_delta;
                
                // Translation error
                double trans_error = error_pose.block<3,1>(0,3).norm() / length;
                trans_errors.push_back(trans_error);
                
                // Rotation error  
                Eigen::Matrix3f R = error_pose.block<3,3>(0,0);
                double trace = R.trace();
                double rot_error = std::acos(std::max(-1.0, std::min(1.0, (trace - 1.0) / 2.0))) / length;
                rot_errors.push_back(rot_error);
            }
        }
        
        // Calculate average errors
        if (!trans_errors.empty()) {
            errors.translation_error = std::accumulate(trans_errors.begin(), trans_errors.end(), 0.0) / trans_errors.size();
        }
        if (!rot_errors.empty()) {
            errors.rotation_error = std::accumulate(rot_errors.begin(), rot_errors.end(), 0.0) / rot_errors.size();
        }
        
    } catch (const std::exception& e) {
        spdlog::warn("[KittiPlayer] Error calculating trajectory errors: {}", e.what());
    }
    
    return errors;
}

Eigen::Matrix4f KittiPlayer::calculate_alignment_transform(const std::vector<Eigen::Matrix4f>& estimated_poses,
                                                          const std::vector<Eigen::Matrix4f>& gt_poses) {
    if (estimated_poses.empty() || gt_poses.empty()) {
        return Eigen::Matrix4f::Identity();
    }
    
    // Use first pose for alignment
    return estimated_poses[0] * gt_poses[0].inverse();
}

} // namespace app
} // namespace lidar_odometry
