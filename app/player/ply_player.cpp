/**
 * @file      ply_player.cpp
 * @brief     PLY dataset player implementation (based on KITTI player)
 * @author    Seungwon Choi
 * @date      2025-10-07
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "ply_player.h"

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
#include <cstring>
#include <sys/resource.h>

#include "util/LogUtils.h"

#include <util/ConfigUtils.h>
#include <util/PointCloudUtils.h>
#include <util/TypeUtils.h>
#include <processing/Estimator.h>
#include <database/LidarFrame.h>
#include <viewer/PangolinViewer.h>

namespace lidar_odometry {
namespace app {

PLYPlayerResult PLYPlayer::run(const PLYPlayerConfig& config) {
    PLYPlayerResult result;
    
    try {
        // 1. Load configuration
        util::ConfigManager::instance().load_from_file(config.config_path);
        const auto& system_config = util::ConfigManager::instance().get_config();
        LOG_INFO("[PLYPlayer] Successfully loaded configuration from: {}", config.config_path);


        LOG_INFO("[PLYPlayer] Dataset path: {}", config.dataset_path);
        
        // 2. Load dataset
        auto point_cloud_data = load_ply_point_cloud_list(config.dataset_path, 
                                                         config.start_frame, 
                                                         config.end_frame, 
                                                         config.frame_skip);
        if (point_cloud_data.empty()) {
            result.error_message = "No PLY files found in dataset";
            return result;
        }
        
        LOG_INFO("[PLYPlayer] Loaded {} PLY files", point_cloud_data.size());
        
        // 3. Initialize systems
        auto viewer = initialize_viewer(config);
        initialize_estimator(system_config);
        
        // 4. Process frames
        PLYFrameContext context;
        context.step_mode = config.step_mode;
        context.auto_play = !config.step_mode;
        
        // LOG_INFO("[PLYPlayer] Processing frames {} to {} (step mode: {})", 
        //             0, point_cloud_data.size(), config.step_mode ? "enabled" : "disabled");
        
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
            
            // Process frame if ready
            if (should_process_frame) {
                const auto& current_data = point_cloud_data[context.current_idx];
                
                // LOG_INFO("[PLYPlayer] Processing frame {}/{}: {}", 
                //            context.current_idx + 1, point_cloud_data.size(), current_data.filename);
                
                // Load point cloud
                auto point_cloud = load_ply_point_cloud(current_data.full_path);
                if (!point_cloud || point_cloud->empty()) {
                    LOG_ERROR("[PLYPlayer] Failed to load or empty point cloud: {}", current_data.full_path);
                    context.current_idx++;
                    continue;
                }
                
                // Process through pipeline
                auto processing_time = process_single_frame(point_cloud, context);
                result.frame_processing_times.push_back(processing_time);
                
                // Update viewer
                if (viewer) {
                    update_viewer(*viewer, context, point_cloud);
                }
                
                // Update context
                context.current_idx++;
                context.processed_frames++;
                context.frame_index = current_data.frame_id;
                context.timestamp = current_data.timestamp;
                
            }
        }
        
        // 5. Finalize results
        result.processed_frames = context.processed_frames;
        result.success = true;
        
        if (!result.frame_processing_times.empty()) {
            result.average_processing_time_ms = 
                std::accumulate(result.frame_processing_times.begin(), 
                               result.frame_processing_times.end(), 0.0) / 
                result.frame_processing_times.size();
        }
        
        // 6. Save results
        if (config.save_trajectory && !context.estimated_poses.empty()) {
            std::filesystem::create_directories(config.output_directory);
            
            std::string trajectory_file = config.output_directory + "/trajectory." + config.trajectory_format;
            
            if (config.trajectory_format == "tum") {
                save_trajectory_tum_format(context, trajectory_file);
            } else {
                save_trajectory_kitti_format(context, trajectory_file);
            }
            
            LOG_INFO("[PLYPlayer] Saved trajectory to: {}", trajectory_file);
        }
        
        LOG_INFO("[PLYPlayer] Processing completed successfully!");
        LOG_INFO("[PLYPlayer] Processed {} frames in {:.2f}s", 
                    result.processed_frames, 
                    result.average_processing_time_ms * result.processed_frames / 1000.0);
        
        // Wait for viewer finish if enabled
        if (viewer) {
            LOG_INFO("[PLYPlayer] Processing completed! Close viewer to exit.");
            while (!viewer->should_close()) {
                // Render runs in thread, no need for explicit call
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
            }
        }
        
    } catch (const std::exception& e) {
        result.error_message = e.what();
        LOG_ERROR("[PLYPlayer] Exception: {}", e.what());
    }
    
    return result;
}

PLYPlayerResult PLYPlayer::run_from_yaml(const std::string& config_path) {
    PLYPlayerConfig config;
    config.config_path = config_path;
    
    try {
        // Load YAML configuration
        util::ConfigManager::instance().load_from_file(config_path);
        const auto& system_config = util::ConfigManager::instance().get_config();
        
        // Auto-construct dataset path based on sequence
        config.dataset_path = system_config.data_directory + "/" + system_config.kitti_sequence;
        config.output_directory = system_config.output_directory;
        
        // Set other configuration from YAML
        config.enable_viewer = system_config.player_enable_viewer;
        config.enable_statistics = system_config.player_enable_statistics;
        config.enable_console_statistics = system_config.player_enable_console_statistics;
        config.step_mode = system_config.player_step_mode;
        config.viewer_width = system_config.viewer_width;
        config.viewer_height = system_config.viewer_height;
        
        // Use default values for player settings
        config.save_trajectory = true;
        config.trajectory_format = "tum";
        
        LOG_INFO("[PLYPlayer] Configuration from YAML:");
        LOG_INFO("  Dataset path: {}", config.dataset_path);
        LOG_INFO("  Sequence: {}", system_config.kitti_sequence);
        LOG_INFO("  Viewer enabled: {}", config.enable_viewer);
        LOG_INFO("  Step mode: {}", config.step_mode);
        
    } catch (const std::exception& e) {
        PLYPlayerResult result;
        result.error_message = "Failed to load YAML configuration: " + std::string(e.what());
        return result;
    }
    
    return run(config);
}

std::vector<PLYPointCloudData> PLYPlayer::load_ply_point_cloud_list(const std::string& dataset_path,
                                                                    int start_frame,
                                                                    int end_frame,
                                                                    int frame_skip) {
    std::vector<PLYPointCloudData> ply_data;
    
    auto ply_files = get_ply_files(dataset_path);
    if (ply_files.empty()) {
        LOG_ERROR("[PLYPlayer] No PLY files found in: {}", dataset_path);
        return ply_data;
    }
    
    LOG_INFO("[PLYPlayer] Found {} PLY files in dataset", ply_files.size());
    
    // Apply frame filtering
    for (size_t i = 0; i < ply_files.size(); ++i) {
        int frame_id = extract_frame_number(ply_files[i]);
        
        // Apply start/end frame filtering
        if (start_frame >= 0 && frame_id < start_frame) continue;
        if (end_frame >= 0 && frame_id > end_frame) break;
        
        // Apply frame skip
        if ((frame_id - start_frame) % frame_skip != 0) continue;
        
        PLYPointCloudData data;
        data.frame_id = frame_id;
        data.filename = ply_files[i];
        data.full_path = dataset_path;
        if (data.full_path.back() != '/') {
            data.full_path += "/";
        }
        data.full_path += ply_files[i];
        
        // Generate synthetic timestamp based on frame ID (10Hz)
        data.timestamp = static_cast<long long>(frame_id * 100000000LL); // 100ms intervals
        
        ply_data.push_back(data);
    }
    
    LOG_INFO("[PLYPlayer] Selected {} frames for processing", ply_data.size());
    return ply_data;
}

std::shared_ptr<lidar_odometry::util::PointCloud> PLYPlayer::load_ply_point_cloud(const std::string& ply_file_path) {
    auto cloud = std::make_shared<util::PointCloud>();
    
    // Check if file exists
    if (!std::filesystem::exists(ply_file_path)) {
        LOG_ERROR("[PLYPlayer] PLY file does not exist: {}", ply_file_path);
        return cloud;
    }
    
    // Parse PLY header
    size_t vertex_count;
    std::vector<PLYPropertyInfo> properties;
    bool is_binary;
    
    if (!parse_ply_header(ply_file_path, vertex_count, properties, is_binary)) {
        LOG_ERROR("[PLYPlayer] Failed to parse PLY header: {}", ply_file_path);
        return cloud;
    }
    
    // Find x, y, z property indices
    int x_idx = -1, y_idx = -1, z_idx = -1;
    for (size_t i = 0; i < properties.size(); ++i) {
        if (properties[i].name == "x") x_idx = i;
        else if (properties[i].name == "y") y_idx = i;
        else if (properties[i].name == "z") z_idx = i;
    }
    
    if (x_idx == -1 || y_idx == -1 || z_idx == -1) {
        LOG_ERROR("[PLYPlayer] PLY file missing x, y, or z coordinates: {}", ply_file_path);
        return cloud;
    }
    
    std::ifstream file(ply_file_path, is_binary ? std::ios::binary : std::ios::in);
    if (!file.is_open()) {
        LOG_ERROR("[PLYPlayer] Failed to open PLY file: {}", ply_file_path);
        return cloud;
    }
    
    // Skip header
    std::string line;
    while (std::getline(file, line)) {
        if (line == "end_header") break;
    }
    
    cloud->reserve(vertex_count);
    
    if (is_binary) {
        // Calculate total bytes per vertex
        size_t bytes_per_vertex = 0;
        for (const auto& prop : properties) {
            bytes_per_vertex += prop.byte_size;
        }
        
        // Read binary data
        std::vector<char> vertex_buffer(bytes_per_vertex);
        for (size_t i = 0; i < vertex_count; ++i) {
            file.read(vertex_buffer.data(), bytes_per_vertex);
            
            if (!file.good()) break;
            
            // Extract x, y, z values
            float x, y, z;
            size_t offset = 0;
            for (size_t prop_idx = 0; prop_idx < properties.size(); ++prop_idx) {
                if (prop_idx == x_idx) {
                    std::memcpy(&x, vertex_buffer.data() + offset, sizeof(float));
                } else if (prop_idx == y_idx) {
                    std::memcpy(&y, vertex_buffer.data() + offset, sizeof(float));
                } else if (prop_idx == z_idx) {
                    std::memcpy(&z, vertex_buffer.data() + offset, sizeof(float));
                }
                offset += properties[prop_idx].byte_size;
            }
            
            cloud->push_back(x, y, z);
        }
    } else {
        // Read ASCII data
        for (size_t i = 0; i < vertex_count; ++i) {
            if (!std::getline(file, line)) break;
            
            std::istringstream iss(line);
            std::vector<float> values;
            float value;
            
            // Read all property values
            while (iss >> value) {
                values.push_back(value);
            }
            
            // Extract x, y, z
            if (values.size() >= properties.size()) {
                float x = values[x_idx];
                float y = values[y_idx];
                float z = values[z_idx];
                cloud->push_back(x, y, z);
            }
        }
    }
    
    file.close();
    
    LOG_DEBUG("[PLYPlayer] Loaded {} points from {}", cloud->size(), ply_file_path);
    return cloud;
}

bool PLYPlayer::parse_ply_header(const std::string& file_path, 
                                size_t& vertex_count, 
                                std::vector<PLYPropertyInfo>& properties,
                                bool& is_binary) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        return false;
    }
    
    vertex_count = 0;
    properties.clear();
    is_binary = false;
    
    std::string line;
    bool in_header = false;
    int x_index = -1, y_index = -1, z_index = -1;
    
    // Helper function to get byte size of property type
    auto get_type_size = [](const std::string& type) -> size_t {
        if (type == "char" || type == "uchar" || type == "int8" || type == "uint8") return 1;
        if (type == "short" || type == "ushort" || type == "int16" || type == "uint16") return 2;
        if (type == "int" || type == "uint" || type == "float" || type == "int32" || type == "uint32" || type == "float32") return 4;
        if (type == "double" || type == "float64") return 8;
        return 4; // default to float
    };
    
    while (std::getline(file, line)) {
        if (line == "ply") {
            in_header = true;
            continue;
        }
        
        if (!in_header) continue;
        
        if (line == "end_header") {
            break;
        }
        
        std::istringstream iss(line);
        std::string token;
        iss >> token;
        
        if (token == "format") {
            std::string format;
            iss >> format;
            is_binary = (format == "binary_little_endian" || format == "binary_big_endian");
        } else if (token == "element") {
            std::string element_type;
            iss >> element_type;
            if (element_type == "vertex") {
                iss >> vertex_count;
            }
        } else if (token == "property") {
            std::string type, name;
            iss >> type >> name;
            
            PLYPropertyInfo prop;
            prop.name = name;
            prop.type = type;
            prop.byte_size = get_type_size(type);
            
            properties.push_back(prop);
            
            // Track XYZ positions
            int current_idx = properties.size() - 1;
            if (name == "x") x_index = current_idx;
            else if (name == "y") y_index = current_idx;
            else if (name == "z") z_index = current_idx;
        }
    }
    
    file.close();
    
    // Validate that we have x, y, z properties
    if (x_index == -1 || y_index == -1 || z_index == -1) {
        LOG_ERROR("[PLYPlayer] PLY file missing x, y, or z coordinates");
        return false;
    }
    
    if (vertex_count == 0) {
        LOG_ERROR("[PLYPlayer] PLY file has no vertices");
        return false;
    }
    
    LOG_DEBUG("[PLYPlayer] PLY header parsed: {} vertices, {} properties, {} format", 
                  vertex_count, properties.size(), is_binary ? "binary" : "ASCII");
    
    return true;
}

std::shared_ptr<viewer::PangolinViewer> PLYPlayer::initialize_viewer(const PLYPlayerConfig& config) {
    if (!config.enable_viewer) {
        return nullptr;
    }
    
    auto viewer = std::make_shared<viewer::PangolinViewer>();
    if (viewer->initialize(config.viewer_width, config.viewer_height)) {
        LOG_INFO("[PLYPlayer] Viewer initialized successfully");
        
        // Wait for viewer to be ready
        while (!viewer->is_ready()) {
            // Render runs in thread, no need for explicit call
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
        }
        LOG_INFO("[PLYPlayer] Viewer is ready!");
        return viewer;
    } else {
        LOG_WARN("[PLYPlayer] Failed to initialize viewer");
        return nullptr;
    }
}

void PLYPlayer::initialize_estimator(const util::SystemConfig& config) {
    m_estimator = std::make_shared<processing::Estimator>(config);
    LOG_INFO("[PLYPlayer] Initialized LiDAR odometry estimator");
}

double PLYPlayer::process_single_frame(std::shared_ptr<lidar_odometry::util::PointCloud> point_cloud,
                                      PLYFrameContext& context) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Create LiDAR frame
        auto lidar_frame = std::make_shared<database::LidarFrame>(
            context.frame_index, 
            context.timestamp, 
            point_cloud
        );
        
        // Process through estimator
        if (m_estimator) {
            m_estimator->process_frame(lidar_frame);
            
            // Get current pose estimate and convert to Matrix4f
            auto current_pose_se3 = m_estimator->get_current_pose();
            Eigen::Matrix4f current_pose = current_pose_se3.Matrix();
            context.estimated_poses.push_back(current_pose);
            context.current_lidar_frame = lidar_frame;
        }
        
    } catch (const std::exception& e) {
        LOG_ERROR("[PLYPlayer] Error processing frame {}: {}", context.frame_index, e.what());
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    return duration.count() / 1000.0; // Convert to milliseconds
}

// KITTI player 스타일의 viewer 업데이트
void PLYPlayer::update_viewer(viewer::PangolinViewer& viewer,
                             const PLYFrameContext& context,
                             util::PointCloudPtr point_cloud) {
    if (context.estimated_poses.empty()) return;
    
    // Use the processed LidarFrame from context instead of creating a new one
    if (context.current_lidar_frame) {
        viewer.update_current_frame(context.current_lidar_frame);
        // Add frame to trajectory for dynamic pose updates
        viewer.add_trajectory_frame(context.current_lidar_frame);
    } else {
        LOG_WARN("[PLYPlayer] No processed LidarFrame available in context");
        // Fallback: create new frame with estimated pose
        Eigen::Matrix4f current_pose = context.estimated_poses.back();
        auto lidar_frame = std::make_shared<database::LidarFrame>(context.frame_index, context.timestamp, point_cloud);
        SE3f se3_pose(current_pose);
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
                    LOG_DEBUG("[PLYPlayer] Added keyframe {} to viewer", keyframe->get_frame_id());
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
    }
}

// KITTI player 스타일의 viewer 컨트롤
bool PLYPlayer::handle_viewer_controls(viewer::PangolinViewer& viewer, PLYFrameContext& context) {
    // Check for exit conditions
    if (viewer.should_close()) {
        LOG_INFO("[PLYPlayer] User requested exit");
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

void PLYPlayer::save_trajectory_kitti_format(const PLYFrameContext& context,
                                            const std::string& output_path) {
    std::ofstream file(output_path);
    if (!file.is_open()) {
        LOG_ERROR("[PLYPlayer] Failed to open trajectory file for writing: {}", output_path);
        return;
    }
    
    for (const auto& pose : context.estimated_poses) {
        file << pose_to_kitti_string(pose) << std::endl;
    }
    
    file.close();
}

void PLYPlayer::save_trajectory_tum_format(const PLYFrameContext& context,
                                          const std::string& output_path) {
    std::ofstream file(output_path);
    if (!file.is_open()) {
        LOG_ERROR("[PLYPlayer] Failed to open trajectory file for writing: {}", output_path);
        return;
    }
    
    for (size_t i = 0; i < context.estimated_poses.size(); ++i) {
        double timestamp = i * 0.1; // 10Hz
        file << pose_to_tum_string(context.estimated_poses[i], timestamp) << std::endl;
    }
    
    file.close();
}

std::vector<std::string> PLYPlayer::get_ply_files(const std::string& directory_path) {
    std::vector<std::string> ply_files;
    
    try {
        if (!std::filesystem::exists(directory_path)) {
            LOG_ERROR("[PLYPlayer] Directory does not exist: {}", directory_path);
            return ply_files;
        }
        
        for (const auto& entry : std::filesystem::directory_iterator(directory_path)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                if (filename.length() >= 4 && filename.substr(filename.length() - 4) == ".ply") {
                    ply_files.push_back(filename);
                }
            }
        }
        
        // Sort files numerically
        std::sort(ply_files.begin(), ply_files.end());
        
    } catch (const std::exception& e) {
        LOG_ERROR("[PLYPlayer] Error reading directory {}: {}", directory_path, e.what());
    }
    
    return ply_files;
}

std::string PLYPlayer::pose_to_kitti_string(const Eigen::Matrix4f& pose) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    
    // KITTI format: r11 r12 r13 t1 r21 r22 r23 t2 r31 r32 r33 t3
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 4; ++col) {
            if (row != 0 || col != 0) oss << " ";
            oss << pose(row, col);
        }
    }
    
    return oss.str();
}

std::string PLYPlayer::pose_to_tum_string(const Eigen::Matrix4f& pose, double timestamp) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    
    // Extract translation
    Eigen::Vector3f translation = pose.block<3,1>(0,3);
    
    // Extract rotation as quaternion
    Eigen::Matrix3f rotation = pose.block<3,3>(0,0);
    Eigen::Quaternionf quat(rotation);
    
    // TUM format: timestamp tx ty tz qx qy qz qw
    oss << timestamp << " "
        << translation.x() << " " << translation.y() << " " << translation.z() << " "
        << quat.x() << " " << quat.y() << " " << quat.z() << " " << quat.w();
    
    return oss.str();
}

int PLYPlayer::extract_frame_number(const std::string& filename) {
    // Extract number from filename like "000123.ply"
    std::string basename = filename;
    if (basename.length() >= 4 && basename.substr(basename.length() - 4) == ".ply") {
        basename = basename.substr(0, basename.length() - 4);
    }
    
    try {
        return std::stoi(basename);
    } catch (const std::exception&) {
        return 0;
    }
}

} // namespace app
} // namespace lidar_odometry