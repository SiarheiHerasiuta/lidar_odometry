/**
 * @file      ply_player.h
 * @brief     PLY dataset     bool enable_viewer = true;                  ///< Enable 3D visualization
    bool enable_statistics = true;              ///< Enable file statistics output
    bool enable_console_statistics = true;     ///< Enable console statistics output
    bool step_mode = false;                     ///< Step-by-step processing mode
    int start_frame = 0;                        ///< Start frame index
    int end_frame = -1;                         ///< End frame index (-1 for all)
    int frame_skip = 1;                         ///< Process every N frames
    bool save_trajectory = true;                ///< Save trajectory file
    std::string trajectory_format = "tum";      ///< Trajectory format ("tum" or "kitti")
    int viewer_width = 3000;                    ///< Viewer window width
    int viewer_height = 2000;                   ///< Viewer window heightr LiDAR odometry pipeline (based on KITTI player)
 * @author    Seungwon Choi
 * @date      2025-10-07
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <Eigen/Dense>
#include "../../src/util/Types.h"

// Forward declarations to avoid heavy includes
namespace lidar_odometry {
    namespace processing {
        class Estimator;
    }
    namespace viewer {
        class PangolinViewer;
    }
    namespace util {
        struct SystemConfig;
        class PointCloud;
    }
    namespace database {
        class LidarFrame;
    }
}

namespace lidar_odometry {
namespace app {

/**
 * @brief PLY point cloud data structure with timestamp
 */
struct PLYPointCloudData {
    long long timestamp;           ///< Timestamp in nanoseconds
    std::string filename;          ///< Filename (e.g., "000001.ply")
    int frame_id;                  ///< Frame ID
    std::string full_path;         ///< Full path to PLY file
};

/**
 * @brief Configuration for PLY player
 */
struct PLYPlayerConfig {
    std::string config_path;                    ///< Path to YAML configuration file
    std::string dataset_path;                   ///< Path to PLY dataset directory
    std::string output_directory;               ///< Output directory for results
    bool enable_viewer = true;                  ///< Enable 3D visualization
    bool enable_statistics = true;              ///< Enable file statistics output
    bool enable_console_statistics = true;     ///< Enable console statistics output
    bool step_mode = false;                     ///< Step-by-step processing mode
    int start_frame = 0;                        ///< Start frame index
    int end_frame = -1;                         ///< End frame index (-1 for all)
    int frame_skip = 1;                         ///< Process every N frames
    bool save_trajectory = true;                ///< Save trajectory results
    std::string trajectory_format = "tum";      ///< Trajectory format ("tum" or "kitti")
    double playback_speed = 1.0;                ///< Playback speed multiplier
    int viewer_width = 3000;                    ///< Viewer window width
    int viewer_height = 2000;                   ///< Viewer window height
};

/**
 * @brief Result structure containing processing statistics
 */
struct PLYPlayerResult {
    bool success = false;
    size_t processed_frames = 0;
    double average_processing_time_ms = 0.0;
    std::vector<double> frame_processing_times;
    std::string error_message;
};

/**
 * @brief Frame processing context for PLY player
 */
struct PLYFrameContext {
    size_t current_idx = 0;
    size_t processed_frames = 0;
    size_t frame_index = 0;
    double timestamp = 0.0;
    std::vector<Eigen::Matrix4f> estimated_poses;
    
    // Current processed frame
    std::shared_ptr<database::LidarFrame> current_lidar_frame;
    
    // UI control
    bool auto_play = true;
    bool step_mode = false;    
    bool advance_frame = false;
};

/**
 * @brief PLY Dataset Player class (based on KITTI player)
 * 
 * Handles LiDAR odometry processing for PLY point cloud files with optional 3D visualization.
 */
class PLYPlayer {
public:
    /**
     * @brief Constructor
     */
    PLYPlayer() = default;
    
    /**
     * @brief Destructor
     */
    ~PLYPlayer() = default;

    /**
     * @brief Run the PLY player with given configuration
     * @param config Player configuration
     * @return Processing result with statistics
     */
    PLYPlayerResult run(const PLYPlayerConfig& config);
    
    /**
     * @brief Run the PLY player with YAML configuration file only
     * @param config_path Path to YAML configuration file
     * @return Processing result with statistics
     */
    PLYPlayerResult run_from_yaml(const std::string& config_path);

private:
    // === Data Loading ===
    
    /**
     * @brief Load PLY point cloud file list from directory
     * @param dataset_path Path to PLY dataset directory
     * @param start_frame Start frame index
     * @param end_frame End frame index
     * @param frame_skip Frame skip interval
     * @return Vector of PLY point cloud data
     */
    std::vector<PLYPointCloudData> load_ply_point_cloud_list(const std::string& dataset_path,
                                                             int start_frame,
                                                             int end_frame,
                                                             int frame_skip);
    
    /**
     * @brief Load single point cloud from PLY file
     * @param ply_file_path Full path to PLY file
     * @return Loaded point cloud
     */
    std::shared_ptr<lidar_odometry::util::PointCloud> load_ply_point_cloud(const std::string& ply_file_path);
    
    /**
     * @brief PLY property information
     */
    struct PLYPropertyInfo {
        std::string name;
        std::string type;
        size_t byte_size;
    };
    
    /**
     * @brief Parse PLY header to understand file structure
     * @param file_path Path to PLY file
     * @param vertex_count Output parameter for number of vertices
     * @param properties Output parameter for property list
     * @param is_binary Output parameter for binary format
     * @return True if header parsed successfully
     */
    bool parse_ply_header(const std::string& file_path, 
                         size_t& vertex_count, 
                         std::vector<PLYPropertyInfo>& properties,
                         bool& is_binary);

    // === System Initialization ===
    
    /**
     * @brief Initialize viewer if enabled
     * @param config Player configuration
     * @return Shared pointer to viewer (nullptr if disabled)
     */
    std::shared_ptr<viewer::PangolinViewer> initialize_viewer(const PLYPlayerConfig& config);
    
    /**
     * @brief Initialize estimator with configuration
     * @param config System configuration
     */
    void initialize_estimator(const util::SystemConfig& config);

    // === Frame Processing ===
    
    /**
     * @brief Process single frame through LiDAR odometry pipeline
     * @param point_cloud Input point cloud
     * @param context Frame processing context
     * @return Processing time in milliseconds
     */
    double process_single_frame(std::shared_ptr<lidar_odometry::util::PointCloud> point_cloud,
                               PLYFrameContext& context);

    // === Viewer Updates ===
    
    /**
     * @brief Update viewer with current frame data
     * @param viewer Reference to viewer
     * @param context Frame processing context
     * @param point_cloud Current point cloud
     */
    void update_viewer(viewer::PangolinViewer& viewer,
                      const PLYFrameContext& context,
                      util::PointCloudPtr point_cloud);
    
    /**
     * @brief Handle viewer UI controls
     * @param viewer Reference to viewer
     * @param context Frame processing context
     * @return True if should continue processing
     */
    bool handle_viewer_controls(viewer::PangolinViewer& viewer, PLYFrameContext& context);

    // === Result Saving ===
    
    /**
     * @brief Save trajectory results in KITTI format
     * @param context Frame processing context
     * @param output_path Output file path
     */
    void save_trajectory_kitti_format(const PLYFrameContext& context,
                                     const std::string& output_path);
    
    /**
     * @brief Save trajectory results in TUM format
     * @param context Frame processing context
     * @param output_path Output file path
     */
    void save_trajectory_tum_format(const PLYFrameContext& context,
                                   const std::string& output_path);

    // === Utility Functions ===
    
    /**
     * @brief Get PLY files from directory
     * @param directory_path Path to directory
     * @return Sorted vector of PLY filenames
     */
    std::vector<std::string> get_ply_files(const std::string& directory_path);
    
    /**
     * @brief Convert pose matrix to KITTI format string
     * @param pose 4x4 transformation matrix
     * @return KITTI format string
     */
    std::string pose_to_kitti_string(const Eigen::Matrix4f& pose);
    
    /**
     * @brief Convert pose matrix to TUM format string
     * @param pose 4x4 transformation matrix
     * @param timestamp Timestamp
     * @return TUM format string
     */
    std::string pose_to_tum_string(const Eigen::Matrix4f& pose, double timestamp);
    
    /**
     * @brief Extract frame number from PLY filename
     * @param filename PLY filename (e.g., "000123.ply")
     * @return Frame number
     */
    int extract_frame_number(const std::string& filename);

private:
    std::shared_ptr<processing::Estimator> m_estimator;     ///< LiDAR odometry estimator
    size_t m_last_keyframe_count = 0;                       ///< Last processed keyframe count
};

} // namespace app
} // namespace lidar_odometry