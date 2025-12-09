/**
 * @file      kitti_player.h
 * @brief     KITTI dataset player for LiDAR odometry pipeline
 * @author    Seungwon Choi
 * @date      2025-09-25
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
#include "../../src/util/MathUtils.h"
#include "../../src/util/PointCloudUtils.h"

// Forward declarations to avoid heavy includes
namespace lidar_slam {
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

namespace lidar_slam {
namespace app {

/**
 * @brief Point cloud data structure with timestamp
 */
struct PointCloudData {
    long long timestamp;           ///< Timestamp in nanoseconds
    std::string filename;          ///< Filename (e.g., "000001.bin")
    int frame_id;                  ///< Frame ID
};

/**
 * @brief Ground truth pose data
 */
struct GroundTruthPose {
    int frame_id;                  ///< Frame ID
    Eigen::Matrix4f pose;          ///< 4x4 transformation matrix
};

/**
 * @brief Configuration for KITTI player
 */
struct KittiPlayerConfig {
    std::string config_path;                    ///< Path to YAML configuration file
    std::string dataset_path;                   ///< Path to KITTI dataset
    std::string ground_truth_path;              ///< Path to ground truth poses
    bool enable_viewer = true;                  ///< Enable 3D visualization
    bool enable_statistics = true;              ///< Enable file statistics output
    bool enable_console_statistics = true;     ///< Enable console statistics output
    bool step_mode = false;                     ///< Step-by-step processing mode
    int start_frame = 0;                        ///< Start frame index
    int end_frame = -1;                         ///< End frame index (-1 for all)
    int frame_skip = 1;                         ///< Process every N frames
    int viewer_width = 1920;                    ///< Viewer window width
    int viewer_height = 1080;                   ///< Viewer window height
};

/**
 * @brief Result structure containing processing statistics
 */
struct KittiPlayerResult {
    bool success = false;
    size_t processed_frames = 0;
    double average_processing_time_ms = 0.0;
    std::vector<double> frame_processing_times;
    std::string error_message;
    
    // Error analysis results (compared to ground truth)
    struct ErrorStats {
        bool available = false;
        size_t total_frame_pairs = 0;
        size_t total_frames = 0;
        size_t gt_poses_count = 0;
        
        // Rotation error statistics (degrees)
        double rotation_rmse = 0.0;
        double rotation_mean = 0.0;
        double rotation_median = 0.0;
        double rotation_min = 0.0;
        double rotation_max = 0.0;
        
        // Translation error statistics (meters)
        double translation_rmse = 0.0;
        double translation_mean = 0.0;
        double translation_median = 0.0;
        double translation_min = 0.0;
        double translation_max = 0.0;
        
        // Absolute trajectory error (ATE)
        double ate_rmse = 0.0;
        double ate_mean = 0.0;
        double ate_median = 0.0;
        double ate_min = 0.0;
        double ate_max = 0.0;
    } error_stats;
    
    // Velocity analysis results
    struct VelocityStats {
        bool available = false;
        
        // Linear velocity statistics (m/s)
        double linear_vel_mean = 0.0;
        double linear_vel_median = 0.0;
        double linear_vel_min = 0.0;
        double linear_vel_max = 0.0;
        
        // Angular velocity statistics (rad/s)
        double angular_vel_mean = 0.0;
        double angular_vel_median = 0.0;
        double angular_vel_min = 0.0;
        double angular_vel_max = 0.0;
    } velocity_stats;
};

/**
 * @brief Frame processing context
 */
struct FrameContext {
    size_t current_idx = 0;
    size_t processed_frames = 0;
    size_t frame_index = 0;
    double timestamp = 0.0;
    std::vector<Eigen::Matrix4f> gt_poses;
    std::vector<Eigen::Matrix4f> estimated_poses;  // Deprecated: use processed_frames_list instead
    std::vector<std::shared_ptr<database::LidarFrame>> processed_frames_list; // Use get_pose() for dynamic updates
    Eigen::Matrix4f gt_to_estimated_transform = Eigen::Matrix4f::Identity();
    bool transform_initialized = false;
    
    // Current processed frame
    std::shared_ptr<database::LidarFrame> current_lidar_frame;
    
    // UI control
    bool auto_play = true;
    bool step_mode = false;    
    bool advance_frame = false;
};

/**
 * @brief KITTI Dataset Player class
 * 
 * Handles LiDAR odometry processing for KITTI dataset with optional 3D visualization
 * and ground truth comparison.
 */
class KittiPlayer {
public:
    /**
     * @brief Constructor
     */
    KittiPlayer() = default;
    
    /**
     * @brief Destructor
     */
    ~KittiPlayer() = default;

    /**
     * @brief Run the KITTI player with given configuration
     * @param config Player configuration
     * @return Processing result with statistics
     */
    KittiPlayerResult run(const KittiPlayerConfig& config);
    
    /**
     * @brief Run the KITTI player with YAML configuration file only
     * @param config_path Path to YAML configuration file
     * @return Processing result with statistics
     */
    KittiPlayerResult run_from_yaml(const std::string& config_path);

private:
    // === Data Loading ===
    
    /**
     * @brief Load point cloud file list from KITTI dataset
     * @param dataset_path Path to KITTI dataset
     * @param start_frame Start frame index
     * @param end_frame End frame index
     * @param frame_skip Frame skip interval
     * @return Vector of point cloud data
     */
    std::vector<PointCloudData> load_point_cloud_list(const std::string& dataset_path,
                                                      int start_frame,
                                                      int end_frame,
                                                      int frame_skip);
    
    /**
     * @brief Load single point cloud from KITTI bin file
     * @param dataset_path Path to dataset
     * @param filename Bin filename
     * @return Loaded point cloud
     */
    std::shared_ptr<lidar_slam::util::PointCloud> load_point_cloud(const std::string& dataset_path, 
                                                         const std::string& filename);
    
    // === System Initialization ===
    
    /**
     * @brief Initialize viewer if enabled
     * @param config Player configuration
     * @return Unique pointer to viewer (nullptr if disabled)
     */
    std::shared_ptr<viewer::PangolinViewer> initialize_viewer(const KittiPlayerConfig& config);
    
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
    double process_single_frame(std::shared_ptr<lidar_slam::util::PointCloud> point_cloud,
                               FrameContext& context);
    
    /**
     * @brief Align estimated trajectory with ground truth
     * @param context Frame processing context
     * @param gt_pose Ground truth pose for current frame
     */
    void align_with_ground_truth(FrameContext& context, const Eigen::Matrix4f& gt_pose);

    // === Viewer Updates ===
    
    /**
     * @brief Update viewer with current frame data
     * @param viewer Reference to viewer
     * @param context Frame processing context
     * @param point_cloud Current point cloud
     */
    void update_viewer(viewer::PangolinViewer& viewer,
                      const FrameContext& context,
                      util::PointCloudPtr point_cloud);
    
    /**
     * @brief Handle viewer UI controls
     * @param viewer Reference to viewer
     * @param context Frame processing context
     * @return True if should continue processing
     */
    bool handle_viewer_controls(viewer::PangolinViewer& viewer, FrameContext& context);

    // === Result Saving ===
    
    /**
     * @brief Save trajectory results in KITTI format
     * @param context Frame processing context
     * @param output_path Output file path
     */
    void save_trajectory_kitti_format(const FrameContext& context,
                                     const std::string& output_path);
    
    /**
     * @brief Save trajectory results in TUM format
     * @param context Frame processing context
     * @param output_path Output file path
     */
    void save_trajectory_tum_format(const FrameContext& context,
                                   const std::string& output_path);
    
    /**
     * @brief Analyze trajectory errors compared to ground truth
     * @param context Frame processing context
     * @return Error statistics
     */
    KittiPlayerResult::ErrorStats analyze_trajectory_errors(const FrameContext& context);
    
    /**
     * @brief Analyze velocity statistics from trajectory
     * @param context Frame processing context
     * @return Velocity statistics
     */
    KittiPlayerResult::VelocityStats analyze_velocity_statistics(const FrameContext& context);
    
    /**
     * @brief Save comprehensive statistics to file
     * @param result Player result with statistics
     * @param output_path Output file path
     */
    void save_statistics(const KittiPlayerResult& result,
                        const std::string& output_path);

    // === Utility Functions ===
    
    /**
     * @brief Get bin files from directory
     * @param directory_path Path to directory
     * @return Sorted vector of bin filenames
     */
    std::vector<std::string> get_bin_files(const std::string& directory_path);
    
    /**
     * @brief Parse KITTI pose from string line
     * @param line Line from KITTI pose file
     * @return 4x4 transformation matrix
     */
    Eigen::Matrix4f parse_kitti_pose(const std::string& line);
    
    /**
     * @brief Convert pose matrix to KITTI format string
     * @param pose 4x4 transformation matrix
     * @return KITTI format string
     */
    std::string pose_to_kitti_string(const Eigen::Matrix4f& pose);
    
    /**
     * @brief Calculate alignment transformation between two trajectories
     * @param estimated_poses Estimated trajectory
     * @param gt_poses Ground truth trajectory
     * @return Alignment transformation matrix
     */
    Eigen::Matrix4f calculate_alignment_transform(const std::vector<Eigen::Matrix4f>& estimated_poses,
                                                  const std::vector<Eigen::Matrix4f>& gt_poses);

    // === Quick Error Calculation ===
    
    /**
     * @brief Simple trajectory error structure for quick evaluation
     */
    struct TrajectoryErrors {
        double translation_error = 0.0;  ///< Translation error percentage
        double rotation_error = 0.0;     ///< Rotation error in radians
    };
    
    /**
     * @brief Calculate trajectory errors from saved KITTI format file
     * @param trajectory_file Path to saved trajectory file
     * @return Simple trajectory errors
     */
    TrajectoryErrors calculate_trajectory_errors(const std::string& trajectory_file);

private:
    std::shared_ptr<processing::Estimator> m_estimator;     ///< LiDAR odometry estimator
    std::vector<Eigen::Matrix4f> m_ground_truth_poses;      ///< Ground truth trajectory poses
    size_t m_last_keyframe_count = 0;                       ///< Track number of keyframes for viewer updates
};

} // namespace app
} // namespace lidar_slam
