/**
 * @file      kitti_lidar_odometry.cpp
 * @brief     Main application for KITTI LiDAR odometry
 * @author    Seungwon Choi
 * @date      2025-09-25
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include <iostream>
#include "util/LogUtils.h"

#include "player/kitti_player.h"

int main(int argc, char** argv) {
    // Parse command line arguments
    std::string config_path = "./config/kitti.yaml";
    
    if (argc > 1) {
        config_path = argv[1];
    }
    
    LOG_INFO("════════════════════════════════════════════════════════════════════");
    LOG_INFO("                    KITTI LiDAR Odometry System                     ");
    LOG_INFO("════════════════════════════════════════════════════════════════════");
    LOG_INFO("Using configuration file: {}", config_path);
    LOG_INFO("");
    
    try {
        // Create and run KITTI player
        lidar_slam::app::KittiPlayer player;
        auto result = player.run_from_yaml(config_path);
        
        if (result.success) {
            LOG_INFO("");
            LOG_INFO("════════════════════════════════════════════════════════════════════");
            LOG_INFO("                        PROCESSING COMPLETED                        ");
            LOG_INFO("════════════════════════════════════════════════════════════════════");
            LOG_INFO(" Successfully processed {} frames", result.processed_frames);
            LOG_INFO(" Average processing time: {:.2f}ms", result.average_processing_time_ms);
            LOG_INFO(" Average frame rate: {:.1f}fps", 1000.0 / result.average_processing_time_ms);
            
            if (result.error_stats.available) {
                LOG_INFO("");
                LOG_INFO(" Trajectory Error Analysis:");
                LOG_INFO("   ATE RMSE: {:.4f}m", result.error_stats.ate_rmse);
                LOG_INFO("   ATE Mean: {:.4f}m", result.error_stats.ate_mean);
                LOG_INFO("   Rotation RMSE: {:.4f}°", result.error_stats.rotation_rmse);
                LOG_INFO("   Translation RMSE: {:.4f}m", result.error_stats.translation_rmse);
            }
            
            LOG_INFO("════════════════════════════════════════════════════════════════════");
            
            return 0;
        } else {
            LOG_ERROR("Processing failed: {}", result.error_message);
            return 1;
        }
        
    } catch (const std::exception& e) {
        LOG_ERROR("Fatal error: {}", e.what());
        return 1;
    }
}
