/**
 * @file      mid360_lidar_odometry.cpp
 * @brief     Main application for MID360 LiDAR odometry using PLY files
 * @author    Seungwon Choi
 * @date      2025-10-07
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include <iostream>
#include "util/LogUtils.h"


#include "player/ply_player.h"

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [config_file] [options]" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "Arguments:" << std::endl;
    std::cout << "  config_file    Path to YAML configuration file (default: ./config/mid360.yaml)" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --help, -h     Show this help message" << std::endl;
    std::cout << "  --step         Enable step-by-step processing mode" << std::endl;
    std::cout << "  --no-viewer    Disable 3D visualization" << std::endl;
    std::cout << "  --no-stats     Disable statistics output" << std::endl;
    std::cout << "  --start N      Start from frame N (default: 0)" << std::endl;
    std::cout << "  --end N        End at frame N (default: all frames)" << std::endl;
    std::cout << "  --skip N       Process every N-th frame (default: 1)" << std::endl;
    std::cout << "  --format F     Trajectory format: tum or kitti (default: tum)" << std::endl;
    std::cout << "  --output DIR   Output directory (default: ./results)" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << program_name << "                                    # Use default config" << std::endl;
    std::cout << "  " << program_name << " config/mid360.yaml                # Specify config file" << std::endl;
    std::cout << "  " << program_name << " --step --no-viewer                # Step mode without viewer" << std::endl;
    std::cout << "  " << program_name << " --start 100 --end 500 --skip 2    # Process frames 100-500, every 2nd frame" << std::endl;
    std::cout << "  " << program_name << " --format kitti                    # KITTI trajectory format" << std::endl;
}

int main(int argc, char** argv) {
    // Default configuration
    std::string config_path = "./config/mid360.yaml";
    lidar_odometry::app::PLYPlayerConfig player_config;
    player_config.config_path = config_path;
    player_config.enable_viewer = true;
    player_config.enable_statistics = true;
    player_config.enable_console_statistics = true;
    player_config.step_mode = false;
    player_config.start_frame = 0;
    player_config.end_frame = -1;
    player_config.frame_skip = 1;
    player_config.trajectory_format = "tum";
    player_config.output_directory = "./results";
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--step") {
            player_config.step_mode = true;
        } else if (arg == "--no-viewer") {
            player_config.enable_viewer = false;
        } else if (arg == "--no-stats") {
            player_config.enable_statistics = false;
            player_config.enable_console_statistics = false;
        } else if (arg == "--start" && i + 1 < argc) {
            player_config.start_frame = std::stoi(argv[++i]);
        } else if (arg == "--end" && i + 1 < argc) {
            player_config.end_frame = std::stoi(argv[++i]);
        } else if (arg == "--skip" && i + 1 < argc) {
            player_config.frame_skip = std::stoi(argv[++i]);
        } else if (arg == "--format" && i + 1 < argc) {
            std::string format = argv[++i];
            if (format == "tum" || format == "kitti") {
                player_config.trajectory_format = format;
            } else {
                LOG_ERROR("Invalid trajectory format: {}. Use 'tum' or 'kitti'", format);
                return 1;
            }
        } else if (arg == "--output" && i + 1 < argc) {
            player_config.output_directory = argv[++i];
        } else if (arg[0] != '-') {
            // Assume it's a config file path
            config_path = arg;
            player_config.config_path = config_path;
        } else {
            LOG_ERROR("Unknown argument: {}", arg);
            print_usage(argv[0]);
            return 1;
        }
    }
    
    // Print banner
    LOG_INFO("════════════════════════════════════════════════════════════════════");
    LOG_INFO("                    MID360 LiDAR Odometry System                    ");
    LOG_INFO("                         PLY File Player                           ");
    LOG_INFO("════════════════════════════════════════════════════════════════════");
    LOG_INFO("Configuration file: {}", config_path);
    LOG_INFO("Processing mode: {}", player_config.step_mode ? "Step-by-step" : "Continuous");
    LOG_INFO("3D Viewer: {}", player_config.enable_viewer ? "Enabled" : "Disabled");
    LOG_INFO("Statistics: {}", player_config.enable_statistics ? "Enabled" : "Disabled");
    
    if (player_config.start_frame > 0 || player_config.end_frame >= 0) {
        LOG_INFO("Frame range: {} to {}", 
                    player_config.start_frame, 
                    player_config.end_frame >= 0 ? std::to_string(player_config.end_frame) : "end");
    }
    
    if (player_config.frame_skip > 1) {
        LOG_INFO("Frame skip: Every {} frames", player_config.frame_skip);
    }
    
    LOG_INFO("Trajectory format: {}", player_config.trajectory_format);
    LOG_INFO("Output directory: {}", player_config.output_directory);
    LOG_INFO("");
    
    try {
        // Create and run PLY player
        lidar_odometry::app::PLYPlayer player;
        auto result = player.run_from_yaml(config_path);
        
        if (result.success) {
            LOG_INFO("");
            LOG_INFO("════════════════════════════════════════════════════════════════════");
            LOG_INFO("                        PROCESSING COMPLETED                        ");
            LOG_INFO("════════════════════════════════════════════════════════════════════");
            LOG_INFO(" Successfully processed {} frames", result.processed_frames);
            LOG_INFO(" Average processing time: {:.2f}ms", result.average_processing_time_ms);
            
            if (result.average_processing_time_ms > 0) {
                LOG_INFO(" Average frame rate: {:.1f}fps", 1000.0 / result.average_processing_time_ms);
            }
            
            LOG_INFO("");
            LOG_INFO(" Output files saved to: results directory");
            LOG_INFO(" - trajectory.{} (estimated trajectory)", player_config.trajectory_format);
            if (player_config.enable_statistics) {
                LOG_INFO(" - statistics.txt (detailed statistics)");
            }
            
            LOG_INFO("════════════════════════════════════════════════════════════════════");
            LOG_INFO("");
            LOG_INFO("To evaluate trajectory accuracy, use tools like:");
            LOG_INFO("  evo_traj {} trajectory.{} --plot --save_plot trajectory_plot", 
                        player_config.trajectory_format, player_config.trajectory_format);
            
            if (player_config.trajectory_format == "tum") {
                LOG_INFO("  evo_ape tum ground_truth.txt trajectory.tum --plot --save_plot ape_plot");
            } else {
                LOG_INFO("  evo_ape kitti ground_truth.txt trajectory.kitti --plot --save_plot ape_plot");
            }
            
            return 0;
        } else {
            LOG_ERROR("");
            LOG_ERROR("════════════════════════════════════════════════════════════════════");
            LOG_ERROR("                         PROCESSING FAILED                          ");
            LOG_ERROR("════════════════════════════════════════════════════════════════════");
            LOG_ERROR("Error: {}", result.error_message);
            LOG_ERROR("");
            LOG_ERROR("Common issues and solutions:");
            LOG_ERROR("1. Check if the PLY files exist in the dataset directory");
            LOG_ERROR("2. Verify the configuration file path and content");
            LOG_ERROR("3. Ensure sufficient disk space for output files");
            LOG_ERROR("4. Check file permissions for input and output directories");
            return 1;
        }
        
    } catch (const std::exception& e) {
        LOG_ERROR("");
        LOG_ERROR("════════════════════════════════════════════════════════════════════");
        LOG_ERROR("                           FATAL ERROR                             ");
        LOG_ERROR("════════════════════════════════════════════════════════════════════");
        LOG_ERROR("Fatal error: {}", e.what());
        LOG_ERROR("");
        LOG_ERROR("This is typically caused by:");
        LOG_ERROR("1. Missing dependencies or libraries");
        LOG_ERROR("2. Corrupted configuration file");
        LOG_ERROR("3. Invalid memory access or segmentation fault");
        LOG_ERROR("4. System resource limitations");
        LOG_ERROR("");
        LOG_ERROR("Try running with debug logging: export SPDLOG_LEVEL=debug");
        return 1;
    }
}