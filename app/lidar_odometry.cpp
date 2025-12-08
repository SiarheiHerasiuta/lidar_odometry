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
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

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
    // Initialize logging
    auto console = spdlog::stdout_color_mt("console");
    spdlog::set_default_logger(console);
    spdlog::set_level(spdlog::level::info);
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    
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
                spdlog::error("Invalid trajectory format: {}. Use 'tum' or 'kitti'", format);
                return 1;
            }
        } else if (arg == "--output" && i + 1 < argc) {
            player_config.output_directory = argv[++i];
        } else if (arg[0] != '-') {
            // Assume it's a config file path
            config_path = arg;
            player_config.config_path = config_path;
        } else {
            spdlog::error("Unknown argument: {}", arg);
            print_usage(argv[0]);
            return 1;
        }
    }
    
    // Print banner
    spdlog::info("════════════════════════════════════════════════════════════════════");
    spdlog::info("                    MID360 LiDAR Odometry System                    ");
    spdlog::info("                         PLY File Player                           ");
    spdlog::info("════════════════════════════════════════════════════════════════════");
    spdlog::info("Configuration file: {}", config_path);
    spdlog::info("Processing mode: {}", player_config.step_mode ? "Step-by-step" : "Continuous");
    spdlog::info("3D Viewer: {}", player_config.enable_viewer ? "Enabled" : "Disabled");
    spdlog::info("Statistics: {}", player_config.enable_statistics ? "Enabled" : "Disabled");
    
    if (player_config.start_frame > 0 || player_config.end_frame >= 0) {
        spdlog::info("Frame range: {} to {}", 
                    player_config.start_frame, 
                    player_config.end_frame >= 0 ? std::to_string(player_config.end_frame) : "end");
    }
    
    if (player_config.frame_skip > 1) {
        spdlog::info("Frame skip: Every {} frames", player_config.frame_skip);
    }
    
    spdlog::info("Trajectory format: {}", player_config.trajectory_format);
    spdlog::info("Output directory: {}", player_config.output_directory);
    spdlog::info("");
    
    try {
        // Create and run PLY player
        lidar_odometry::app::PLYPlayer player;
        auto result = player.run_from_yaml(config_path);
        
        if (result.success) {
            spdlog::info("");
            spdlog::info("════════════════════════════════════════════════════════════════════");
            spdlog::info("                        PROCESSING COMPLETED                        ");
            spdlog::info("════════════════════════════════════════════════════════════════════");
            spdlog::info(" Successfully processed {} frames", result.processed_frames);
            spdlog::info(" Average processing time: {:.2f}ms", result.average_processing_time_ms);
            
            if (result.average_processing_time_ms > 0) {
                spdlog::info(" Average frame rate: {:.1f}fps", 1000.0 / result.average_processing_time_ms);
            }
            
            spdlog::info("");
            spdlog::info(" Output files saved to: results directory");
            spdlog::info(" - trajectory.{} (estimated trajectory)", player_config.trajectory_format);
            if (player_config.enable_statistics) {
                spdlog::info(" - statistics.txt (detailed statistics)");
            }
            
            spdlog::info("════════════════════════════════════════════════════════════════════");
            spdlog::info("");
            spdlog::info("To evaluate trajectory accuracy, use tools like:");
            spdlog::info("  evo_traj {} trajectory.{} --plot --save_plot trajectory_plot", 
                        player_config.trajectory_format, player_config.trajectory_format);
            
            if (player_config.trajectory_format == "tum") {
                spdlog::info("  evo_ape tum ground_truth.txt trajectory.tum --plot --save_plot ape_plot");
            } else {
                spdlog::info("  evo_ape kitti ground_truth.txt trajectory.kitti --plot --save_plot ape_plot");
            }
            
            return 0;
        } else {
            spdlog::error("");
            spdlog::error("════════════════════════════════════════════════════════════════════");
            spdlog::error("                         PROCESSING FAILED                          ");
            spdlog::error("════════════════════════════════════════════════════════════════════");
            spdlog::error("Error: {}", result.error_message);
            spdlog::error("");
            spdlog::error("Common issues and solutions:");
            spdlog::error("1. Check if the PLY files exist in the dataset directory");
            spdlog::error("2. Verify the configuration file path and content");
            spdlog::error("3. Ensure sufficient disk space for output files");
            spdlog::error("4. Check file permissions for input and output directories");
            return 1;
        }
        
    } catch (const std::exception& e) {
        spdlog::error("");
        spdlog::error("════════════════════════════════════════════════════════════════════");
        spdlog::error("                           FATAL ERROR                             ");
        spdlog::error("════════════════════════════════════════════════════════════════════");
        spdlog::error("Fatal error: {}", e.what());
        spdlog::error("");
        spdlog::error("This is typically caused by:");
        spdlog::error("1. Missing dependencies or libraries");
        spdlog::error("2. Corrupted configuration file");
        spdlog::error("3. Invalid memory access or segmentation fault");
        spdlog::error("4. System resource limitations");
        spdlog::error("");
        spdlog::error("Try running with debug logging: export SPDLOG_LEVEL=debug");
        return 1;
    }
}