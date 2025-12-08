/**
 * @file      PoseGraphOptimizer.h
 * @brief     GTSAM ISAM2-based pose graph optimization for loop closure.
 * @author    Seungwon Choi
 * @date      2025-10-18
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#pragma once

#include "../util/Types.h"
#include <sophus/se3.hpp>
#include <memory>
#include <vector>
#include <map>
#include <set>
#include <mutex>

namespace gtsam {
    class NonlinearFactorGraph;
    class Values;
    class ISAM2;
}

namespace lidar_odometry {
namespace optimization {

class PoseGraphOptimizer {
public:
    using SE3f = Sophus::SE3f;
    
    PoseGraphOptimizer();
    ~PoseGraphOptimizer();
    
    // ===== Incremental API (LIO-SAM style) =====
    
    /**
     * @brief Add first keyframe with prior factor and run ISAM2 update
     * @param keyframe_id Keyframe ID
     * @param pose World pose of the keyframe (Twl)
     * @return true if update succeeded
     */
    bool add_first_keyframe(int keyframe_id, const SE3f& pose);
    
    /**
     * @brief Add keyframe with odometry constraint from previous keyframe and run ISAM2 update
     * @param prev_keyframe_id Previous keyframe ID
     * @param curr_keyframe_id Current keyframe ID  
     * @param curr_pose World pose of current keyframe (Twl)
     * @param relative_pose Relative pose from prev to curr (prev^-1 * curr)
     * @param odom_trans_noise Translation noise for odometry
     * @param odom_rot_noise Rotation noise for odometry
     * @return true if update succeeded
     */
    bool add_keyframe_with_odom(int prev_keyframe_id, int curr_keyframe_id,
                                const SE3f& curr_pose,
                                const SE3f& relative_pose,
                                double odom_trans_noise = 0.1,
                                double odom_rot_noise = 0.1);
    
    /**
     * @brief Add loop closure constraint and run ISAM2 update with extra iterations
     * @param from_keyframe_id Loop matched keyframe ID
     * @param to_keyframe_id Loop query (current) keyframe ID
     * @param relative_pose Relative pose from matched to current (matched^-1 * current)
     * @param loop_trans_noise Translation noise for loop closure
     * @param loop_rot_noise Rotation noise for loop closure
     * @return true if update succeeded
     */
    bool add_loop_and_optimize(int from_keyframe_id, int to_keyframe_id,
                               const SE3f& relative_pose,
                               double loop_trans_noise = 0.05,
                               double loop_rot_noise = 0.05);
    
    // ===== Query API =====
    
    bool get_optimized_pose(int keyframe_id, SE3f& optimized_pose) const;
    
    std::map<int, SE3f> get_all_optimized_poses() const;
    
    bool has_keyframe(int keyframe_id) const { 
        return m_keyframe_set.find(keyframe_id) != m_keyframe_set.end(); 
    }
    
    size_t get_keyframe_count() const { return m_keyframe_ids.size(); }
    
    size_t get_loop_closure_count() const { return m_loop_closure_count; }
    
    void clear();

private:
    mutable std::mutex m_mutex;  // Thread safety for all operations
    
    std::unique_ptr<gtsam::ISAM2> m_isam2;
    std::unique_ptr<gtsam::NonlinearFactorGraph> m_pending_graph;
    std::unique_ptr<gtsam::Values> m_pending_values;
    
    std::vector<int> m_keyframe_ids;
    std::set<int> m_keyframe_set;  // For O(1) lookup
    
    size_t m_loop_closure_count;
    size_t m_odometry_count;
    bool m_is_initialized;
};

} // namespace optimization
} // namespace lidar_odometry
