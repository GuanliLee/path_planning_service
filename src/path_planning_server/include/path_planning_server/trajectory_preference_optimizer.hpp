#pragma once

#include <moveit_msgs/msg/constraints.hpp>
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/robot_trajectory/robot_trajectory.hpp>

#include <cstdint>

namespace path_planning_server
{
struct TrajectoryPreferenceSettings
{
  double time_limit{ 1.0 };
  std::int64_t max_iterations{ 20 };
  double step_size{ 0.01 };
  double finite_difference_step{ 0.0001 };
  double convergence_tolerance{ 0.0001 };
  double tcp_path_length_weight{ 1.0 };
  double tcp_x_axis_world_z_weight{ 0.2 };
  double joint_travel_weight{ 0.1 };
};

class TrajectoryPreferenceOptimizer
{
public:
  explicit TrajectoryPreferenceOptimizer(TrajectoryPreferenceSettings settings);

  robot_trajectory::RobotTrajectoryPtr optimize(
      const planning_scene::PlanningSceneConstPtr& planning_scene,
      const robot_trajectory::RobotTrajectory& seed,
      const moveit_msgs::msg::Constraints& path_constraints) const;

private:
  TrajectoryPreferenceSettings settings_;
};
}  // namespace path_planning_server
