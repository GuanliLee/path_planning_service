#include "path_planning_server/trajectory_preference_adapter.hpp"

#include <moveit_msgs/msg/move_it_error_codes.hpp>
#include <pluginlib/class_list_macros.hpp>

#include <cstdint>
#include <string>

namespace path_planning_server
{
namespace
{
template <typename T>
T readParameter(const rclcpp::Node::SharedPtr& node, const std::string& name, const T& default_value)
{
  if (!node->has_parameter(name))
  {
    return node->declare_parameter<T>(name, default_value);
  }

  T value;
  node->get_parameter(name, value);
  return value;
}

std::string parameterName(const std::string& parameter_namespace, const std::string& relative_name)
{
  if (parameter_namespace.empty())
  {
    return relative_name;
  }
  return parameter_namespace + "." + relative_name;
}

void markPlanningFailed(planning_interface::MotionPlanResponse& response)
{
  response.error_code.val = moveit_msgs::msg::MoveItErrorCodes::PLANNING_FAILED;
  response.trajectory.reset();
}
}  // namespace

void TrajectoryPreferenceAdapter::initialize(const rclcpp::Node::SharedPtr& node,
                                             const std::string& parameter_namespace)
{
  node_ = node;
  const std::string prefix = parameterName(parameter_namespace, "trajectory_optimizer.preference_cost.");
  settings_.time_limit = readParameter<double>(node_, prefix + "time_limit", settings_.time_limit);
  settings_.max_iterations =
      readParameter<std::int64_t>(node_, prefix + "max_iterations", settings_.max_iterations);
  settings_.step_size = readParameter<double>(node_, prefix + "step_size", settings_.step_size);
  settings_.finite_difference_step =
      readParameter<double>(node_, prefix + "finite_difference_step", settings_.finite_difference_step);
  settings_.convergence_tolerance =
      readParameter<double>(node_, prefix + "convergence_tolerance", settings_.convergence_tolerance);
  settings_.tcp_path_length_weight =
      readParameter<double>(node_, prefix + "tcp_path_length_weight", settings_.tcp_path_length_weight);
  settings_.tcp_x_axis_world_z_weight = readParameter<double>(
      node_, prefix + "tcp_x_axis_world_z_weight", settings_.tcp_x_axis_world_z_weight);
  settings_.joint_travel_weight =
      readParameter<double>(node_, prefix + "joint_travel_weight", settings_.joint_travel_weight);

  RCLCPP_INFO(node_->get_logger(),
              "Trajectory preference optimizer: time_limit=%.3f max_iterations=%ld step_size=%.6f "
              "finite_difference_step=%.6f convergence_tolerance=%.6f "
              "tcp_path_length_weight=%.3f tcp_x_axis_world_z_weight=%.3f joint_travel_weight=%.3f",
              settings_.time_limit, static_cast<long>(settings_.max_iterations), settings_.step_size,
              settings_.finite_difference_step, settings_.convergence_tolerance,
              settings_.tcp_path_length_weight, settings_.tcp_x_axis_world_z_weight,
              settings_.joint_travel_weight);
}

std::string TrajectoryPreferenceAdapter::getDescription() const
{
  return "TrajectoryPreferenceAdapter";
}

void TrajectoryPreferenceAdapter::adapt(const planning_scene::PlanningSceneConstPtr& planning_scene,
                                        const planning_interface::MotionPlanRequest& request,
                                        planning_interface::MotionPlanResponse& response) const
{
  if (!planning_scene || !response.trajectory || response.trajectory->getWayPointCount() < 2)
  {
    RCLCPP_ERROR(node_->get_logger(),
                 "Trajectory preference optimizer requires a planning scene and an OMPL seed trajectory");
    markPlanningFailed(response);
    return;
  }

  RCLCPP_INFO(node_->get_logger(), "Optimizing trajectory with preference costs");
  TrajectoryPreferenceOptimizer optimizer(settings_);
  response.trajectory = optimizer.optimize(planning_scene, *response.trajectory, request.path_constraints);
}
}  // namespace path_planning_server

PLUGINLIB_EXPORT_CLASS(path_planning_server::TrajectoryPreferenceAdapter,
                       planning_interface::PlanningResponseAdapter)
