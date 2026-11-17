#include "path_planning_server/trajectory_optimizer_adapter.hpp"

#include <chomp_motion_planner/chomp_parameters.hpp>
#include <chomp_motion_planner/chomp_planner.hpp>
#include <moveit/collision_distance_field/collision_detector_allocator_hybrid.hpp>
#include <moveit/kinematic_constraints/utils.hpp>
#include <moveit/planning_interface/planning_interface.hpp>
#include <moveit/robot_model/joint_model_group.hpp>
#include <moveit/robot_state/conversions.hpp>
#include <moveit/robot_trajectory/robot_trajectory.hpp>
#include <moveit_msgs/msg/generic_trajectory.hpp>
#include <moveit_msgs/msg/move_it_error_codes.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <pluginlib/class_loader.hpp>

#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>

namespace path_planning_server
{
struct TrajectoryOptimizerAdapter::StompPlugin
{
  StompPlugin() : loader("moveit_core", "planning_interface::PlannerManager")
  {
  }

  pluginlib::ClassLoader<planning_interface::PlannerManager> loader;
  planning_interface::PlannerManagerPtr planner;
};

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

planning_interface::MotionPlanRequest makeOptimizerRequest(
    const planning_interface::MotionPlanRequest& request,
    const robot_trajectory::RobotTrajectory& seed)
{
  planning_interface::MotionPlanRequest optimizer_request = request;
  const moveit::core::JointModelGroup* joint_group = seed.getGroup();
  moveit::core::robotStateToRobotStateMsg(seed.getFirstWayPoint(), optimizer_request.start_state);
  optimizer_request.goal_constraints = {
    kinematic_constraints::constructGoalConstraints(seed.getLastWayPoint(), joint_group)
  };
  return optimizer_request;
}

void markPlanningFailed(planning_interface::MotionPlanResponse& response)
{
  response.error_code.val = moveit_msgs::msg::MoveItErrorCodes::PLANNING_FAILED;
  response.trajectory.reset();
}

}  // namespace

TrajectoryOptimizerAdapter::TrajectoryOptimizerAdapter() = default;
TrajectoryOptimizerAdapter::~TrajectoryOptimizerAdapter() = default;

void TrajectoryOptimizerAdapter::initialize(const rclcpp::Node::SharedPtr& node,
                                            const std::string& parameter_namespace)
{
  node_ = node;
  parameter_namespace_ = parameter_namespace;
  const std::string mode_name =
      readParameter<std::string>(node_, parameterName(parameter_namespace_, "trajectory_optimizer.mode"), "none");

  if (mode_name == "none")
  {
    mode_ = Mode::NONE;
    RCLCPP_INFO(node_->get_logger(), "Trajectory optimizer mode: none");
    return;
  }

  if (mode_name == "chomp")
  {
    mode_ = Mode::CHOMP;
    const std::string prefix = parameterName(parameter_namespace_, "trajectory_optimizer.chomp.");
    chomp_.planning_time_limit =
        readParameter<double>(node_, prefix + "planning_time_limit", chomp_.planning_time_limit);
    chomp_.max_iterations = readParameter<std::int64_t>(node_, prefix + "max_iterations", chomp_.max_iterations);
    chomp_.max_iterations_after_collision_free = readParameter<std::int64_t>(
        node_, prefix + "max_iterations_after_collision_free", chomp_.max_iterations_after_collision_free);
    chomp_.smoothness_cost_weight =
        readParameter<double>(node_, prefix + "smoothness_cost_weight", chomp_.smoothness_cost_weight);
    chomp_.obstacle_cost_weight =
        readParameter<double>(node_, prefix + "obstacle_cost_weight", chomp_.obstacle_cost_weight);
    chomp_.learning_rate = readParameter<double>(node_, prefix + "learning_rate", chomp_.learning_rate);
    chomp_.ridge_factor = readParameter<double>(node_, prefix + "ridge_factor", chomp_.ridge_factor);
    chomp_.min_clearance = readParameter<double>(node_, prefix + "min_clearance", chomp_.min_clearance);
    chomp_.joint_update_limit =
        readParameter<double>(node_, prefix + "joint_update_limit", chomp_.joint_update_limit);

    RCLCPP_INFO(node_->get_logger(),
                "Trajectory optimizer mode: chomp; planning_time_limit=%.3f max_iterations=%ld "
                "max_iterations_after_collision_free=%ld smoothness_cost_weight=%.3f obstacle_cost_weight=%.3f "
                "learning_rate=%.3f ridge_factor=%.3f min_clearance=%.3f joint_update_limit=%.3f",
                chomp_.planning_time_limit, static_cast<long>(chomp_.max_iterations),
                static_cast<long>(chomp_.max_iterations_after_collision_free), chomp_.smoothness_cost_weight,
                chomp_.obstacle_cost_weight, chomp_.learning_rate, chomp_.ridge_factor, chomp_.min_clearance,
                chomp_.joint_update_limit);
    return;
  }

  if (mode_name == "stomp")
  {
    mode_ = Mode::STOMP;
    const std::string prefix = parameterName(parameter_namespace_, "trajectory_optimizer.stomp.");
    stomp_.num_timesteps = readParameter<std::int64_t>(node_, prefix + "num_timesteps", stomp_.num_timesteps);
    stomp_.num_iterations = readParameter<std::int64_t>(node_, prefix + "num_iterations", stomp_.num_iterations);
    stomp_.num_iterations_after_valid = readParameter<std::int64_t>(
        node_, prefix + "num_iterations_after_valid", stomp_.num_iterations_after_valid);
    stomp_.num_rollouts = readParameter<std::int64_t>(node_, prefix + "num_rollouts", stomp_.num_rollouts);
    stomp_.max_rollouts = readParameter<std::int64_t>(node_, prefix + "max_rollouts", stomp_.max_rollouts);
    stomp_.exponentiated_cost_sensitivity = readParameter<double>(
        node_, prefix + "exponentiated_cost_sensitivity", stomp_.exponentiated_cost_sensitivity);
    stomp_.control_cost_weight =
        readParameter<double>(node_, prefix + "control_cost_weight", stomp_.control_cost_weight);
    stomp_.delta_t = readParameter<double>(node_, prefix + "delta_t", stomp_.delta_t);

    RCLCPP_INFO(node_->get_logger(),
                "Trajectory optimizer mode: stomp; num_timesteps=%ld num_iterations=%ld "
                "num_iterations_after_valid=%ld num_rollouts=%ld max_rollouts=%ld "
                "exponentiated_cost_sensitivity=%.3f control_cost_weight=%.3f delta_t=%.3f",
                static_cast<long>(stomp_.num_timesteps), static_cast<long>(stomp_.num_iterations),
                static_cast<long>(stomp_.num_iterations_after_valid), static_cast<long>(stomp_.num_rollouts),
                static_cast<long>(stomp_.max_rollouts), stomp_.exponentiated_cost_sensitivity,
                stomp_.control_cost_weight, stomp_.delta_t);
    return;
  }

  throw std::invalid_argument("trajectory_optimizer.mode must be one of: none, chomp, stomp");
}

std::string TrajectoryOptimizerAdapter::getDescription() const
{
  return "TrajectoryOptimizerAdapter";
}

void TrajectoryOptimizerAdapter::adapt(const planning_scene::PlanningSceneConstPtr& planning_scene,
                                       const planning_interface::MotionPlanRequest& request,
                                       planning_interface::MotionPlanResponse& response) const
{
  if (mode_ == Mode::NONE)
  {
    return;
  }

  if (!planning_scene || !response.trajectory || response.trajectory->getWayPointCount() < 2)
  {
    RCLCPP_ERROR(node_->get_logger(), "Trajectory optimizer requires a planning scene and an OMPL seed trajectory");
    markPlanningFailed(response);
    return;
  }

  if (mode_ == Mode::CHOMP)
  {
    runChomp(planning_scene, request, response);
  }
  else
  {
    runStomp(planning_scene, request, response);
  }
}

void TrajectoryOptimizerAdapter::runChomp(const planning_scene::PlanningSceneConstPtr& planning_scene,
                                          const planning_interface::MotionPlanRequest& request,
                                          planning_interface::MotionPlanResponse& response) const
{
  RCLCPP_INFO(node_->get_logger(), "Optimizing OMPL trajectory with CHOMP");
  const auto started_at = std::chrono::steady_clock::now();
  const planning_interface::MotionPlanRequest optimizer_request = makeOptimizerRequest(request, *response.trajectory);

  chomp::ChompParameters parameters;
  parameters.planning_time_limit_ = chomp_.planning_time_limit;
  parameters.max_iterations_ = static_cast<int>(chomp_.max_iterations);
  parameters.max_iterations_after_collision_free_ = static_cast<int>(chomp_.max_iterations_after_collision_free);
  parameters.smoothness_cost_weight_ = chomp_.smoothness_cost_weight;
  parameters.obstacle_cost_weight_ = chomp_.obstacle_cost_weight;
  parameters.learning_rate_ = chomp_.learning_rate;
  parameters.ridge_factor_ = chomp_.ridge_factor;
  parameters.min_clearance_ = chomp_.min_clearance;
  parameters.joint_update_limit_ = chomp_.joint_update_limit;
  parameters.setTrajectoryInitializationMethod("fillTrajectory");

  planning_interface::MotionPlanDetailedResponse optimized_response;
  optimized_response.trajectory.push_back(
      std::make_shared<robot_trajectory::RobotTrajectory>(*response.trajectory));
  planning_scene::PlanningScenePtr chomp_scene = planning_scene->diff();
  chomp_scene->allocateCollisionDetector(collision_detection::CollisionDetectorAllocatorHybrid::create());
  chomp::ChompPlanner planner;
  planner.solve(chomp_scene, optimizer_request, parameters, optimized_response);

  if (optimized_response.error_code.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS ||
      optimized_response.trajectory.empty() || !optimized_response.trajectory.front())
  {
    RCLCPP_ERROR(node_->get_logger(), "CHOMP trajectory optimization failed with code %d",
                 optimized_response.error_code.val);
    markPlanningFailed(response);
    return;
  }

  response.trajectory = optimized_response.trajectory.front();
  response.error_code.val = optimized_response.error_code.val;
  response.planning_time += std::chrono::duration<double>(std::chrono::steady_clock::now() - started_at).count();
}

void TrajectoryOptimizerAdapter::runStomp(const planning_scene::PlanningSceneConstPtr& planning_scene,
                                          const planning_interface::MotionPlanRequest& request,
                                          planning_interface::MotionPlanResponse& response) const
{
  RCLCPP_INFO(node_->get_logger(), "Optimizing OMPL trajectory with STOMP");
  const auto started_at = std::chrono::steady_clock::now();
  planning_interface::MotionPlanRequest optimizer_request = makeOptimizerRequest(request, *response.trajectory);

  moveit_msgs::msg::RobotTrajectory seed_message;
  response.trajectory->getRobotTrajectoryMsg(seed_message);
  moveit_msgs::msg::GenericTrajectory reference_trajectory;
  reference_trajectory.joint_trajectory.push_back(seed_message.joint_trajectory);
  optimizer_request.reference_trajectories = { reference_trajectory };

  const moveit::core::JointModelGroup* joint_group = response.trajectory->getGroup();
  optimizer_request.trajectory_constraints.constraints.clear();
  optimizer_request.trajectory_constraints.constraints.reserve(response.trajectory->getWayPointCount());
  for (std::size_t index = 0; index < response.trajectory->getWayPointCount(); ++index)
  {
    optimizer_request.trajectory_constraints.constraints.push_back(
        kinematic_constraints::constructGoalConstraints(response.trajectory->getWayPoint(index), joint_group));
  }

  try
  {
    if (!stomp_plugin_)
    {
      stomp_plugin_ = std::make_unique<StompPlugin>();
      stomp_plugin_->planner = stomp_plugin_->loader.createSharedInstance("stomp_moveit/StompPlanner");
      const std::string stomp_namespace = parameterName(parameter_namespace_, "trajectory_optimizer.stomp");
      if (!stomp_plugin_->planner->initialize(planning_scene->getRobotModel(), node_, stomp_namespace))
      {
        RCLCPP_ERROR(node_->get_logger(), "STOMP planner initialization failed");
        markPlanningFailed(response);
        return;
      }
    }

    moveit_msgs::msg::MoveItErrorCodes context_error;
    planning_interface::PlanningContextPtr context =
        stomp_plugin_->planner->getPlanningContext(planning_scene, optimizer_request, context_error);
    if (!context)
    {
      RCLCPP_ERROR(node_->get_logger(), "STOMP planning context creation failed with code %d", context_error.val);
      markPlanningFailed(response);
      return;
    }

    planning_interface::MotionPlanResponse optimized_response;
    context->solve(optimized_response);
    if (!optimized_response || !optimized_response.trajectory)
    {
      RCLCPP_ERROR(node_->get_logger(), "STOMP trajectory optimization failed with code %d",
                   optimized_response.error_code.val);
      markPlanningFailed(response);
      return;
    }

    response.trajectory = optimized_response.trajectory;
    response.error_code = optimized_response.error_code;
    response.planning_time += std::chrono::duration<double>(std::chrono::steady_clock::now() - started_at).count();
  }
  catch (const pluginlib::PluginlibException& exception)
  {
    RCLCPP_ERROR(node_->get_logger(), "STOMP plugin error: %s", exception.what());
    markPlanningFailed(response);
  }
}
}  // namespace path_planning_server

PLUGINLIB_EXPORT_CLASS(path_planning_server::TrajectoryOptimizerAdapter,
                       planning_interface::PlanningResponseAdapter)
