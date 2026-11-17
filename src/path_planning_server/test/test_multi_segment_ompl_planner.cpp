#include "path_planning_server/multi_segment_ompl_planner.hpp"

#include <geometry_msgs/msg/pose.hpp>
#include <moveit/kinematic_constraints/utils.hpp>
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/robot_state/conversions.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <moveit/utils/robot_model_test_utils.hpp>
#include <moveit_msgs/msg/move_it_error_codes.hpp>
#include <rclcpp/rclcpp.hpp>

#include <urdf_model/pose.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace
{

void require(bool condition, const std::string & message)
{
  if (!condition) {
    std::cerr << "FAILED: " << message << '\n';
    std::exit(EXIT_FAILURE);
  }
}

moveit::core::RobotModelPtr makeRobotModel()
{
  moveit::core::RobotModelBuilder builder("multi_segment_test_robot", "base");
  std::vector<geometry_msgs::msg::Pose> origins(3);
  for (auto & origin : origins) {
    origin.orientation.w = 1.0;
    origin.position.x = 0.4;
  }
  builder.addChain(
    "base->link_1->link_2->tool", "continuous", origins,
    urdf::Vector3(0.0, 1.0, 0.0));
  builder.addGroupChain("base", "tool", "arm");
  require(builder.isValid(), "test robot model should be valid");
  return builder.build();
}

moveit_msgs::msg::Constraints makeGoal(
  const moveit::core::RobotModelPtr & model,
  const std::vector<double> & positions)
{
  const auto * group = model->getJointModelGroup("arm");
  moveit::core::RobotState state(model);
  state.setToDefaultValues();
  state.setJointGroupPositions(group, positions);
  state.update();
  return kinematic_constraints::constructGoalConstraints(state, group, 1e-4);
}

std::vector<double> positionsAt(
  const robot_trajectory::RobotTrajectory & trajectory,
  std::size_t waypoint_index)
{
  std::vector<double> positions;
  trajectory.getWayPoint(waypoint_index).copyJointGroupPositions(
    trajectory.getGroup(), positions);
  return positions;
}

double distance(
  const std::vector<double> & left,
  const std::vector<double> & right)
{
  double squared_distance = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    const double difference = left[index] - right[index];
    squared_distance += difference * difference;
  }
  return std::sqrt(squared_distance);
}

std::size_t findWaypoint(
  const robot_trajectory::RobotTrajectory & trajectory,
  const std::vector<double> & target,
  std::size_t first_index)
{
  for (std::size_t index = first_index; index < trajectory.getWayPointCount(); ++index) {
    if (distance(positionsAt(trajectory, index), target) < 1e-3) {
      return index;
    }
  }
  return trajectory.getWayPointCount();
}

}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  const auto model = makeRobotModel();
  const auto scene = std::make_shared<planning_scene::PlanningScene>(model);
  const auto node = std::make_shared<rclcpp::Node>("multi_segment_ompl_planner_test");

  path_planning_server::MultiSegmentOmplPlanner planner;
  require(planner.initialize(model, node, ""), "planner should initialize");

  planning_interface::PlannerConfigurationSettings configuration;
  configuration.name = "arm";
  configuration.group = "arm";
  configuration.config["type"] = "geometric::RRTConnect";
  planner.setPlannerConfigurations({{"arm", configuration}});

  const std::vector<double> start = {0.0, 0.0, 0.0};
  const std::vector<double> waypoint_1 = {0.4, -0.2, 0.3};
  const std::vector<double> waypoint_2 = {-0.3, 0.5, -0.4};
  const std::vector<double> goal = {0.7, 0.2, -0.6};

  moveit::core::RobotState start_state(model);
  start_state.setToDefaultValues();
  start_state.setJointGroupPositions(model->getJointModelGroup("arm"), start);
  start_state.update();

  planning_interface::MotionPlanRequest request;
  request.group_name = "arm";
  request.allowed_planning_time = 3.0;
  request.num_planning_attempts = 1;
  moveit::core::robotStateToRobotStateMsg(start_state, request.start_state);
  request.trajectory_constraints.constraints = {
    makeGoal(model, waypoint_1), makeGoal(model, waypoint_2)};
  request.goal_constraints = {makeGoal(model, goal)};

  moveit_msgs::msg::MoveItErrorCodes context_error;
  const auto context = planner.getPlanningContext(scene, request, context_error);
  require(context != nullptr, "planning context should be created");
  require(
    context_error.val == moveit_msgs::msg::MoveItErrorCodes::SUCCESS,
    "planning context should be valid");

  planning_interface::MotionPlanResponse response;
  context->solve(response);
  require(static_cast<bool>(response), "multi-segment plan should succeed");
  require(response.trajectory != nullptr, "multi-segment plan should return a trajectory");

  const std::size_t first_waypoint_index = findWaypoint(*response.trajectory, waypoint_1, 0U);
  require(
    first_waypoint_index < response.trajectory->getWayPointCount(),
    "trajectory should pass the first waypoint");
  const std::size_t second_waypoint_index = findWaypoint(
    *response.trajectory, waypoint_2, first_waypoint_index + 1U);
  require(
    second_waypoint_index < response.trajectory->getWayPointCount(),
    "trajectory should pass the second waypoint after the first");
  require(
    distance(
      positionsAt(
        *response.trajectory, response.trajectory->getWayPointCount() - 1U),
      goal) < 1e-3,
    "trajectory should end at the final goal");

  for (std::size_t index = 1; index < response.trajectory->getWayPointCount(); ++index) {
    require(
      distance(
        positionsAt(*response.trajectory, index - 1U),
        positionsAt(*response.trajectory, index)) > 1e-12,
      "trajectory should not contain a duplicated segment seam");
  }

  rclcpp::shutdown();
  return EXIT_SUCCESS;
}
