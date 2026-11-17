#include "path_planning_server/trajectory_optimizer_adapter.hpp"

#include <moveit/planning_interface/planning_response.hpp>
#include <moveit_msgs/msg/move_it_error_codes.hpp>
#include <rclcpp/rclcpp.hpp>

#include <cmath>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace
{
std::shared_ptr<rclcpp::Node> makeNode(const std::string& name, const std::string& parameter_namespace,
                                      const std::string& mode)
{
  rclcpp::NodeOptions options;
  options.parameter_overrides(
      { rclcpp::Parameter(parameter_namespace + ".trajectory_optimizer.mode", mode) });
  return std::make_shared<rclcpp::Node>(name, options);
}

bool expect(bool condition, const std::string& message)
{
  if (!condition)
  {
    std::cerr << message << '\n';
  }
  return condition;
}
}  // namespace

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  bool passed = true;

  {
    path_planning_server::TrajectoryOptimizerAdapter adapter;
    adapter.initialize(
        makeNode("trajectory_optimizer_none_test", "ompl_none", "none"), "ompl_none");
    passed &= expect(adapter.getDescription() == "TrajectoryOptimizerAdapter",
                     "Adapter description does not match");

    planning_interface::MotionPlanRequest request;
    planning_interface::MotionPlanResponse response;
    response.error_code.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
    response.planning_time = 1.25;
    response.planner_id = "ompl";
    adapter.adapt(nullptr, request, response);

    passed &= expect(response.error_code.val == moveit_msgs::msg::MoveItErrorCodes::SUCCESS,
                     "none mode changed the planning result");
    passed &= expect(std::abs(response.planning_time - 1.25) < 1e-12,
                     "none mode changed planning time");
    passed &= expect(response.planner_id == "ompl", "none mode changed planner id");
  }

  for (const std::string mode : { "chomp", "stomp" })
  {
    try
    {
      path_planning_server::TrajectoryOptimizerAdapter adapter;
      const std::string parameter_namespace = "ompl_" + mode;
      adapter.initialize(
          makeNode("trajectory_optimizer_" + mode + "_test", parameter_namespace, mode),
          parameter_namespace);
      planning_interface::MotionPlanRequest request;
      planning_interface::MotionPlanResponse response;
      response.error_code.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
      adapter.adapt(nullptr, request, response);
      passed &= expect(response.error_code.val == moveit_msgs::msg::MoveItErrorCodes::PLANNING_FAILED,
                       mode + " mode did not report a missing seed as a planning failure");
    }
    catch (const std::exception& exception)
    {
      std::cerr << mode << " mode initialization failed: " << exception.what() << '\n';
      passed = false;
    }
  }

  bool rejected_invalid_mode = false;
  try
  {
    path_planning_server::TrajectoryOptimizerAdapter adapter;
    adapter.initialize(
        makeNode("trajectory_optimizer_invalid_test", "ompl_invalid", "invalid"),
        "ompl_invalid");
  }
  catch (const std::exception&)
  {
    rejected_invalid_mode = true;
  }
  passed &= expect(rejected_invalid_mode, "Invalid optimizer mode was accepted");

  rclcpp::shutdown();
  return passed ? 0 : 1;
}
