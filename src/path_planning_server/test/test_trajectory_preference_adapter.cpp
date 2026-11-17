#include "path_planning_server/trajectory_preference_adapter.hpp"

#include <moveit/planning_interface/planning_response.hpp>
#include <moveit_msgs/msg/move_it_error_codes.hpp>
#include <rclcpp/rclcpp.hpp>

#include <iostream>
#include <memory>

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  bool passed = true;

  auto node = std::make_shared<rclcpp::Node>("trajectory_preference_adapter_test");
  path_planning_server::TrajectoryPreferenceAdapter adapter;
  adapter.initialize(node, "ompl_none_preference");
  if (adapter.getDescription() != "TrajectoryPreferenceAdapter")
  {
    std::cerr << "Adapter description does not match\n";
    passed = false;
  }

  planning_interface::MotionPlanRequest request;
  planning_interface::MotionPlanResponse response;
  response.error_code.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
  adapter.adapt(nullptr, request, response);
  if (response.error_code.val != moveit_msgs::msg::MoveItErrorCodes::PLANNING_FAILED)
  {
    std::cerr << "Adapter did not report a missing seed as a planning failure\n";
    passed = false;
  }

  rclcpp::shutdown();
  return passed ? 0 : 1;
}
