#pragma once

#include <memory>
#include <string>
#include <vector>

#include <moveit/ompl_interface/ompl_interface.hpp>
#include <moveit/planning_interface/planning_interface.hpp>
#include <rclcpp/node.hpp>

namespace path_planning_server
{

class MultiSegmentOmplPlanner : public planning_interface::PlannerManager
{
public:
  bool initialize(
    const moveit::core::RobotModelConstPtr & model,
    const rclcpp::Node::SharedPtr & node,
    const std::string & parameter_namespace) override;

  std::string getDescription() const override;
  void getPlanningAlgorithms(std::vector<std::string> & algorithms) const override;
  planning_interface::PlanningContextPtr getPlanningContext(
    const planning_scene::PlanningSceneConstPtr & planning_scene,
    const planning_interface::MotionPlanRequest & request,
    moveit_msgs::msg::MoveItErrorCodes & error_code) const override;
  bool canServiceRequest(
    const planning_interface::MotionPlanRequest & request) const override;
  void setPlannerConfigurations(
    const planning_interface::PlannerConfigurationMap & configurations) override;

private:
  moveit::core::RobotModelConstPtr robot_model_;
  std::shared_ptr<ompl_interface::OMPLInterface> ompl_interface_;
};

}  // namespace path_planning_server
