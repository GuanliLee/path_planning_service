#pragma once

#include "path_planning_server/trajectory_preference_optimizer.hpp"

#include <moveit/planning_interface/planning_response_adapter.hpp>
#include <rclcpp/node.hpp>

#include <string>

namespace path_planning_server
{
class TrajectoryPreferenceAdapter : public planning_interface::PlanningResponseAdapter
{
public:
  void initialize(const rclcpp::Node::SharedPtr& node,
                  const std::string& parameter_namespace) override;
  std::string getDescription() const override;
  void adapt(const planning_scene::PlanningSceneConstPtr& planning_scene,
             const planning_interface::MotionPlanRequest& request,
             planning_interface::MotionPlanResponse& response) const override;

private:
  rclcpp::Node::SharedPtr node_;
  TrajectoryPreferenceSettings settings_;
};
}  // namespace path_planning_server
