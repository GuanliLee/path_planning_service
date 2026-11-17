#include "path_planning_server/multi_segment_ompl_planner.hpp"

#include <algorithm>
#include <chrono>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <moveit/robot_state/conversions.hpp>
#include <moveit/robot_trajectory/robot_trajectory.hpp>
#include <moveit_msgs/msg/move_it_error_codes.hpp>
#include <pluginlib/class_list_macros.hpp>

namespace path_planning_server
{
namespace
{

class MultiSegmentOmplPlanningContext : public planning_interface::PlanningContext
{
public:
  MultiSegmentOmplPlanningContext(
    const moveit::core::RobotModelConstPtr & robot_model,
    std::shared_ptr<ompl_interface::OMPLInterface> ompl_interface,
    const std::string & group)
  : planning_interface::PlanningContext("MultiSegmentOmplPlanner", group),
    robot_model_(robot_model),
    ompl_interface_(std::move(ompl_interface))
  {
  }

  void solve(planning_interface::MotionPlanResponse & response) override
  {
    const auto started_at = std::chrono::steady_clock::now();
    const std::size_t segment_count =
      request_.trajectory_constraints.constraints.size() + 1U;
    auto combined_trajectory =
      std::make_shared<robot_trajectory::RobotTrajectory>(robot_model_, group_);

    planning_interface::MotionPlanRequest segment_request = request_;
    segment_request.trajectory_constraints.constraints.clear();

    for (std::size_t segment_index = 0; segment_index < segment_count; ++segment_index) {
      const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started_at).count();
      const double remaining_time = request_.allowed_planning_time - elapsed;
      if (remaining_time <= 0.0) {
        response.error_code.val = moveit_msgs::msg::MoveItErrorCodes::TIMED_OUT;
        response.planning_time = elapsed;
        return;
      }
      segment_request.allowed_planning_time =
        remaining_time / static_cast<double>(segment_count - segment_index);

      if (segment_index < request_.trajectory_constraints.constraints.size()) {
        segment_request.goal_constraints = {
          request_.trajectory_constraints.constraints[segment_index]};
      } else {
        segment_request.goal_constraints = request_.goal_constraints;
      }

      if (segment_index > 0U) {
        moveit::core::robotStateToRobotStateMsg(
          combined_trajectory->getLastWayPoint(), segment_request.start_state);
      }

      moveit_msgs::msg::MoveItErrorCodes context_error;
      active_context_ = ompl_interface_->getPlanningContext(
        planning_scene_, segment_request, context_error);
      if (!active_context_) {
        response.error_code.val = context_error.val;
        response.planning_time = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - started_at).count();
        return;
      }

      planning_interface::MotionPlanResponse segment_response;
      active_context_->solve(segment_response);
      active_context_.reset();
      if (!segment_response || !segment_response.trajectory ||
        segment_response.trajectory->empty())
      {
        response.error_code = segment_response.error_code;
        response.planning_time = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - started_at).count();
        return;
      }

      if (segment_index == 0U) {
        response.start_state = segment_response.start_state;
      }
      response.planner_id = segment_response.planner_id;

      const std::size_t first_waypoint = segment_index == 0U ? 0U : 1U;
      for (std::size_t waypoint_index = first_waypoint;
        waypoint_index < segment_response.trajectory->getWayPointCount();
        ++waypoint_index)
      {
        combined_trajectory->addSuffixWayPoint(
          segment_response.trajectory->getWayPoint(waypoint_index),
          segment_response.trajectory->getWayPointDurationFromPrevious(waypoint_index));
      }
    }

    response.trajectory = std::move(combined_trajectory);
    response.planning_time = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started_at).count();
    response.error_code.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
  }

  void solve(planning_interface::MotionPlanDetailedResponse & response) override
  {
    planning_interface::MotionPlanResponse simple_response;
    solve(simple_response);
    response.error_code.val = simple_response.error_code.val;
    response.error_code.message = simple_response.error_code.message;
    response.error_code.source = simple_response.error_code.source;
    response.planner_id = simple_response.planner_id;
    if (simple_response.trajectory) {
      response.trajectory.push_back(std::move(simple_response.trajectory));
      response.description.emplace_back("plan");
      response.processing_time.push_back(simple_response.planning_time);
    }
  }

  bool terminate() override
  {
    return active_context_ ? active_context_->terminate() : true;
  }

  void clear() override
  {
    if (active_context_) {
      active_context_->clear();
      active_context_.reset();
    }
  }

private:
  moveit::core::RobotModelConstPtr robot_model_;
  std::shared_ptr<ompl_interface::OMPLInterface> ompl_interface_;
  planning_interface::PlanningContextPtr active_context_;
};

}  // namespace

bool MultiSegmentOmplPlanner::initialize(
  const moveit::core::RobotModelConstPtr & model,
  const rclcpp::Node::SharedPtr & node,
  const std::string & parameter_namespace)
{
  if (!planning_interface::PlannerManager::initialize(model, node, parameter_namespace)) {
    return false;
  }
  robot_model_ = model;
  ompl_interface_ =
    std::make_shared<ompl_interface::OMPLInterface>(model, node, parameter_namespace);
  config_settings_ = ompl_interface_->getPlannerConfigurations();
  return true;
}

std::string MultiSegmentOmplPlanner::getDescription() const
{
  return "MultiSegmentOmplPlanner";
}

void MultiSegmentOmplPlanner::getPlanningAlgorithms(
  std::vector<std::string> & algorithms) const
{
  algorithms.clear();
  algorithms.reserve(config_settings_.size());
  for (const auto & configuration : config_settings_) {
    algorithms.push_back(configuration.first);
  }
}

planning_interface::PlanningContextPtr MultiSegmentOmplPlanner::getPlanningContext(
  const planning_scene::PlanningSceneConstPtr & planning_scene,
  const planning_interface::MotionPlanRequest & request,
  moveit_msgs::msg::MoveItErrorCodes & error_code) const
{
  if (!planning_scene || !canServiceRequest(request)) {
    error_code.val = moveit_msgs::msg::MoveItErrorCodes::INVALID_GOAL_CONSTRAINTS;
    return nullptr;
  }

  auto context = std::make_shared<MultiSegmentOmplPlanningContext>(
    robot_model_, ompl_interface_, request.group_name);
  context->setPlanningScene(planning_scene);
  context->setMotionPlanRequest(request);
  error_code.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
  return context;
}

bool MultiSegmentOmplPlanner::canServiceRequest(
  const planning_interface::MotionPlanRequest & request) const
{
  return !request.trajectory_constraints.constraints.empty() &&
         !request.goal_constraints.empty();
}

void MultiSegmentOmplPlanner::setPlannerConfigurations(
  const planning_interface::PlannerConfigurationMap & configurations)
{
  planning_interface::PlannerManager::setPlannerConfigurations(configurations);
  if (ompl_interface_) {
    ompl_interface_->setPlannerConfigurations(configurations);
  }
}

}  // namespace path_planning_server

PLUGINLIB_EXPORT_CLASS(
  path_planning_server::MultiSegmentOmplPlanner,
  planning_interface::PlannerManager)
