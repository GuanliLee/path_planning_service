#include "path_planning_server/trajectory_preference_optimizer.hpp"

#include <moveit/robot_model/joint_model_group.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <rclcpp/logging.hpp>

#include <Eigen/Geometry>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace path_planning_server
{
namespace
{
struct CostBreakdown
{
  double tcp_path_length{ 0.0 };
  double tcp_x_axis_world_z{ 0.0 };
  double joint_travel{ 0.0 };

  double total(const TrajectoryPreferenceSettings& settings) const
  {
    return settings.tcp_path_length_weight * tcp_path_length +
           settings.tcp_x_axis_world_z_weight * tcp_x_axis_world_z +
           settings.joint_travel_weight * joint_travel;
  }
};

double jointTravelCost(const robot_trajectory::RobotTrajectory& trajectory)
{
  const moveit::core::JointModelGroup* joint_group = trajectory.getGroup();
  std::vector<double> previous;
  trajectory.getWayPoint(0).copyJointGroupPositions(joint_group, previous);

  double cost = 0.0;
  for (std::size_t index = 1; index < trajectory.getWayPointCount(); ++index)
  {
    std::vector<double> current;
    trajectory.getWayPoint(index).copyJointGroupPositions(joint_group, current);
    for (std::size_t variable = 0; variable < current.size(); ++variable)
    {
      const double difference = current[variable] - previous[variable];
      cost += difference * difference;
    }
    previous = std::move(current);
  }
  return cost;
}

double tcpPathLengthCost(const robot_trajectory::RobotTrajectory& trajectory)
{
  const std::string& tip_link = trajectory.getGroup()->getLinkModelNames().back();
  Eigen::Vector3d previous = trajectory.getWayPoint(0).getGlobalLinkTransform(tip_link).translation();

  double cost = 0.0;
  for (std::size_t index = 1; index < trajectory.getWayPointCount(); ++index)
  {
    const Eigen::Vector3d current =
        trajectory.getWayPoint(index).getGlobalLinkTransform(tip_link).translation();
    cost += (current - previous).squaredNorm();
    previous = current;
  }
  return cost;
}

double tcpXAxisWorldZCost(const robot_trajectory::RobotTrajectory& trajectory)
{
  const std::string& tip_link = trajectory.getGroup()->getLinkModelNames().back();

  double cost = 0.0;
  for (std::size_t index = 0; index < trajectory.getWayPointCount(); ++index)
  {
    const Eigen::Matrix3d orientation =
        trajectory.getWayPoint(index).getGlobalLinkTransform(tip_link).rotation();
    const Eigen::Vector3d tcp_x_axis = orientation * Eigen::Vector3d::UnitX();
    const double angle = std::acos(
        std::clamp(tcp_x_axis.dot(Eigen::Vector3d::UnitZ()), -1.0, 1.0));
    cost += angle * angle;
  }
  return cost;
}

CostBreakdown evaluateRawCosts(const robot_trajectory::RobotTrajectory& trajectory)
{
  return {
    tcpPathLengthCost(trajectory),
    tcpXAxisWorldZCost(trajectory),
    jointTravelCost(trajectory),
  };
}

double normalizeCost(double cost, double initial_cost)
{
  return initial_cost > 0.0 ? cost / initial_cost : cost;
}

CostBreakdown evaluateNormalizedCosts(const robot_trajectory::RobotTrajectory& trajectory,
                                      const CostBreakdown& initial_raw_costs)
{
  const CostBreakdown raw_costs = evaluateRawCosts(trajectory);
  return {
    normalizeCost(raw_costs.tcp_path_length, initial_raw_costs.tcp_path_length),
    normalizeCost(raw_costs.tcp_x_axis_world_z, initial_raw_costs.tcp_x_axis_world_z),
    normalizeCost(raw_costs.joint_travel, initial_raw_costs.joint_travel),
  };
}

void logCosts(const CostBreakdown& before, const CostBreakdown& after,
              const TrajectoryPreferenceSettings& settings, double elapsed_time,
              const char* stop_reason)
{
  RCLCPP_INFO(rclcpp::get_logger("trajectory_preference_optimizer"),
              "Trajectory preference costs: tcp_path_length=%.9f -> %.9f "
              "tcp_x_axis_world_z=%.9f -> %.9f joint_travel=%.9f -> %.9f total=%.9f -> %.9f "
              "elapsed_time=%.6f stop_reason=%s",
              before.tcp_path_length, after.tcp_path_length,
              before.tcp_x_axis_world_z, after.tcp_x_axis_world_z,
              before.joint_travel, after.joint_travel, before.total(settings), after.total(settings),
              elapsed_time, stop_reason);
}
}  // namespace

TrajectoryPreferenceOptimizer::TrajectoryPreferenceOptimizer(TrajectoryPreferenceSettings settings)
  : settings_(std::move(settings))
{
}

robot_trajectory::RobotTrajectoryPtr TrajectoryPreferenceOptimizer::optimize(
    const planning_scene::PlanningSceneConstPtr& planning_scene,
    const robot_trajectory::RobotTrajectory& seed,
    const moveit_msgs::msg::Constraints& path_constraints) const
{
  const auto started_at = std::chrono::steady_clock::now();
  auto current = std::make_shared<robot_trajectory::RobotTrajectory>(seed, true);
  const CostBreakdown initial_raw_costs = evaluateRawCosts(*current);
  const CostBreakdown initial_costs = evaluateNormalizedCosts(*current, initial_raw_costs);
  const auto elapsedTime = [&started_at]() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - started_at).count();
  };
  const auto timedOut = [this, &elapsedTime]() {
    return elapsedTime() >= settings_.time_limit;
  };
  const auto finish = [&current, &initial_costs, &initial_raw_costs, &elapsedTime,
                       this](const char* stop_reason) {
    logCosts(initial_costs, evaluateNormalizedCosts(*current, initial_raw_costs), settings_,
             elapsedTime(), stop_reason);
    return current;
  };

  if (current->getWayPointCount() < 3)
  {
    return finish("converged");
  }

  const moveit::core::JointModelGroup* joint_group = current->getGroup();
  const std::size_t variable_count = joint_group->getVariableCount();
  double current_cost = initial_costs.total(settings_);

  for (std::int64_t iteration = 0; iteration < settings_.max_iterations; ++iteration)
  {
    if (timedOut())
    {
      return finish("time_limit");
    }

    std::vector<std::vector<double>> gradient(current->getWayPointCount(),
                                               std::vector<double>(variable_count, 0.0));
    for (std::size_t index = 1; index + 1 < current->getWayPointCount(); ++index)
    {
      moveit::core::RobotState& waypoint = *current->getWayPointPtr(index);
      std::vector<double> positions;
      waypoint.copyJointGroupPositions(joint_group, positions);
      for (std::size_t variable = 0; variable < variable_count; ++variable)
      {
        if (timedOut())
        {
          return finish("time_limit");
        }

        std::vector<double> perturbed = positions;
        perturbed[variable] += settings_.finite_difference_step;
        waypoint.setJointGroupPositions(joint_group, perturbed);
        waypoint.update();
        const double plus_cost =
            evaluateNormalizedCosts(*current, initial_raw_costs).total(settings_);

        perturbed[variable] -= 2.0 * settings_.finite_difference_step;
        waypoint.setJointGroupPositions(joint_group, perturbed);
        waypoint.update();
        const double minus_cost =
            evaluateNormalizedCosts(*current, initial_raw_costs).total(settings_);

        gradient[index][variable] =
            (plus_cost - minus_cost) / (2.0 * settings_.finite_difference_step);
        waypoint.setJointGroupPositions(joint_group, positions);
        waypoint.update();
      }
    }

    auto candidate = std::make_shared<robot_trajectory::RobotTrajectory>(*current, true);
    for (std::size_t index = 1; index + 1 < candidate->getWayPointCount(); ++index)
    {
      moveit::core::RobotState& waypoint = *candidate->getWayPointPtr(index);
      std::vector<double> positions;
      waypoint.copyJointGroupPositions(joint_group, positions);
      for (std::size_t variable = 0; variable < variable_count; ++variable)
      {
        positions[variable] -= settings_.step_size * gradient[index][variable];
      }
      waypoint.setJointGroupPositions(joint_group, positions);
      waypoint.update();
    }

    const double candidate_cost =
        evaluateNormalizedCosts(*candidate, initial_raw_costs).total(settings_);
    if (candidate_cost >= current_cost)
    {
      return finish("converged");
    }
    if (timedOut())
    {
      return finish("time_limit");
    }
    if (!planning_scene->isPathValid(*candidate, path_constraints, joint_group->getName()))
    {
      return finish("converged");
    }

    const double improvement = current_cost - candidate_cost;
    current = std::move(candidate);
    current_cost = candidate_cost;
    if (timedOut())
    {
      return finish("time_limit");
    }
    if (improvement < settings_.convergence_tolerance)
    {
      return finish("converged");
    }
  }

  return finish("max_iterations");
}
}  // namespace path_planning_server
