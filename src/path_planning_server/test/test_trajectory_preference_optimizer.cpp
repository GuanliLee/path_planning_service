#include "path_planning_server/trajectory_preference_optimizer.hpp"

#include <geometry_msgs/msg/pose.hpp>
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <moveit/robot_trajectory/robot_trajectory.hpp>
#include <moveit/utils/robot_model_test_utils.hpp>

#include <urdf_model/pose.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace
{
void require(bool condition, const std::string& message)
{
  if (!condition)
  {
    std::cerr << "FAILED: " << message << '\n';
    std::exit(EXIT_FAILURE);
  }
}

moveit::core::RobotModelPtr makeRobotModel()
{
  moveit::core::RobotModelBuilder builder("preference_test_robot", "base");
  std::vector<geometry_msgs::msg::Pose> origins(3);
  for (geometry_msgs::msg::Pose& origin : origins)
  {
    origin.orientation.w = 1.0;
    origin.position.x = 0.4;
  }
  builder.addChain("base->link_1->link_2->tool", "continuous", origins, urdf::Vector3(0.0, 1.0, 0.0));
  builder.addGroupChain("base", "tool", "arm");
  require(builder.isValid(), "test robot model should be valid");
  return builder.build();
}

robot_trajectory::RobotTrajectory makeSeed(const moveit::core::RobotModelPtr& model)
{
  robot_trajectory::RobotTrajectory seed(model, "arm");
  const moveit::core::JointModelGroup* group = model->getJointModelGroup("arm");
  moveit::core::RobotState state(model);
  state.setToDefaultValues();

  const std::vector<std::vector<double>> waypoints = {
    { 0.0, 0.0, 0.0 },
    { 1.2, -0.8, 0.6 },
    { -1.0, 1.0, -0.7 },
    { 1.4, -0.6, 0.8 },
    { 0.8, -0.4, 0.2 },
  };
  for (const std::vector<double>& positions : waypoints)
  {
    state.setJointGroupPositions(group, positions);
    state.update();
    seed.addSuffixWayPoint(state, 0.1);
  }
  return seed;
}

std::vector<double> jointPositions(const robot_trajectory::RobotTrajectory& trajectory, std::size_t index)
{
  std::vector<double> positions;
  trajectory.getWayPoint(index).copyJointGroupPositions(trajectory.getGroup(), positions);
  return positions;
}

double jointTravelCost(const robot_trajectory::RobotTrajectory& trajectory)
{
  double cost = 0.0;
  std::vector<double> previous = jointPositions(trajectory, 0);
  for (std::size_t index = 1; index < trajectory.getWayPointCount(); ++index)
  {
    const std::vector<double> current = jointPositions(trajectory, index);
    for (std::size_t joint = 0; joint < current.size(); ++joint)
    {
      const double difference = current[joint] - previous[joint];
      cost += difference * difference;
    }
    previous = current;
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
    const double angle = std::acos(std::clamp(tcp_x_axis.dot(Eigen::Vector3d::UnitZ()), -1.0, 1.0));
    cost += angle * angle;
  }
  return cost;
}

double totalPreferenceCost(const robot_trajectory::RobotTrajectory& trajectory, double tcp_weight,
                           double x_axis_weight, double joint_weight)
{
  return tcp_weight * tcpPathLengthCost(trajectory) +
         x_axis_weight * tcpXAxisWorldZCost(trajectory) +
         joint_weight * jointTravelCost(trajectory);
}

void requireEqual(const std::vector<double>& actual, const std::vector<double>& expected,
                  const std::string& message)
{
  require(actual.size() == expected.size(), message + " size");
  for (std::size_t index = 0; index < actual.size(); ++index)
  {
    require(std::abs(actual[index] - expected[index]) < 1e-12, message);
  }
}
void testJointTravelCost(const moveit::core::RobotModelPtr& model,
                         const planning_scene::PlanningSceneConstPtr& planning_scene)
{
  const robot_trajectory::RobotTrajectory seed = makeSeed(model);

  path_planning_server::TrajectoryPreferenceSettings settings;
  settings.max_iterations = 20;
  settings.step_size = 0.05;
  settings.finite_difference_step = 0.0001;
  settings.convergence_tolerance = 0.0000001;
  settings.tcp_path_length_weight = 0.0;
  settings.tcp_x_axis_world_z_weight = 0.0;
  settings.joint_travel_weight = 1.0;

  path_planning_server::TrajectoryPreferenceOptimizer optimizer(settings);
  const robot_trajectory::RobotTrajectoryPtr optimized =
      optimizer.optimize(planning_scene, seed, moveit_msgs::msg::Constraints{});

  require(optimized != nullptr, "optimizer should return a trajectory");
  require(jointTravelCost(*optimized) < jointTravelCost(seed), "joint travel cost should decrease");
  requireEqual(jointPositions(*optimized, 0), jointPositions(seed, 0), "first waypoint should remain fixed");
  requireEqual(jointPositions(*optimized, optimized->getWayPointCount() - 1),
               jointPositions(seed, seed.getWayPointCount() - 1), "last waypoint should remain fixed");
}

void testTcpPathLengthCost(const moveit::core::RobotModelPtr& model,
                           const planning_scene::PlanningSceneConstPtr& planning_scene)
{
  const robot_trajectory::RobotTrajectory seed = makeSeed(model);

  path_planning_server::TrajectoryPreferenceSettings settings;
  settings.max_iterations = 20;
  settings.step_size = 0.05;
  settings.finite_difference_step = 0.0001;
  settings.convergence_tolerance = 0.0000001;
  settings.tcp_path_length_weight = 1.0;
  settings.tcp_x_axis_world_z_weight = 0.0;
  settings.joint_travel_weight = 0.0;

  path_planning_server::TrajectoryPreferenceOptimizer optimizer(settings);
  const robot_trajectory::RobotTrajectoryPtr optimized =
      optimizer.optimize(planning_scene, seed, moveit_msgs::msg::Constraints{});

  require(optimized != nullptr, "optimizer should return a trajectory");
  require(tcpPathLengthCost(*optimized) < tcpPathLengthCost(seed), "TCP path length cost should decrease");
  requireEqual(jointPositions(*optimized, 0), jointPositions(seed, 0), "first waypoint should remain fixed");
  requireEqual(jointPositions(*optimized, optimized->getWayPointCount() - 1),
               jointPositions(seed, seed.getWayPointCount() - 1), "last waypoint should remain fixed");
}

void testTcpXAxisWorldZCost(const moveit::core::RobotModelPtr& model,
                           const planning_scene::PlanningSceneConstPtr& planning_scene)
{
  const robot_trajectory::RobotTrajectory seed = makeSeed(model);

  path_planning_server::TrajectoryPreferenceSettings settings;
  settings.max_iterations = 20;
  settings.step_size = 0.05;
  settings.finite_difference_step = 0.0001;
  settings.convergence_tolerance = 0.0000001;
  settings.tcp_path_length_weight = 0.0;
  settings.tcp_x_axis_world_z_weight = 1.0;
  settings.joint_travel_weight = 0.0;

  path_planning_server::TrajectoryPreferenceOptimizer optimizer(settings);
  const robot_trajectory::RobotTrajectoryPtr optimized =
      optimizer.optimize(planning_scene, seed, moveit_msgs::msg::Constraints{});

  require(optimized != nullptr, "optimizer should return a trajectory");
  require(tcpXAxisWorldZCost(*optimized) < tcpXAxisWorldZCost(seed),
          "TCP X-axis to world Z-axis cost should decrease");
  requireEqual(jointPositions(*optimized, 0), jointPositions(seed, 0), "first waypoint should remain fixed");
  requireEqual(jointPositions(*optimized, optimized->getWayPointCount() - 1),
               jointPositions(seed, seed.getWayPointCount() - 1), "last waypoint should remain fixed");
}

void testCombinedCost(const moveit::core::RobotModelPtr& model,
                      const planning_scene::PlanningSceneConstPtr& planning_scene)
{
  const robot_trajectory::RobotTrajectory seed = makeSeed(model);

  path_planning_server::TrajectoryPreferenceSettings settings;
  settings.max_iterations = 20;
  settings.step_size = 0.05;
  settings.finite_difference_step = 0.0001;
  settings.convergence_tolerance = 0.0000001;
  settings.tcp_path_length_weight = 1.0;
  settings.tcp_x_axis_world_z_weight = 0.2;
  settings.joint_travel_weight = 0.1;

  path_planning_server::TrajectoryPreferenceOptimizer optimizer(settings);
  const robot_trajectory::RobotTrajectoryPtr optimized =
      optimizer.optimize(planning_scene, seed, moveit_msgs::msg::Constraints{});

  require(totalPreferenceCost(*optimized, 1.0, 0.2, 0.1) < totalPreferenceCost(seed, 1.0, 0.2, 0.1),
          "combined preference cost should decrease");
  require(planning_scene->isPathValid(*optimized, optimized->getGroupName()),
          "combined preference trajectory should remain valid");
  requireEqual(jointPositions(*optimized, 0), jointPositions(seed, 0), "first waypoint should remain fixed");
  requireEqual(jointPositions(*optimized, optimized->getWayPointCount() - 1),
               jointPositions(seed, seed.getWayPointCount() - 1), "last waypoint should remain fixed");
}

void testTimeLimit(const moveit::core::RobotModelPtr& model,
                   const planning_scene::PlanningSceneConstPtr& planning_scene)
{
  const robot_trajectory::RobotTrajectory seed = makeSeed(model);

  path_planning_server::TrajectoryPreferenceSettings settings;
  settings.time_limit = 0.0;
  settings.max_iterations = 100;

  path_planning_server::TrajectoryPreferenceOptimizer optimizer(settings);
  const robot_trajectory::RobotTrajectoryPtr optimized =
      optimizer.optimize(planning_scene, seed, moveit_msgs::msg::Constraints{});

  require(optimized != nullptr, "timed-out optimizer should return a trajectory");
  require(planning_scene->isPathValid(*optimized, optimized->getGroupName()),
          "timed-out optimizer should return a valid trajectory");
  require(optimized->getWayPointCount() == seed.getWayPointCount(),
          "timed-out optimizer should preserve waypoint count");
  for (std::size_t index = 0; index < seed.getWayPointCount(); ++index)
  {
    requireEqual(jointPositions(*optimized, index), jointPositions(seed, index),
                 "timed-out optimizer should return the seed trajectory");
  }
}
}  // namespace

int main()
{
  const moveit::core::RobotModelPtr model = makeRobotModel();
  const auto planning_scene = std::make_shared<planning_scene::PlanningScene>(model);
  testJointTravelCost(model, planning_scene);
  testTcpPathLengthCost(model, planning_scene);
  testTcpXAxisWorldZCost(model, planning_scene);
  testCombinedCost(model, planning_scene);
  testTimeLimit(model, planning_scene);

  std::cout << "Trajectory preference optimizer individual-cost tests passed\n";
  return EXIT_SUCCESS;
}
