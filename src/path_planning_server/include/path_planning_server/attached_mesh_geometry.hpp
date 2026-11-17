#pragma once

#include <string>

#include <Eigen/Geometry>

#include <shape_msgs/msg/mesh.hpp>

namespace path_planning_server
{

bool attachedMeshMinimumWorldZ(
  const shape_msgs::msg::Mesh & mesh,
  const Eigen::Isometry3d & world_link6,
  const Eigen::Isometry3d & link6_mesh,
  double & minimum_world_z,
  std::string & message);

}  // namespace path_planning_server
