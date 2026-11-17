#include "path_planning_server/attached_mesh_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace path_planning_server
{

bool attachedMeshMinimumWorldZ(
  const shape_msgs::msg::Mesh & mesh,
  const Eigen::Isometry3d & world_link6,
  const Eigen::Isometry3d & link6_mesh,
  double & minimum_world_z,
  std::string & message)
{
  if (mesh.vertices.empty()) {
    message = "Attached grasp mesh has no vertices";
    return false;
  }

  const Eigen::Isometry3d world_mesh = world_link6 * link6_mesh;
  minimum_world_z = std::numeric_limits<double>::infinity();
  for (const auto & vertex : mesh.vertices) {
    if (!std::isfinite(vertex.x) || !std::isfinite(vertex.y) ||
      !std::isfinite(vertex.z))
    {
      message = "Attached grasp mesh contains a non-finite vertex";
      return false;
    }
    const Eigen::Vector3d vertex_world = world_mesh * Eigen::Vector3d(
      vertex.x, vertex.y, vertex.z);
    if (!std::isfinite(vertex_world.z())) {
      message = "Attached grasp mesh produced a non-finite world Z";
      return false;
    }
    minimum_world_z = std::min(minimum_world_z, vertex_world.z());
  }

  message.clear();
  return true;
}

}  // namespace path_planning_server
