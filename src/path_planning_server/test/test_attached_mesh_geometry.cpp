#include "path_planning_server/attached_mesh_geometry.hpp"

#include <cmath>
#include <iostream>
#include <string>

#include <Eigen/Geometry>

namespace
{

geometry_msgs::msg::Point point(const double x, const double y, const double z)
{
  geometry_msgs::msg::Point result;
  result.x = x;
  result.y = y;
  result.z = z;
  return result;
}

bool runTranslatedMeshTest()
{
  shape_msgs::msg::Mesh mesh;
  mesh.vertices = {
    point(-0.02, -0.01, 0.0),
    point(0.02, 0.01, 0.15),
  };
  Eigen::Isometry3d world_link6 = Eigen::Isometry3d::Identity();
  world_link6.translation().z() = 0.30;
  Eigen::Isometry3d link6_mesh = Eigen::Isometry3d::Identity();
  link6_mesh.translation().z() = -0.07;

  double minimum_world_z = 0.0;
  std::string message;
  if (!path_planning_server::attachedMeshMinimumWorldZ(
      mesh, world_link6, link6_mesh, minimum_world_z, message))
  {
    std::cerr << message << '\n';
    return false;
  }
  if (std::abs(minimum_world_z - 0.23) > 1.0e-9) {
    std::cerr << "Unexpected translated mesh minimum Z: "
              << minimum_world_z << '\n';
    return false;
  }
  return true;
}

bool runRotatedMeshTest()
{
  shape_msgs::msg::Mesh mesh;
  mesh.vertices = {
    point(0.0, -0.03, 0.0),
    point(0.0, 0.02, 0.10),
  };
  Eigen::Isometry3d world_link6 = Eigen::Isometry3d::Identity();
  world_link6.translation().z() = 0.40;
  Eigen::Isometry3d link6_mesh = Eigen::Isometry3d::Identity();
  link6_mesh.linear() = Eigen::AngleAxisd(
    1.5707963267948966, Eigen::Vector3d::UnitX()).toRotationMatrix();

  double minimum_world_z = 0.0;
  std::string message;
  if (!path_planning_server::attachedMeshMinimumWorldZ(
      mesh, world_link6, link6_mesh, minimum_world_z, message))
  {
    std::cerr << message << '\n';
    return false;
  }
  if (std::abs(minimum_world_z - 0.37) > 1.0e-8) {
    std::cerr << "Unexpected rotated mesh minimum Z: "
              << minimum_world_z << '\n';
    return false;
  }
  return true;
}

bool runEmptyMeshTest()
{
  shape_msgs::msg::Mesh mesh;
  double minimum_world_z = 0.0;
  std::string message;
  return !path_planning_server::attachedMeshMinimumWorldZ(
    mesh,
    Eigen::Isometry3d::Identity(),
    Eigen::Isometry3d::Identity(),
    minimum_world_z,
    message);
}

}  // namespace

int main()
{
  return runTranslatedMeshTest() && runRotatedMeshTest() && runEmptyMeshTest() ?
         0 : 1;
}
