#include "path_planning_server/grasp_cylinder_fitter.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace
{

using path_planning_server::CylinderPoint;
using path_planning_server::FittedCylinder;

bool verifyCylinder(
  const std::vector<CylinderPoint> & points,
  const FittedCylinder & cylinder)
{
  const auto & quaternion = cylinder.orientation_xyzw;
  const double axis_x = 2.0 * (
    quaternion[0] * quaternion[2] + quaternion[3] * quaternion[1]);
  const double axis_y = 2.0 * (
    quaternion[1] * quaternion[2] - quaternion[3] * quaternion[0]);
  const double axis_z = 1.0 - 2.0 * (
    quaternion[0] * quaternion[0] + quaternion[1] * quaternion[1]);

  if (std::abs(axis_z) > 1e-10) {
    std::cerr << "Cylinder axis is not perpendicular to TCP Z\n";
    return false;
  }

  for (const auto & point : points) {
    const double dx = point.x - cylinder.center.x;
    const double dy = point.y - cylinder.center.y;
    const double dz = point.z - cylinder.center.z;
    const double axial = dx * axis_x + dy * axis_y + dz * axis_z;
    const double radial_squared = dx * dx + dy * dy + dz * dz - axial * axial;
    if (std::abs(axial) > cylinder.height * 0.5 + 1e-8 ||
      radial_squared > cylinder.radius * cylinder.radius + 1e-10)
    {
      std::cerr << "Cylinder does not enclose all input points\n";
      return false;
    }
  }
  return true;
}

bool runSyntheticTest()
{
  constexpr double angle = 0.37;
  const double axis_x = std::cos(angle);
  const double axis_y = std::sin(angle);
  const double radial_x = -axis_y;
  const double radial_y = axis_x;
  constexpr double expected_height = 0.12;
  constexpr double expected_radius = 0.025;
  std::vector<CylinderPoint> points;
  for (int height_index = 0; height_index <= 12; ++height_index) {
    const double axial = -expected_height * 0.5 +
      expected_height * height_index / 12.0;
    for (int circle_index = 0; circle_index < 36; ++circle_index) {
      const double circle_angle =
        2.0 * 3.14159265358979323846 * circle_index / 36.0;
      const double radial = expected_radius * std::cos(circle_angle);
      points.push_back({
        0.03 + axis_x * axial + radial_x * radial,
        -0.02 + axis_y * axial + radial_y * radial,
        0.01 + expected_radius * std::sin(circle_angle)});
    }
  }

  FittedCylinder cylinder;
  std::string message;
  if (!path_planning_server::fitMinimumEnclosingCylinder(
      points, cylinder, message))
  {
    std::cerr << message << '\n';
    return false;
  }
  if (std::abs(cylinder.height - expected_height) > 1e-5 ||
    std::abs(cylinder.radius - expected_radius) > 1e-5)
  {
    std::cerr << "Unexpected synthetic cylinder dimensions\n";
    return false;
  }
  return verifyCylinder(points, cylinder);
}

bool runDegenerateInputTest()
{
  FittedCylinder cylinder;
  std::string message;
  return !path_planning_server::fitMinimumEnclosingCylinder(
    {{0.0, 0.0, 0.0}}, cylinder, message);
}

}  // namespace

int main()
{
  return runSyntheticTest() && runDegenerateInputTest() ? 0 : 1;
}
