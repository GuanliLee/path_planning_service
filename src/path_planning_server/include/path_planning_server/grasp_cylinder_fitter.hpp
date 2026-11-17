#pragma once

#include <array>
#include <string>
#include <vector>

namespace path_planning_server
{

struct CylinderPoint
{
  double x;
  double y;
  double z;
};

struct FittedCylinder
{
  double height;
  double radius;
  CylinderPoint center;
  std::array<double, 4> orientation_xyzw;
};

bool fitMinimumEnclosingCylinder(
  const std::vector<CylinderPoint> & points,
  FittedCylinder & cylinder,
  std::string & message);

}  // namespace path_planning_server
