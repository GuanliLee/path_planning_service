#include "path_planning_server/ompl_waypoint_loader.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

namespace
{

void require(bool condition, const std::string & message)
{
  if (!condition) {
    std::cerr << "FAILED: " << message << '\n';
    std::exit(EXIT_FAILURE);
  }
}

}  // namespace

int main()
{
  path_planning_server::OmplWaypointLoader loader;
  path_planning_server::OmplWaypointConfig config;
  std::string message;

  require(
    loader.load("right_shelf_lift.yaml", config, message),
    "sample waypoint config should load");
  require(config.arm_name == "right", "sample arm should be right");
  require(config.waypoints.size() == 2U, "sample should contain two waypoints");
  require(
    config.waypoints.front().header.frame_id == "world",
    "sample frame should be world");

  const auto & orientation = config.waypoints.front().pose.orientation;
  const double norm = std::sqrt(
    orientation.x * orientation.x + orientation.y * orientation.y +
    orientation.z * orientation.z + orientation.w * orientation.w);
  require(std::abs(norm - 1.0) < 1e-12, "orientation should be normalized");

  require(
    !loader.load("../right_shelf_lift.yaml", config, message),
    "parent path should be rejected");
  require(
    !loader.load("missing.yaml", config, message),
    "missing config should fail");
  return EXIT_SUCCESS;
}
