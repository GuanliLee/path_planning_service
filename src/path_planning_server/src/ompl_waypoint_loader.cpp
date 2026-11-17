#include "path_planning_server/ompl_waypoint_loader.hpp"

#include <array>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <utility>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <yaml-cpp/yaml.h>

namespace path_planning_server
{

OmplWaypointLoader::OmplWaypointLoader()
: config_directory_(
    ament_index_cpp::get_package_share_directory("path_planning_server") +
    "/config/ompl_waypoints")
{
}

bool OmplWaypointLoader::load(
  const std::string & config_file,
  OmplWaypointConfig & config,
  std::string & message) const
{
  const std::filesystem::path relative_path(config_file);
  if (config_file.empty() || relative_path.is_absolute() ||
    relative_path.has_parent_path() || relative_path.filename() != relative_path)
  {
    message = "Invalid OMPL waypoint config file";
    return false;
  }

  try {
    const YAML::Node root = YAML::LoadFile(
      (std::filesystem::path(config_directory_) / relative_path).string());

    OmplWaypointConfig loaded;
    loaded.arm_name = root["arm_name"].as<std::string>();
    if (loaded.arm_name != "left" && loaded.arm_name != "right") {
      throw std::runtime_error("arm_name must be left or right");
    }

    const std::string frame_id = root["frame_id"].as<std::string>();
    if (frame_id != "world") {
      throw std::runtime_error("frame_id must be world");
    }

    const YAML::Node waypoint_nodes = root["waypoints"];
    if (!waypoint_nodes || !waypoint_nodes.IsSequence() || waypoint_nodes.size() == 0U) {
      throw std::runtime_error("waypoints must be a non-empty sequence");
    }

    loaded.waypoints.reserve(waypoint_nodes.size());
    for (const YAML::Node & waypoint_node : waypoint_nodes) {
      const auto position = waypoint_node["position"].as<std::vector<double>>();
      const auto orientation = waypoint_node["orientation"].as<std::vector<double>>();
      if (position.size() != 3U || orientation.size() != 4U) {
        throw std::runtime_error(
                "each waypoint must contain position[3] and orientation[4]");
      }

      const std::array<double, 7> values = {
        position[0], position[1], position[2],
        orientation[0], orientation[1], orientation[2], orientation[3]};
      for (const double value : values) {
        if (!std::isfinite(value)) {
          throw std::runtime_error("waypoint contains non-finite values");
        }
      }

      const double orientation_norm = std::sqrt(
        orientation[0] * orientation[0] +
        orientation[1] * orientation[1] +
        orientation[2] * orientation[2] +
        orientation[3] * orientation[3]);
      if (orientation_norm == 0.0) {
        throw std::runtime_error("waypoint orientation is zero");
      }

      geometry_msgs::msg::PoseStamped waypoint;
      waypoint.header.frame_id = frame_id;
      waypoint.pose.position.x = position[0];
      waypoint.pose.position.y = position[1];
      waypoint.pose.position.z = position[2];
      waypoint.pose.orientation.x = orientation[0] / orientation_norm;
      waypoint.pose.orientation.y = orientation[1] / orientation_norm;
      waypoint.pose.orientation.z = orientation[2] / orientation_norm;
      waypoint.pose.orientation.w = orientation[3] / orientation_norm;
      loaded.waypoints.push_back(std::move(waypoint));
    }

    config = std::move(loaded);
    return true;
  } catch (const std::exception & error) {
    message = "Failed to load OMPL waypoint config: " + std::string(error.what());
    return false;
  }
}

}  // namespace path_planning_server
