#pragma once

#include <string>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>

namespace path_planning_server
{

struct OmplWaypointConfig
{
  std::string arm_name;
  std::vector<geometry_msgs::msg::PoseStamped> waypoints;
};

class OmplWaypointLoader
{
public:
  OmplWaypointLoader();

  bool load(
    const std::string & config_file,
    OmplWaypointConfig & config,
    std::string & message) const;

private:
  std::string config_directory_;
};

}  // namespace path_planning_server
