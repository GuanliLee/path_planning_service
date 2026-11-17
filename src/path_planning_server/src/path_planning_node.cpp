#include "path_planning_server/path_planning_node.hpp"

#include "path_planning_server/attached_mesh_geometry.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <exception>
#include <functional>
#include <future>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <boost/variant/get.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <geometric_shapes/mesh_operations.h>
#include <geometric_shapes/shape_operations.h>
#include <moveit/kinematic_constraints/utils.hpp>
#include <moveit_msgs/msg/allowed_collision_entry.hpp>
#include <moveit_msgs/msg/attached_collision_object.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/msg/link_padding.hpp>
#include <moveit_msgs/msg/planning_scene.hpp>
#include <moveit_msgs/msg/planning_scene_components.hpp>
#include <moveit_msgs/msg/position_constraint.hpp>
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/robot_trajectory/robot_trajectory.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <moveit/trajectory_processing/time_optimal_trajectory_generation.hpp>
#include <moveit/utils/moveit_error_code.hpp>
#include <rclcpp/duration.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <tf2/exceptions.hpp>
#include <tf2/time.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include "path_planning_server/grasp_cylinder_fitter.hpp"

namespace path_planning_server
{

namespace
{

constexpr char kLeftTrajectoryAction[] =
  "/left/arm_controller/follow_joint_trajectory";
constexpr char kRightTrajectoryAction[] =
  "/right/arm_controller/follow_joint_trajectory";
class ActiveRequestGuard
{
public:
  ActiveRequestGuard(std::atomic<bool> & active, std::atomic<bool> & canceled)
  : active_(active), canceled_(canceled)
  {
    active_.store(true);
  }
  ~ActiveRequestGuard()
  {
    active_.store(false);
    canceled_.store(false);
  }
private:
  std::atomic<bool> & active_;
  std::atomic<bool> & canceled_;
};

class PlanningPipelineGuard
{
public:
  explicit PlanningPipelineGuard(
    moveit::planning_interface::MoveGroupInterface & move_group)
  : move_group_(move_group)
  {
  }

  ~PlanningPipelineGuard()
  {
    move_group_.setPlanningPipelineId("ompl");
  }

private:
  moveit::planning_interface::MoveGroupInterface & move_group_;
};

constexpr char kLeftGripperAction[] =
  "/left/gripper_controller/follow_joint_trajectory";
constexpr char kRightGripperAction[] =
  "/right/gripper_controller/follow_joint_trajectory";
constexpr double kMaxGripperOpening = 0.1;
constexpr double kIkCheckTimeout = 0.05;
constexpr double kJointTargetTolerance = 1.0e-3;
constexpr std::chrono::seconds kActionServerWaitTimeout{5};
constexpr std::chrono::seconds kPlanningSceneServiceWaitTimeout{2};
using SteadyClock = std::chrono::steady_clock;

double elapsedMs(const SteadyClock::time_point & start)
{
  return std::chrono::duration<double, std::milli>(SteadyClock::now() - start).count();
}

const sensor_msgs::msg::PointField * findPointField(
  const sensor_msgs::msg::PointCloud2 & cloud,
  const std::string & name)
{
  const auto field = std::find_if(
    cloud.fields.begin(), cloud.fields.end(),
    [&name](const auto & candidate) {return candidate.name == name;});
  return field == cloud.fields.end() ? nullptr : &*field;
}

float readFloat32(const std::uint8_t * data, const bool swap_bytes)
{
  std::array<std::uint8_t, sizeof(float)> bytes;
  std::copy_n(data, bytes.size(), bytes.begin());
  if (swap_bytes) {
    std::reverse(bytes.begin(), bytes.end());
  }
  float value;
  std::memcpy(&value, bytes.data(), sizeof(value));
  return value;
}

Eigen::Isometry3d poseToEigen(const geometry_msgs::msg::Pose & pose)
{
  Eigen::Quaterniond orientation(
    pose.orientation.w, pose.orientation.x,
    pose.orientation.y, pose.orientation.z);
  orientation.normalize();
  Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
  transform.linear() = orientation.toRotationMatrix();
  transform.translation() = Eigen::Vector3d(
    pose.position.x, pose.position.y, pose.position.z);
  return transform;
}

geometry_msgs::msg::Pose eigenToPose(const Eigen::Isometry3d & transform)
{
  geometry_msgs::msg::Pose pose;
  pose.position.x = transform.translation().x();
  pose.position.y = transform.translation().y();
  pose.position.z = transform.translation().z();
  const Eigen::Quaterniond orientation(transform.linear());
  pose.orientation.x = orientation.x();
  pose.orientation.y = orientation.y();
  pose.orientation.z = orientation.z();
  pose.orientation.w = orientation.w();
  return pose;
}

std::string percentEncodeUriPathSegment(const std::string & segment)
{
  constexpr char hex_digits[] = "0123456789ABCDEF";
  std::string encoded;
  encoded.reserve(segment.size());
  for (const unsigned char character : segment) {
    const bool unreserved =
      (character >= 'a' && character <= 'z') ||
      (character >= 'A' && character <= 'Z') ||
      (character >= '0' && character <= '9') ||
      character == '-' || character == '_' ||
      character == '.' || character == '~';
    if (unreserved) {
      encoded.push_back(static_cast<char>(character));
    } else {
      encoded.push_back('%');
      encoded.push_back(hex_digits[character >> 4]);
      encoded.push_back(hex_digits[character & 0x0F]);
    }
  }
  return encoded;
}

bool isMeshInputEmpty(
  const path_planning_interfaces::msg::AxisGraspInput & input)
{
  const std::array<double, 9> values = {
    input.axis_start_world.x,
    input.axis_start_world.y,
    input.axis_start_world.z,
    input.axis_end_world.x,
    input.axis_end_world.y,
    input.axis_end_world.z,
    input.grasp_point_world.x,
    input.grasp_point_world.y,
    input.grasp_point_world.z
  };
  return std::all_of(
    values.begin(), values.end(), [](const double value) {return value == 0.0;});
}

bool hasFiniteMeshInput(
  const path_planning_interfaces::msg::AxisGraspInput & input)
{
  const std::array<double, 9> values = {
    input.axis_start_world.x,
    input.axis_start_world.y,
    input.axis_start_world.z,
    input.axis_end_world.x,
    input.axis_end_world.y,
    input.axis_end_world.z,
    input.grasp_point_world.x,
    input.grasp_point_world.y,
    input.grasp_point_world.z
  };
  return std::all_of(
    values.begin(), values.end(), [](const double value) {return std::isfinite(value);});
}

std::size_t ensureAcmEntry(
  moveit_msgs::msg::AllowedCollisionMatrix & acm,
  const std::string & name)
{
  const auto existing =
    std::find(acm.entry_names.begin(), acm.entry_names.end(), name);
  if (existing != acm.entry_names.end()) {
    const std::size_t size = acm.entry_names.size();
    for (auto & entry : acm.entry_values) {
      entry.enabled.resize(size, false);
    }
    return static_cast<std::size_t>(
      std::distance(acm.entry_names.begin(), existing));
  }

  const std::size_t index = acm.entry_names.size();
  acm.entry_names.push_back(name);
  for (auto & entry : acm.entry_values) {
    entry.enabled.resize(index + 1, false);
  }
  moveit_msgs::msg::AllowedCollisionEntry entry;
  entry.enabled.assign(index + 1, false);
  acm.entry_values.push_back(entry);
  return index;
}

bool hasAcmEntry(
  const moveit_msgs::msg::AllowedCollisionMatrix & acm,
  const std::string & name)
{
  return std::find(acm.entry_names.begin(), acm.entry_names.end(), name) !=
         acm.entry_names.end();
}

void setAcmPair(
  moveit_msgs::msg::AllowedCollisionMatrix & acm,
  const std::string & first,
  const std::string & second,
  const bool allowed,
  const bool add_missing)
{
  if (!add_missing && (!hasAcmEntry(acm, first) || !hasAcmEntry(acm, second))) {
    return;
  }

  const std::size_t first_index = ensureAcmEntry(acm, first);
  const std::size_t second_index = ensureAcmEntry(acm, second);
  const std::size_t size = acm.entry_names.size();
  for (auto & entry : acm.entry_values) {
    entry.enabled.resize(size, false);
  }
  acm.entry_values[first_index].enabled[second_index] = allowed;
  acm.entry_values[second_index].enabled[first_index] = allowed;
}

}  // namespace

PathPlanningNode::PathPlanningNode(const rclcpp::NodeOptions & options)
: Node("path_planning_server", options)
{
  const auto string_parameter =
    [this](const std::string & name, const std::string & default_value) {
      if (!has_parameter(name)) {
        return declare_parameter<std::string>(name, default_value);
      }
      return get_parameter(name).as_string();
    };
  const auto double_parameter =
    [this](const std::string & name, const double default_value) {
      if (!has_parameter(name)) {
        return declare_parameter<double>(name, default_value);
      }
      return get_parameter(name).as_double();
    };
  const auto integer_parameter =
    [this](const std::string & name, const std::int64_t default_value) {
      if (!has_parameter(name)) {
        return declare_parameter<std::int64_t>(name, default_value);
      }
      return get_parameter(name).as_int();
    };
  const auto bool_parameter =
    [this](const std::string & name, const bool default_value) {
      if (!has_parameter(name)) {
        return declare_parameter<bool>(name, default_value);
      }
      return get_parameter(name).as_bool();
    };

  world_frame_ = string_parameter("world_frame", "world");
  left_planning_group_ = string_parameter("left_planning_group", "left_arm");
  right_planning_group_ = string_parameter("right_planning_group", "right_arm");
  left_end_effector_link_ =
    string_parameter("left_end_effector_link", "left_link6");
  right_end_effector_link_ =
    string_parameter("right_end_effector_link", "right_link6");
  scene_index_file_ =
    string_parameter("scene_index_file", "config/scenes/scene_index.yaml");
  default_scene_id_ =
    static_cast<std::uint32_t>(integer_parameter("default_scene_id", 1));
  planning_time_ = double_parameter("planning_time", 5.0);
  planning_attempts_ =
    static_cast<unsigned int>(integer_parameter("planning_attempts", 1));
  velocity_scaling_ = double_parameter("velocity_scaling", 0.1);
  acceleration_scaling_ = double_parameter("acceleration_scaling", 0.1);
  position_tolerance_ = double_parameter("position_tolerance", 0.005);
  orientation_tolerance_ = double_parameter("orientation_tolerance", 0.05);
  gripper_move_duration_ = double_parameter("gripper_move_duration", 1.0);
  gripper_release_duration_ = double_parameter("gripper_release_duration", 3.0);
  if (!std::isfinite(gripper_move_duration_) || gripper_move_duration_ <= 0.0) {
    throw std::invalid_argument("gripper_move_duration must be finite and positive");
  }
  if (!std::isfinite(gripper_release_duration_) || gripper_release_duration_ <= 0.0) {
    throw std::invalid_argument("gripper_release_duration must be finite and positive");
  }
  gripper_effort_ = double_parameter("gripper_effort", 4.0);
  if (!std::isfinite(gripper_effort_) || gripper_effort_ < 0.0 || gripper_effort_ > 5.0) {
    throw std::invalid_argument("gripper_effort must be finite and in [0.0, 5.0]");
  }
  gripper_contact_opening_threshold_ =
    double_parameter("gripper_contact_opening_threshold", 0.02);
  gripper_contact_goal_tolerance_ =
    double_parameter("gripper_contact_goal_tolerance", 0.04);
  gripper_contact_goal_time_tolerance_ =
    double_parameter("gripper_contact_goal_time_tolerance", 0.3);
  if (!std::isfinite(gripper_contact_goal_time_tolerance_) ||
    gripper_contact_goal_time_tolerance_ < 0.0)
  {
    throw std::invalid_argument("gripper_contact_goal_time_tolerance must be finite and non-negative");
  }
  treat_gripper_contact_abort_as_success_ =
    bool_parameter("treat_gripper_contact_abort_as_success", true);
  position_constraint_min_x_ =
    double_parameter("position_constraint.min_x_m", 0.0);
  position_constraint_max_x_ =
    double_parameter("position_constraint.max_x_m", 0.65);
  left_position_constraint_min_y_ =
    double_parameter("position_constraint.left.min_y_m", -0.45);
  left_position_constraint_max_y_ =
    double_parameter("position_constraint.left.max_y_m", 0.3);
  right_position_constraint_min_y_ =
    double_parameter("position_constraint.right.min_y_m", -0.3);
  right_position_constraint_max_y_ =
    double_parameter("position_constraint.right.max_y_m", 0.45);
  position_constraint_min_z_ =
    double_parameter("position_constraint.min_z_m", 0.05);
  position_constraint_max_z_ =
    double_parameter("position_constraint.max_z_m", 0.7);
  right_camera_stand_padding_ =
    double_parameter("robot_link_padding.right_camera_stand_m", 0.0);
  right_camera_link_padding_ =
    double_parameter("robot_link_padding.right_camera_link_m", 0.0);
  for (const auto & [name, padding] : std::array<std::pair<const char *, double>, 2>{
      std::pair{"robot_link_padding.right_camera_stand_m", right_camera_stand_padding_},
      std::pair{"robot_link_padding.right_camera_link_m", right_camera_link_padding_}})
  {
    if (!std::isfinite(padding) || padding < 0.0 || padding > 0.1) {
      throw std::invalid_argument(
              std::string(name) + " must be finite and in [0.0, 0.1] m");
    }
  }
  const std::string grasp_object_type =
    string_parameter("grasp_object.type", "ellipsoid");
  if (grasp_object_type == "ellipsoid") {
    grasp_object_type_ = GraspObjectType::Ellipsoid;
  } else if (grasp_object_type == "cylinder") {
    grasp_object_type_ = GraspObjectType::Cylinder;
  } else if (grasp_object_type == "mesh") {
    grasp_object_type_ = GraspObjectType::Mesh;
  } else {
    throw std::invalid_argument(
            "grasp_object.type must be ellipsoid, cylinder, or mesh");
  }
  grasp_object_tcp_offset_z_ =
    double_parameter("grasp_object.tcp_offset_z_m", 0.1358);
  if (!std::isfinite(grasp_object_tcp_offset_z_)) {
    throw std::invalid_argument("grasp_object.tcp_offset_z_m must be finite");
  }
  grasp_mesh_root_resource_ = string_parameter(
    "grasp_mesh.root_resource",
    "package://path_planning_server/meshes/beverages");
  const std::string grasp_ellipsoid_mesh_resource = string_parameter(
    "grasp_ellipsoid.mesh_resource",
    "package://path_planning_server/meshes/grasp_ellipsoid/unit_sphere.stl");
  const double grasp_ellipsoid_radius_x =
    double_parameter("grasp_ellipsoid.radius_x_m", 0.06);
  const double grasp_ellipsoid_radius_y =
    double_parameter("grasp_ellipsoid.radius_y_m", 0.06);
  const double grasp_ellipsoid_radius_z =
    double_parameter("grasp_ellipsoid.radius_z_m", 0.12);
  grasp_ellipsoid_center_z_ =
    double_parameter("grasp_ellipsoid.center_z_m", 0.1358);
  execute_trajectory_ = bool_parameter("execute_trajectory", false);

  std::unique_ptr<shapes::Mesh> grasp_ellipsoid_shape(
    shapes::createMeshFromResource(
      grasp_ellipsoid_mesh_resource,
      Eigen::Vector3d(
        grasp_ellipsoid_radius_x,
        grasp_ellipsoid_radius_y,
        grasp_ellipsoid_radius_z)));
  if (!grasp_ellipsoid_shape) {
    throw std::runtime_error(
      "Failed to load grasp ellipsoid mesh: " + grasp_ellipsoid_mesh_resource);
  }
  shapes::ShapeMsg grasp_ellipsoid_message;
  if (!shapes::constructMsgFromShape(
      grasp_ellipsoid_shape.get(), grasp_ellipsoid_message))
  {
    throw std::runtime_error("Failed to convert grasp ellipsoid mesh");
  }
  const auto * grasp_ellipsoid_mesh =
    boost::get<shape_msgs::msg::Mesh>(&grasp_ellipsoid_message);
  if (!grasp_ellipsoid_mesh) {
    throw std::runtime_error("Grasp ellipsoid resource is not a mesh");
  }
  grasp_ellipsoid_mesh_ = *grasp_ellipsoid_mesh;
}

void PathPlanningNode::initialize()
{
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
  tf_listener_ =
    std::make_shared<tf2_ros::TransformListener>(*tf_buffer_, shared_from_this(), true);

  left_move_group_ = std::make_unique<MoveGroupInterface>(
    shared_from_this(), left_planning_group_, tf_buffer_);
  right_move_group_ = std::make_unique<MoveGroupInterface>(
    shared_from_this(), right_planning_group_, tf_buffer_);
  configureMoveGroup(*left_move_group_, left_end_effector_link_);
  configureMoveGroup(*right_move_group_, right_end_effector_link_);

  planning_scene_ =
    std::make_shared<moveit::planning_interface::PlanningSceneInterface>();
  get_planning_scene_client_ = create_client<GetPlanningScene>("/get_planning_scene");
  grasp_visual_publisher_ = create_publisher<visualization_msgs::msg::MarkerArray>(
    "/grasp_object_visualization",
    rclcpp::QoS(rclcpp::KeepLast(10)).reliable().transient_local());
  deleteMeshVisual("left");
  deleteMeshVisual("right");
  static_scene_loader_ =
    std::make_unique<StaticSceneLoader>(scene_index_file_, planning_scene_);
  ompl_waypoint_loader_ = std::make_unique<OmplWaypointLoader>();

  cancel_callback_group_ =
    create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  std::string scene_message;
  if (!static_scene_loader_->loadScene(default_scene_id_, scene_message)) {
    throw std::runtime_error(scene_message);
  }
  applyConfiguredRobotLinkPadding();

  cancel_service_ = create_service<Trigger>(
    "/cancel_plan_execution",
    std::bind(
      &PathPlanningNode::handleCancelRequest, this,
      std::placeholders::_1, std::placeholders::_2),
    rclcpp::ServicesQoS(),
    cancel_callback_group_);

  action_callback_group_ =
    create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  left_trajectory_client_ = rclcpp_action::create_client<FollowJointTrajectory>(
    shared_from_this(), kLeftTrajectoryAction, action_callback_group_);
  right_trajectory_client_ = rclcpp_action::create_client<FollowJointTrajectory>(
    shared_from_this(), kRightTrajectoryAction, action_callback_group_);
  left_gripper_client_ = rclcpp_action::create_client<FollowJointTrajectory>(
    shared_from_this(), kLeftGripperAction, action_callback_group_);
  right_gripper_client_ = rclcpp_action::create_client<FollowJointTrajectory>(
    shared_from_this(), kRightGripperAction, action_callback_group_);

  service_callback_group_ =
    create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  service_ = create_service<PlanToPose>(
    "/plan_to_pose",
    std::bind(
      &PathPlanningNode::handlePlanRequest, this,
      std::placeholders::_1, std::placeholders::_2),
    rclcpp::ServicesQoS(),
    service_callback_group_);
  service_v2_ = create_service<PlanToPoseV2>(
    "/plan_to_pose_v2",
    std::bind(
      &PathPlanningNode::handlePlanV2Request, this,
      std::placeholders::_1, std::placeholders::_2),
    rclcpp::ServicesQoS(),
    service_callback_group_);
  service_v3_ = create_service<PlanToPoseV3>(
    "/plan_to_pose_v3",
    std::bind(
      &PathPlanningNode::handlePlanV3Request, this,
      std::placeholders::_1, std::placeholders::_2),
    rclcpp::ServicesQoS(),
    service_callback_group_);
  cartesian_service_ = create_service<PlanCartesianPath>(
    "/plan_cartesian_path",
    std::bind(
      &PathPlanningNode::handleCartesianPlanRequest, this,
      std::placeholders::_1, std::placeholders::_2),
    rclcpp::ServicesQoS(),
    service_callback_group_);
  joint_service_ = create_service<PlanToJoints>(
    "/plan_to_joints",
    std::bind(
      &PathPlanningNode::handleJointPlanRequest, this,
      std::placeholders::_1, std::placeholders::_2),
    rclcpp::ServicesQoS(),
    service_callback_group_);

  const char * grasp_object_type_name = "ellipsoid";
  if (grasp_object_type_ == GraspObjectType::Cylinder) {
    grasp_object_type_name = "cylinder";
  } else if (grasp_object_type_ == GraspObjectType::Mesh) {
    grasp_object_type_name = "mesh";
  }

  RCLCPP_INFO(
    get_logger(),
    "Ready: /plan_to_pose, /plan_to_pose_v2, /plan_to_pose_v3, "
    "/plan_cartesian_path, /plan_to_joints, "
    "and /cancel_plan_execution, "
    "groups [%s, %s], scene %u, grasp object: %s, execute trajectory: %s",
    left_planning_group_.c_str(), right_planning_group_.c_str(), default_scene_id_,
    grasp_object_type_name,
    execute_trajectory_ ? "true" : "false");
}

void PathPlanningNode::applyConfiguredRobotLinkPadding()
{
  moveit_msgs::msg::LinkPadding camera_stand_padding;
  camera_stand_padding.link_name = "right_camera_stand";
  camera_stand_padding.padding = right_camera_stand_padding_;

  moveit_msgs::msg::LinkPadding camera_link_padding;
  camera_link_padding.link_name = "right_camera_link";
  camera_link_padding.padding = right_camera_link_padding_;

  moveit_msgs::msg::PlanningScene scene;
  scene.is_diff = true;
  scene.link_padding = {camera_stand_padding, camera_link_padding};
  if (!planning_scene_->applyPlanningScene(scene)) {
    throw std::runtime_error("Failed to apply configured robot link padding");
  }

  RCLCPP_INFO(
    get_logger(),
    "Applied right wrist camera collision padding: stand=%.3f m, camera=%.3f m",
    right_camera_stand_padding_, right_camera_link_padding_);
}

void PathPlanningNode::configureMoveGroup(
  MoveGroupInterface & move_group,
  const std::string & end_effector_link)
{
  move_group.setPoseReferenceFrame(world_frame_);
  if (!move_group.setEndEffectorLink(end_effector_link)) {
    throw std::runtime_error("End effector link not found: " + end_effector_link);
  }
  move_group.setPlanningTime(planning_time_);
  move_group.setNumPlanningAttempts(planning_attempts_);
  move_group.setMaxVelocityScalingFactor(velocity_scaling_);
  move_group.setMaxAccelerationScalingFactor(acceleration_scaling_);
  move_group.setGoalPositionTolerance(position_tolerance_);
  move_group.setGoalOrientationTolerance(orientation_tolerance_);
}

moveit_msgs::msg::Constraints PathPlanningNode::makeLink6PositionConstraint(
  const std::string & end_effector_link) const
{
  const double min_y = end_effector_link == left_end_effector_link_ ?
    left_position_constraint_min_y_ : right_position_constraint_min_y_;
  const double max_y = end_effector_link == left_end_effector_link_ ?
    left_position_constraint_max_y_ : right_position_constraint_max_y_;

  moveit_msgs::msg::PositionConstraint position_constraint;
  position_constraint.header.frame_id = world_frame_;
  position_constraint.link_name = end_effector_link;
  position_constraint.target_point_offset.z = grasp_object_tcp_offset_z_;
  position_constraint.weight = 1.0;

  shape_msgs::msg::SolidPrimitive region;
  region.type = shape_msgs::msg::SolidPrimitive::BOX;
  region.dimensions = {
    position_constraint_max_x_ - position_constraint_min_x_,
    max_y - min_y,
    position_constraint_max_z_ - position_constraint_min_z_};

  geometry_msgs::msg::Pose region_pose;
  region_pose.position.x =
    (position_constraint_min_x_ + position_constraint_max_x_) / 2.0;
  region_pose.position.y = (min_y + max_y) / 2.0;
  region_pose.position.z =
    (position_constraint_min_z_ + position_constraint_max_z_) / 2.0;
  region_pose.orientation.w = 1.0;

  position_constraint.constraint_region.primitives.push_back(region);
  position_constraint.constraint_region.primitive_poses.push_back(region_pose);

  moveit_msgs::msg::Constraints constraints;
  constraints.position_constraints.push_back(position_constraint);
  return constraints;
}

moveit_msgs::msg::TrajectoryConstraints PathPlanningNode::makeWaypointConstraints(
  const OmplWaypointConfig & config,
  const std::string & end_effector_link) const
{
  moveit_msgs::msg::TrajectoryConstraints constraints;
  constraints.constraints.reserve(config.waypoints.size());
  for (const auto & waypoint : config.waypoints) {
    constraints.constraints.push_back(
      kinematic_constraints::constructGoalConstraints(
        end_effector_link, waypoint, position_tolerance_, orientation_tolerance_));
  }
  return constraints;
}

void PathPlanningNode::handleCancelRequest(
  const std::shared_ptr<Trigger::Request> request,
  std::shared_ptr<Trigger::Response> response)
{
  (void)request;
  cancel_requested_.store(true);
  try {
    for (const auto & client : {
        left_trajectory_client_, right_trajectory_client_,
        left_gripper_client_, right_gripper_client_})
    {
      if (client) {
        client->async_cancel_all_goals();
      }
    }
  } catch (const std::exception & error) {
    response->success = false;
    response->message = std::string("failed to request action cancellation: ") + error.what();
    return;
  }
  response->success = true;
  response->message = request_active_.load() ? "cancel requested for active planning/execution" : "cancel queued for pending planning request";
  RCLCPP_WARN(get_logger(), "%s", response->message.c_str());
}

void PathPlanningNode::handlePlanRequest(
  const std::shared_ptr<PlanToPose::Request> request,
  std::shared_ptr<PlanToPose::Response> response)
{
  const auto request_start = SteadyClock::now();
  response->success = false;
  if (cancel_requested_.exchange(false)) {
    response->message = "Execution canceled before planning started";
    return;
  }
  ActiveRequestGuard active_request(request_active_, cancel_requested_);


  MoveGroupInterface * move_group = nullptr;
  std::string planning_group;
  std::string end_effector_link;
  if (request->arm_name == "left") {
    move_group = left_move_group_.get();
    planning_group = left_planning_group_;
    end_effector_link = left_end_effector_link_;
  } else if (request->arm_name == "right") {
    move_group = right_move_group_.get();
    planning_group = right_planning_group_;
    end_effector_link = right_end_effector_link_;
  } else {
    response->message = "Invalid arm name";
    return;
  }

  if (!static_scene_loader_->hasScene(request->scene_id)) {
    response->message = "Scene ID not found";
    return;
  }
  if (request->target_pose.header.frame_id.empty()) {
    response->message = "Target frame is empty";
    return;
  }

  const auto & pose = request->target_pose.pose;
  const std::array<double, 7> values = {
    pose.position.x, pose.position.y, pose.position.z,
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
  };
  if (!std::all_of(values.begin(), values.end(), [](const double value) {
      return std::isfinite(value);
    }))
  {
    response->message = "Target pose contains non-finite values";
    return;
  }

  const double quaternion_norm_squared =
    pose.orientation.x * pose.orientation.x +
    pose.orientation.y * pose.orientation.y +
    pose.orientation.z * pose.orientation.z +
    pose.orientation.w * pose.orientation.w;
  if (quaternion_norm_squared == 0.0) {
    response->message = "Target orientation is zero";
    return;
  }
  if (request->gripper_command &&
    (!std::isfinite(request->gripper_opening_m) ||
    request->gripper_opening_m < 0.0 ||
    request->gripper_opening_m > kMaxGripperOpening))
  {
    response->message = "Gripper opening must be finite and in [0.0, 0.1] meters";
    return;
  }
  RCLCPP_INFO(
    get_logger(),
    "[TIMING] plan_to_pose phase=request_start arm=%s scene=%u gripper=%s "
    "opening=%.4f keep_ellipsoid=%s frame=%s",
    request->arm_name.c_str(), static_cast<unsigned int>(request->scene_id),
    request->gripper_command ? "true" : "false", request->gripper_opening_m,
    request->keep_grasp_ellipsoid ? "true" : "false",
    request->target_pose.header.frame_id.c_str());

  std::string scene_message;
  auto phase_start = SteadyClock::now();
  if (!static_scene_loader_->loadScene(request->scene_id, scene_message)) {
    response->message = scene_message;
    RCLCPP_INFO(
      get_logger(),
      "[TIMING] plan_to_pose phase=load_scene success=false elapsed_ms=%.1f total_ms=%.1f "
      "message=%s",
      elapsedMs(phase_start), elapsedMs(request_start), response->message.c_str());
    return;
  }
  RCLCPP_INFO(
    get_logger(),
    "[TIMING] plan_to_pose phase=load_scene success=true elapsed_ms=%.1f",
    elapsedMs(phase_start));

  if (!request->keep_grasp_ellipsoid) {
    std::string ellipsoid_message;
    phase_start = SteadyClock::now();
    if (!setGraspEllipsoid(request->arm_name, false, ellipsoid_message)) {
      response->message = "Failed to clear grasp ellipsoid before planning: " + ellipsoid_message;
      RCLCPP_INFO(
        get_logger(),
        "[TIMING] plan_to_pose phase=clear_grasp_ellipsoid_before_plan success=false "
        "elapsed_ms=%.1f total_ms=%.1f message=%s",
        elapsedMs(phase_start), elapsedMs(request_start), response->message.c_str());
      return;
    }
    RCLCPP_INFO(
      get_logger(),
      "[TIMING] plan_to_pose phase=clear_grasp_ellipsoid_before_plan success=true elapsed_ms=%.1f",
      elapsedMs(phase_start));
  }

  geometry_msgs::msg::PoseStamped target_pose;
  phase_start = SteadyClock::now();
  if (!transformTarget(request->target_pose, target_pose, response->message)) {
    RCLCPP_INFO(
      get_logger(),
      "[TIMING] plan_to_pose phase=transform_target success=false elapsed_ms=%.1f "
      "total_ms=%.1f message=%s",
      elapsedMs(phase_start), elapsedMs(request_start), response->message.c_str());
    return;
  }
  RCLCPP_INFO(
    get_logger(),
    "[TIMING] plan_to_pose phase=transform_target success=true elapsed_ms=%.1f",
    elapsedMs(phase_start));
  auto & orientation = target_pose.pose.orientation;
  const double orientation_norm = std::sqrt(
    orientation.x * orientation.x +
    orientation.y * orientation.y +
    orientation.z * orientation.z +
    orientation.w * orientation.w);
  orientation.x /= orientation_norm;
  orientation.y /= orientation_norm;
  orientation.z /= orientation_norm;
  orientation.w /= orientation_norm;

  phase_start = SteadyClock::now();
  auto current_state = move_group->getCurrentState(1.0);
  if (!current_state) {
    response->message = "Current robot state unavailable";
    RCLCPP_INFO(
      get_logger(),
      "[TIMING] plan_to_pose phase=get_current_state success=false elapsed_ms=%.1f "
      "total_ms=%.1f message=%s",
      elapsedMs(phase_start), elapsedMs(request_start), response->message.c_str());
    return;
  }
  RCLCPP_INFO(
    get_logger(),
    "[TIMING] plan_to_pose phase=get_current_state success=true elapsed_ms=%.1f",
    elapsedMs(phase_start));

  phase_start = SteadyClock::now();
  if (!validateTargetPose(
      *move_group, *current_state, planning_group, end_effector_link,
      target_pose, request->keep_grasp_ellipsoid, response->message))
  {
    RCLCPP_INFO(
      get_logger(),
      "[TIMING] plan_to_pose phase=validate_target success=false elapsed_ms=%.1f "
      "total_ms=%.1f message=%s",
      elapsedMs(phase_start), elapsedMs(request_start), response->message.c_str());
    return;
  }
  RCLCPP_INFO(
    get_logger(),
    "[TIMING] plan_to_pose phase=validate_target success=true elapsed_ms=%.1f",
    elapsedMs(phase_start));

  move_group->clearPoseTargets();
  // CurrentStateMonitor only contains joints and TF. Serializing that state as
  // a complete start state would clear attached collision objects, including
  // the grasp ellipsoid, from this planning request. Use an empty diff start
  // state so MoveIt retains attached objects from its live PlanningScene.
  move_group->setStartStateToCurrentState();
  if (!move_group->setPoseTarget(target_pose, end_effector_link)) {
    response->message = "Failed to set pose target";
    return;
  }

  MoveGroupInterface::Plan plan;
  phase_start = SteadyClock::now();
  const auto result = move_group->plan(plan);
  move_group->clearPoseTargets();

  if (cancel_requested_.load()) {
    response->message = "Execution canceled during planning";
    RCLCPP_INFO(
      get_logger(),
      "[TIMING] plan_to_pose phase=planning canceled=true elapsed_ms=%.1f total_ms=%.1f",
      elapsedMs(phase_start), elapsedMs(request_start));
    return;
  }
  if (result != moveit::core::MoveItErrorCode::SUCCESS) {
    response->message = "Planning failed";
    RCLCPP_INFO(
      get_logger(),
      "[TIMING] plan_to_pose phase=planning success=false elapsed_ms=%.1f total_ms=%.1f "
      "message=%s",
      elapsedMs(phase_start), elapsedMs(request_start), response->message.c_str());
    return;
  }
  RCLCPP_INFO(
    get_logger(),
    "[TIMING] plan_to_pose phase=planning success=true elapsed_ms=%.1f points=%zu",
    elapsedMs(phase_start), plan.trajectory.joint_trajectory.points.size());
  if (plan.trajectory.joint_trajectory.points.empty()) {
    response->message = "Planned trajectory is empty";
    RCLCPP_INFO(
      get_logger(),
      "[TIMING] plan_to_pose phase=planning success=false elapsed_ms=%.1f total_ms=%.1f "
      "message=%s",
      elapsedMs(phase_start), elapsedMs(request_start), response->message.c_str());
    return;
  }

  response->trajectory = plan.trajectory;
  if (!execute_trajectory_) {
    response->success = true;
    response->message = "Planning succeeded";
    RCLCPP_INFO(
      get_logger(),
      "[TIMING] plan_to_pose phase=request_finish success=true total_ms=%.1f message=%s",
      elapsedMs(request_start), response->message.c_str());
    return;
  }

  if (cancel_requested_.load()) {
    response->message = "Execution canceled before trajectory start";
    return;
  }
  phase_start = SteadyClock::now();
  if (!executeTrajectory(
      request->arm_name, response->trajectory, response->message))
  {
    RCLCPP_INFO(
      get_logger(),
      "[TIMING] plan_to_pose phase=execute_trajectory success=false elapsed_ms=%.1f "
      "total_ms=%.1f message=%s",
      elapsedMs(phase_start), elapsedMs(request_start), response->message.c_str());
    return;
  }
  RCLCPP_INFO(
    get_logger(),
    "[TIMING] plan_to_pose phase=execute_trajectory success=true elapsed_ms=%.1f",
    elapsedMs(phase_start));

  if (cancel_requested_.load()) {
    response->message = "Execution canceled after trajectory";
    return;
  }
  const bool release_grasp_ellipsoid_after_gripper =
    request->keep_grasp_ellipsoid &&
    request->gripper_command &&
    request->gripper_opening_m > gripper_contact_opening_threshold_;

  if (request->gripper_command) {
    std::string gripper_message;
    const double gripper_duration = release_grasp_ellipsoid_after_gripper ?
      gripper_release_duration_ : gripper_move_duration_;
    phase_start = SteadyClock::now();
    if (!executeGripperCommand(
        request->arm_name, request->gripper_opening_m, gripper_duration, gripper_message))
    {
      response->message =
        "Arm trajectory execution succeeded, but " + gripper_message;
      RCLCPP_INFO(
        get_logger(),
        "[TIMING] plan_to_pose phase=execute_gripper success=false elapsed_ms=%.1f "
        "total_ms=%.1f message=%s",
        elapsedMs(phase_start), elapsedMs(request_start), response->message.c_str());
      return;
    }
    RCLCPP_INFO(
      get_logger(),
      "[TIMING] plan_to_pose phase=execute_gripper success=true opening=%.4f "
      "duration_s=%.2f elapsed_ms=%.1f",
      request->gripper_opening_m, gripper_duration, elapsedMs(phase_start));
  }

  if (cancel_requested_.load()) {
    response->message = "Execution canceled after gripper command";
    return;
  }
  std::string ellipsoid_message;
  const bool final_grasp_ellipsoid_enabled =
    request->keep_grasp_ellipsoid && !release_grasp_ellipsoid_after_gripper;
  phase_start = SteadyClock::now();
  if (!setGraspEllipsoid(
      request->arm_name, final_grasp_ellipsoid_enabled, ellipsoid_message))
  {
    response->message = request->gripper_command ?
      "Arm trajectory and gripper execution succeeded, but " + ellipsoid_message :
      "Arm trajectory execution succeeded, but " + ellipsoid_message;
    RCLCPP_INFO(
      get_logger(),
      "[TIMING] plan_to_pose phase=update_grasp_ellipsoid success=false enabled=%s "
      "elapsed_ms=%.1f total_ms=%.1f message=%s",
      final_grasp_ellipsoid_enabled ? "true" : "false",
      elapsedMs(phase_start), elapsedMs(request_start), response->message.c_str());
    return;
  }
  RCLCPP_INFO(
    get_logger(),
    "[TIMING] plan_to_pose phase=update_grasp_ellipsoid success=true enabled=%s elapsed_ms=%.1f",
    final_grasp_ellipsoid_enabled ? "true" : "false", elapsedMs(phase_start));

  response->success = true;
  response->message = request->gripper_command ?
    "Planning, trajectory, gripper, and grasp ellipsoid update succeeded" :
    "Planning, trajectory, and grasp ellipsoid update succeeded";
  RCLCPP_INFO(
    get_logger(),
    "[TIMING] plan_to_pose phase=request_finish success=true total_ms=%.1f message=%s",
    elapsedMs(request_start), response->message.c_str());
}

void PathPlanningNode::handlePlanV2Request(
  const std::shared_ptr<PlanToPoseV2::Request> request,
  std::shared_ptr<PlanToPoseV2::Response> response)
{
  const auto request_start = SteadyClock::now();
  response->success = false;
  if (cancel_requested_.exchange(false)) {
    response->message = "Execution canceled before planning started";
    return;
  }
  ActiveRequestGuard active_request(request_active_, cancel_requested_);

  MoveGroupInterface * move_group = nullptr;
  std::string planning_group;
  std::string end_effector_link;
  if (request->arm_name == "left") {
    move_group = left_move_group_.get();
    planning_group = left_planning_group_;
    end_effector_link = left_end_effector_link_;
  } else if (request->arm_name == "right") {
    move_group = right_move_group_.get();
    planning_group = right_planning_group_;
    end_effector_link = right_end_effector_link_;
  } else {
    response->message = "Invalid arm name";
    return;
  }

  if (!static_scene_loader_->hasScene(request->scene_id)) {
    response->message = "Scene ID not found";
    return;
  }
  if (request->target_name.empty()) {
    response->message = "Target name is empty";
    return;
  }
  if (request->target_pose.header.frame_id.empty()) {
    response->message = "Target frame is empty";
    return;
  }

  const auto & pose = request->target_pose.pose;
  const std::array<double, 7> values = {
    pose.position.x, pose.position.y, pose.position.z,
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
  };
  if (!std::all_of(values.begin(), values.end(), [](const double value) {
      return std::isfinite(value);
    }))
  {
    response->message = "Target pose contains non-finite values";
    return;
  }

  const double quaternion_norm_squared =
    pose.orientation.x * pose.orientation.x +
    pose.orientation.y * pose.orientation.y +
    pose.orientation.z * pose.orientation.z +
    pose.orientation.w * pose.orientation.w;
  if (quaternion_norm_squared == 0.0) {
    response->message = "Target orientation is zero";
    return;
  }
  if (request->gripper_command &&
    (!std::isfinite(request->gripper_opening_m) ||
    request->gripper_opening_m < 0.0 ||
    request->gripper_opening_m > kMaxGripperOpening))
  {
    response->message = "Gripper opening must be finite and in [0.0, 0.1] meters";
    return;
  }

  RCLCPP_INFO(
    get_logger(),
    "[TIMING] plan_to_pose_v2 phase=request_start arm=%s scene=%u target=%s "
    "gripper=%s opening=%.4f keep_obj=%s remove_obj=%s",
    request->arm_name.c_str(), static_cast<unsigned int>(request->scene_id),
    request->target_name.c_str(), request->gripper_command ? "true" : "false",
    request->gripper_opening_m, request->keep_grasp_obj ? "true" : "false",
    request->remove_obj ? "true" : "false");

  std::string scene_message;
  if (!static_scene_loader_->loadScene(request->scene_id, scene_message)) {
    response->message = scene_message;
    return;
  }

  geometry_msgs::msg::PoseStamped target_pose;
  if (!transformTarget(request->target_pose, target_pose, response->message)) {
    return;
  }
  auto & orientation = target_pose.pose.orientation;
  const double orientation_norm = std::sqrt(
    orientation.x * orientation.x +
    orientation.y * orientation.y +
    orientation.z * orientation.z +
    orientation.w * orientation.w);
  orientation.x /= orientation_norm;
  orientation.y /= orientation_norm;
  orientation.z /= orientation_norm;
  orientation.w /= orientation_norm;

  bool keep_attached_object_unchanged = false;
  bool grasp_object_was_attached = false;
  if (grasp_object_type_ == GraspObjectType::Cylinder) {
    const auto & cylinder_input = request->grasp_object.cylinder_input;
    const bool cylinder_input_empty =
      cylinder_input.point_cloud.width == 0U ||
      cylinder_input.point_cloud.height == 0U;
    const std::string object_id = graspObjectId(request->arm_name);
    const bool object_is_attached =
      planning_scene_->getAttachedObjects({object_id}).count(object_id) != 0U;
    grasp_object_was_attached = object_is_attached;

    if (request->remove_obj) {
      cylinderCache(request->arm_name).reset();
    } else if (!cylinder_input_empty && !updateCylinderCache(
        request->arm_name, cylinder_input, target_pose, response->message))
    {
      return;
    }

    keep_attached_object_unchanged =
      cylinder_input_empty && request->keep_grasp_obj && object_is_attached;
    if (request->keep_grasp_obj && !cylinderCache(request->arm_name) &&
      !object_is_attached)
    {
      response->message = "Grasp object cache is empty";
      return;
    }
  } else if (grasp_object_type_ == GraspObjectType::Mesh) {
    const auto & mesh_input = request->grasp_object.mesh_input;
    const bool mesh_input_empty = isMeshInputEmpty(mesh_input);
    if (!mesh_input_empty && !hasFiniteMeshInput(mesh_input)) {
      response->message = "Mesh input contains non-finite values";
      return;
    }
    const std::string object_id = graspObjectId(request->arm_name);
    const bool object_is_attached =
      planning_scene_->getAttachedObjects({object_id}).count(object_id) != 0U;
    grasp_object_was_attached = object_is_attached;

    if (request->remove_obj) {
      deleteMeshVisual(request->arm_name);
      meshCache(request->arm_name).reset();
    } else if (!mesh_input_empty && !updateMeshCache(
        request->arm_name, request->target_name, mesh_input, target_pose,
        response->message))
    {
      return;
    }

    keep_attached_object_unchanged =
      mesh_input_empty && request->keep_grasp_obj && object_is_attached;
    if (request->keep_grasp_obj && !meshCache(request->arm_name) &&
      !object_is_attached)
    {
      response->message = "Grasp object cache is empty";
      return;
    }
  }

  const auto current_state = move_group->getCurrentState(1.0);
  if (!current_state) {
    response->message = "Current robot state unavailable";
    return;
  }

  if (!validateTargetPoseV2(
      *move_group, *current_state, planning_group, end_effector_link,
      target_pose, request->keep_grasp_obj, response->message))
  {
    return;
  }

  move_group->clearPoseTargets();
  move_group->setStartStateToCurrentState();
  if (!move_group->setPoseTarget(target_pose, end_effector_link)) {
    response->message = "Failed to set pose target";
    return;
  }

  MoveGroupInterface::Plan plan;
  const auto result = move_group->plan(plan);
  move_group->clearPoseTargets();

  if (cancel_requested_.load()) {
    response->message = "Execution canceled during planning";
    return;
  }
  if (result != moveit::core::MoveItErrorCode::SUCCESS) {
    response->message = "Planning failed";
    return;
  }
  if (plan.trajectory.joint_trajectory.points.empty()) {
    response->message = "Planned trajectory is empty";
    return;
  }

  response->trajectory = plan.trajectory;
  if (!execute_trajectory_) {
    response->success = true;
    response->message =
      "Planning succeeded; grasp object cache updated, attachment unchanged in virtual mode";
    RCLCPP_INFO(
      get_logger(),
      "[TIMING] plan_to_pose_v2 phase=request_finish success=true total_ms=%.1f",
      elapsedMs(request_start));
    return;
  }

  if (cancel_requested_.load()) {
    response->message = "Execution canceled before trajectory start";
    return;
  }
  if (!executeTrajectory(
      request->arm_name, response->trajectory, response->message))
  {
    return;
  }

  if (cancel_requested_.load()) {
    response->message = "Execution canceled after trajectory";
    return;
  }
  const bool release_grasp_object_after_gripper =
    request->gripper_command && !request->keep_grasp_obj &&
    request->gripper_opening_m > gripper_contact_opening_threshold_ &&
    grasp_object_was_attached;
  if (request->gripper_command) {
    const double move_duration = release_grasp_object_after_gripper ?
      gripper_release_duration_ : gripper_move_duration_;
    std::string gripper_message;
    if (!executeGripperCommand(
        request->arm_name, request->gripper_opening_m,
        move_duration, gripper_message))
    {
      response->message =
        "Arm trajectory execution succeeded, but " + gripper_message;
      return;
    }
  }

  if (cancel_requested_.load()) {
    response->message = "Execution canceled after gripper command";
    return;
  }
  if (!keep_attached_object_unchanged) {
    std::string grasp_object_message;
    // Once the fingers have opened, remove the released object from MoveIt's
    // collision scene.  The deterministic local -Z retreat still checks the
    // robot against the shelf and all other static scene geometry.
    const bool object_update_succeeded = setGraspObject(
      request->arm_name, request->keep_grasp_obj, grasp_object_message);
    if (!object_update_succeeded)
    {
      response->message = request->gripper_command ?
        "Arm trajectory and gripper execution succeeded, but " + grasp_object_message :
        "Arm trajectory execution succeeded, but " + grasp_object_message;
      return;
    }
  }

  response->success = true;
  response->message = request->gripper_command ?
    "Planning, trajectory, gripper, and grasp object update succeeded" :
    "Planning, trajectory, and grasp object update succeeded";
  RCLCPP_INFO(
    get_logger(),
    "[TIMING] plan_to_pose_v2 phase=request_finish success=true total_ms=%.1f",
    elapsedMs(request_start));
}

void PathPlanningNode::handlePlanV3Request(
  const std::shared_ptr<PlanToPoseV3::Request> request,
  std::shared_ptr<PlanToPoseV3::Response> response)
{
  const auto request_start = SteadyClock::now();
  response->success = false;
  if (cancel_requested_.exchange(false)) {
    response->message = "Execution canceled before planning started";
    return;
  }
  ActiveRequestGuard active_request(request_active_, cancel_requested_);

  MoveGroupInterface * move_group = nullptr;
  std::string planning_group;
  std::string end_effector_link;
  if (request->arm_name == "left") {
    move_group = left_move_group_.get();
    planning_group = left_planning_group_;
    end_effector_link = left_end_effector_link_;
  } else if (request->arm_name == "right") {
    move_group = right_move_group_.get();
    planning_group = right_planning_group_;
    end_effector_link = right_end_effector_link_;
  } else {
    response->message = "Invalid arm name";
    return;
  }

  const std::string optimizer_mode =
    request->trajectory_optimizer_mode.empty() ?
    "none" : request->trajectory_optimizer_mode;
  if (optimizer_mode != "none" && optimizer_mode != "chomp" &&
    optimizer_mode != "stomp")
  {
    response->message = "Invalid trajectory optimizer mode";
    return;
  }

  std::optional<OmplWaypointConfig> waypoint_config;
  if (!request->ompl_waypoint_config.empty()) {
    OmplWaypointConfig loaded_config;
    if (!ompl_waypoint_loader_->load(
        request->ompl_waypoint_config, loaded_config, response->message))
    {
      return;
    }
    if (loaded_config.arm_name != request->arm_name) {
      response->message = "OMPL waypoint config arm does not match request arm";
      return;
    }
    waypoint_config = std::move(loaded_config);
  }

  std::string planning_pipeline_id =
    (waypoint_config ? "ompl_waypoint_" : "ompl_") + optimizer_mode;
  if (request->use_preference_optimizer) {
    planning_pipeline_id += "_preference";
  }
  move_group->setPlanningPipelineId(planning_pipeline_id);
  PlanningPipelineGuard pipeline_guard(*move_group);

  if (!static_scene_loader_->hasScene(request->scene_id)) {
    response->message = "Scene ID not found";
    return;
  }
  if (request->target_name.empty()) {
    response->message = "Target name is empty";
    return;
  }
  if (request->target_pose.header.frame_id.empty()) {
    response->message = "Target frame is empty";
    return;
  }

  const auto & pose = request->target_pose.pose;
  const std::array<double, 7> values = {
    pose.position.x, pose.position.y, pose.position.z,
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
  };
  if (!std::all_of(values.begin(), values.end(), [](const double value) {
      return std::isfinite(value);
    }))
  {
    response->message = "Target pose contains non-finite values";
    return;
  }

  const double quaternion_norm_squared =
    pose.orientation.x * pose.orientation.x +
    pose.orientation.y * pose.orientation.y +
    pose.orientation.z * pose.orientation.z +
    pose.orientation.w * pose.orientation.w;
  if (quaternion_norm_squared == 0.0) {
    response->message = "Target orientation is zero";
    return;
  }
  if (request->gripper_command &&
    (!std::isfinite(request->gripper_opening_m) ||
    request->gripper_opening_m < 0.0 ||
    request->gripper_opening_m > kMaxGripperOpening))
  {
    response->message = "Gripper opening must be finite and in [0.0, 0.1] meters";
    return;
  }
  if (request->adjust_grasp_mesh_min_z &&
    (!std::isfinite(request->grasp_mesh_min_z_m) ||
    !std::isfinite(request->max_grasp_mesh_z_adjustment_m) ||
    request->max_grasp_mesh_z_adjustment_m < 0.0 ||
    request->max_grasp_mesh_z_adjustment_m > 0.20))
  {
    response->message =
      "Mesh minimum-Z adjustment requires a finite minimum Z and a finite "
      "maximum adjustment in [0.0, 0.20] meters";
    return;
  }
  if (request->allow_grasp_mesh_z_lowering &&
    !request->adjust_grasp_mesh_min_z)
  {
    response->message =
      "Mesh Z lowering requires mesh minimum-Z adjustment to be enabled";
    return;
  }

  RCLCPP_INFO(
    get_logger(),
    "[TIMING] plan_to_pose_v3 phase=request_start arm=%s scene=%u target=%s "
    "gripper=%s opening=%.4f keep_obj=%s remove_obj=%s adjust_mesh_min_z=%s "
    "allow_mesh_z_lowering=%s",
    request->arm_name.c_str(), static_cast<unsigned int>(request->scene_id),
    request->target_name.c_str(), request->gripper_command ? "true" : "false",
    request->gripper_opening_m, request->keep_grasp_obj ? "true" : "false",
    request->remove_obj ? "true" : "false",
    request->adjust_grasp_mesh_min_z ? "true" : "false",
    request->allow_grasp_mesh_z_lowering ? "true" : "false");

  std::string scene_message;
  if (!static_scene_loader_->loadScene(request->scene_id, scene_message)) {
    response->message = scene_message;
    return;
  }

  geometry_msgs::msg::PoseStamped target_pose;
  if (!transformTarget(request->target_pose, target_pose, response->message)) {
    return;
  }
  auto & orientation = target_pose.pose.orientation;
  const double orientation_norm = std::sqrt(
    orientation.x * orientation.x +
    orientation.y * orientation.y +
    orientation.z * orientation.z +
    orientation.w * orientation.w);
  orientation.x /= orientation_norm;
  orientation.y /= orientation_norm;
  orientation.z /= orientation_norm;
  orientation.w /= orientation_norm;

  bool keep_attached_object_unchanged = false;
  bool grasp_object_was_attached = false;
  if (grasp_object_type_ == GraspObjectType::Cylinder) {
    const auto & cylinder_input = request->grasp_object.cylinder_input;
    const bool cylinder_input_empty =
      cylinder_input.point_cloud.width == 0U ||
      cylinder_input.point_cloud.height == 0U;
    const std::string object_id = graspObjectId(request->arm_name);
    const bool object_is_attached =
      planning_scene_->getAttachedObjects({object_id}).count(object_id) != 0U;
    grasp_object_was_attached = object_is_attached;

    if (request->remove_obj) {
      cylinderCache(request->arm_name).reset();
    } else if (!cylinder_input_empty && !updateCylinderCache(
        request->arm_name, cylinder_input, target_pose, response->message))
    {
      return;
    }

    keep_attached_object_unchanged =
      cylinder_input_empty && request->keep_grasp_obj && object_is_attached;
    if (request->keep_grasp_obj && !cylinderCache(request->arm_name) &&
      !object_is_attached)
    {
      response->message = "Grasp object cache is empty";
      return;
    }
  } else if (grasp_object_type_ == GraspObjectType::Mesh) {
    const auto & mesh_input = request->grasp_object.mesh_input;
    const bool mesh_input_empty = isMeshInputEmpty(mesh_input);
    if (!mesh_input_empty && !hasFiniteMeshInput(mesh_input)) {
      response->message = "Mesh input contains non-finite values";
      return;
    }
    const std::string object_id = graspObjectId(request->arm_name);
    const bool object_is_attached =
      planning_scene_->getAttachedObjects({object_id}).count(object_id) != 0U;
    grasp_object_was_attached = object_is_attached;

    if (request->remove_obj) {
      deleteMeshVisual(request->arm_name);
      meshCache(request->arm_name).reset();
    } else if (!mesh_input_empty && !updateMeshCache(
        request->arm_name, request->target_name, mesh_input, target_pose,
        response->message))
    {
      return;
    }

    keep_attached_object_unchanged =
      mesh_input_empty && request->keep_grasp_obj && object_is_attached;
    if (request->keep_grasp_obj && !meshCache(request->arm_name) &&
      !object_is_attached)
    {
      response->message = "Grasp object cache is empty";
      return;
    }
  }

  std::string mesh_min_z_summary;
  if (request->adjust_grasp_mesh_min_z) {
    if (grasp_object_type_ != GraspObjectType::Mesh) {
      response->message =
        "Mesh minimum-Z adjustment requires grasp_object.type=mesh";
      return;
    }
    if (!request->keep_grasp_obj) {
      response->message =
        "Mesh minimum-Z adjustment requires keep_grasp_obj=true";
      return;
    }
    const auto & cached = meshCache(request->arm_name);
    if (!cached) {
      response->message =
        "Mesh minimum-Z adjustment requires a cached grasp mesh";
      return;
    }

    double predicted_minimum_z = 0.0;
    if (!attachedMeshMinimumWorldZ(
        cached->mesh,
        poseToEigen(target_pose.pose),
        poseToEigen(cached->pose_in_link6),
        predicted_minimum_z,
        response->message))
    {
      return;
    }
    const double signed_surface_alignment =
      request->grasp_mesh_min_z_m - predicted_minimum_z;
    const double z_adjustment = request->allow_grasp_mesh_z_lowering ?
      signed_surface_alignment : std::max(0.0, signed_surface_alignment);
    if (std::abs(z_adjustment) >
      request->max_grasp_mesh_z_adjustment_m + 1.0e-9)
    {
      response->message =
        "Required attached mesh Z adjustment " + std::to_string(z_adjustment) +
        " m has magnitude exceeding configured maximum " +
        std::to_string(request->max_grasp_mesh_z_adjustment_m) +
        " m (predicted minimum Z " + std::to_string(predicted_minimum_z) +
        " m, required minimum Z " +
        std::to_string(request->grasp_mesh_min_z_m) + " m)";
      return;
    }

    target_pose.pose.position.z += z_adjustment;
    mesh_min_z_summary =
      "attached mesh minimum Z " + std::to_string(predicted_minimum_z) +
      " m, required " + std::to_string(request->grasp_mesh_min_z_m) +
      " m, target world-Z adjustment +" + std::to_string(z_adjustment) + " m";
    RCLCPP_INFO(
      get_logger(),
      "Adjusted target from attached mesh lowest point: arm=%s "
      "mesh_min_z=%.6f required_min_z=%.6f adjustment=%+.6f target_z=%.6f "
      "allow_lowering=%s",
      request->arm_name.c_str(), predicted_minimum_z,
      request->grasp_mesh_min_z_m, z_adjustment,
      target_pose.pose.position.z,
      request->allow_grasp_mesh_z_lowering ? "true" : "false");
  }

  const auto current_state = move_group->getCurrentState(1.0);
  if (!current_state) {
    response->message = "Current robot state unavailable";
    return;
  }

  if (!validateTargetPoseV2(
      *move_group, *current_state, planning_group, end_effector_link,
      target_pose, request->keep_grasp_obj, response->message))
  {
    return;
  }

  move_group->clearPoseTargets();
  move_group->setStartStateToCurrentState();
  if (!move_group->setPoseTarget(target_pose, end_effector_link)) {
    response->message = "Failed to set pose target";
    return;
  }
  if (request->use_tcp_position_constraint) {
    move_group->setPathConstraints(
      makeLink6PositionConstraint(end_effector_link));
  }
  if (waypoint_config) {
    move_group->setTrajectoryConstraints(
      makeWaypointConstraints(*waypoint_config, end_effector_link));
  }

  MoveGroupInterface::Plan plan;
  const auto result = move_group->plan(plan);
  if (waypoint_config) {
    move_group->clearTrajectoryConstraints();
  }
  if (request->use_tcp_position_constraint) {
    move_group->clearPathConstraints();
  }
  move_group->clearPoseTargets();

  if (cancel_requested_.load()) {
    response->message = "Execution canceled during planning";
    return;
  }
  if (result != moveit::core::MoveItErrorCode::SUCCESS) {
    response->message = "Planning failed";
    return;
  }
  if (plan.trajectory.joint_trajectory.points.empty()) {
    response->message = "Planned trajectory is empty";
    return;
  }

  response->trajectory = plan.trajectory;
  if (!execute_trajectory_) {
    response->success = true;
    response->message =
      "Planning succeeded; grasp object cache updated, attachment unchanged in virtual mode";
    if (!mesh_min_z_summary.empty()) {
      response->message += "; " + mesh_min_z_summary;
    }
    RCLCPP_INFO(
      get_logger(),
      "[TIMING] plan_to_pose_v3 phase=request_finish success=true total_ms=%.1f",
      elapsedMs(request_start));
    return;
  }

  if (cancel_requested_.load()) {
    response->message = "Execution canceled before trajectory start";
    return;
  }
  if (!executeTrajectory(
      request->arm_name, response->trajectory, response->message))
  {
    return;
  }

  if (cancel_requested_.load()) {
    response->message = "Execution canceled after trajectory";
    return;
  }
  const bool release_grasp_object_after_gripper =
    request->gripper_command && !request->keep_grasp_obj &&
    request->gripper_opening_m > gripper_contact_opening_threshold_ &&
    grasp_object_was_attached;
  if (request->gripper_command) {
    const double move_duration = release_grasp_object_after_gripper ?
      gripper_release_duration_ : gripper_move_duration_;
    std::string gripper_message;
    if (!executeGripperCommand(
        request->arm_name, request->gripper_opening_m,
        move_duration, gripper_message))
    {
      response->message =
        "Arm trajectory execution succeeded, but " + gripper_message;
      return;
    }
  }

  if (cancel_requested_.load()) {
    response->message = "Execution canceled after gripper command";
    return;
  }
  if (!keep_attached_object_unchanged) {
    std::string grasp_object_message;
    // Once the fingers have opened, remove the released object from MoveIt's
    // collision scene.  The deterministic local -Z retreat still checks the
    // robot against the shelf and all other static scene geometry.
    const bool object_update_succeeded = setGraspObject(
      request->arm_name, request->keep_grasp_obj, grasp_object_message);
    if (!object_update_succeeded)
    {
      response->message = request->gripper_command ?
        "Arm trajectory and gripper execution succeeded, but " + grasp_object_message :
        "Arm trajectory execution succeeded, but " + grasp_object_message;
      return;
    }
  }

  response->success = true;
  response->message = request->gripper_command ?
    "Planning, trajectory, gripper, and grasp object update succeeded" :
    "Planning, trajectory, and grasp object update succeeded";
  if (!mesh_min_z_summary.empty()) {
    response->message += "; " + mesh_min_z_summary;
  }
  RCLCPP_INFO(
    get_logger(),
    "[TIMING] plan_to_pose_v3 phase=request_finish success=true total_ms=%.1f",
    elapsedMs(request_start));
}

void PathPlanningNode::handleCartesianPlanRequest(
  const std::shared_ptr<PlanCartesianPath::Request> request,
  std::shared_ptr<PlanCartesianPath::Response> response)
{
  const auto request_start = SteadyClock::now();
  response->success = false;
  response->fraction = 0.0;
  if (cancel_requested_.exchange(false)) {
    response->message = "Execution canceled before Cartesian planning started";
    return;
  }
  ActiveRequestGuard active_request(request_active_, cancel_requested_);

  MoveGroupInterface * move_group = nullptr;
  std::string planning_group;
  if (request->arm_name == "left") {
    move_group = left_move_group_.get();
    planning_group = left_planning_group_;
  } else if (request->arm_name == "right") {
    move_group = right_move_group_.get();
    planning_group = right_planning_group_;
  } else {
    response->message = "Invalid arm name";
    return;
  }

  if (!static_scene_loader_->hasScene(request->scene_id)) {
    response->message = "Scene ID not found";
    return;
  }
  if (!std::isfinite(request->eef_step_m) ||
    request->eef_step_m <= 0.0 || request->eef_step_m > 0.05)
  {
    response->message = "Cartesian eef_step_m must be finite and in (0.0, 0.05]";
    return;
  }
  if (!std::isfinite(request->max_velocity_scaling_factor) ||
    request->max_velocity_scaling_factor <= 0.0 ||
    request->max_velocity_scaling_factor > 1.0 ||
    !std::isfinite(request->max_acceleration_scaling_factor) ||
    request->max_acceleration_scaling_factor <= 0.0 ||
    request->max_acceleration_scaling_factor > 1.0)
  {
    response->message =
      "Cartesian velocity and acceleration scaling factors must be finite and in (0.0, 1.0]";
    return;
  }

  geometry_msgs::msg::PoseStamped target_pose;
  if (!transformTarget(request->target_pose, target_pose, response->message)) {
    return;
  }
  const auto & position = target_pose.pose.position;
  auto & orientation = target_pose.pose.orientation;
  const std::array<double, 7> pose_values = {
    position.x, position.y, position.z,
    orientation.x, orientation.y, orientation.z, orientation.w
  };
  if (!std::all_of(
      pose_values.begin(), pose_values.end(),
      [](const double value) {return std::isfinite(value);}))
  {
    response->message = "Cartesian target pose contains non-finite values";
    return;
  }
  const double orientation_norm = std::sqrt(
    orientation.x * orientation.x + orientation.y * orientation.y +
    orientation.z * orientation.z + orientation.w * orientation.w);
  if (orientation_norm <= 1.0e-9) {
    response->message = "Cartesian target orientation is zero";
    return;
  }
  orientation.x /= orientation_norm;
  orientation.y /= orientation_norm;
  orientation.z /= orientation_norm;
  orientation.w /= orientation_norm;

  RCLCPP_INFO(
    get_logger(),
    "[TIMING] plan_cartesian_path phase=request_start arm=%s scene=%u "
    "eef_step=%.4f velocity_scale=%.3f acceleration_scale=%.3f "
    "target=(%.6f,%.6f,%.6f)",
    request->arm_name.c_str(), static_cast<unsigned int>(request->scene_id),
    request->eef_step_m, request->max_velocity_scaling_factor,
    request->max_acceleration_scaling_factor,
    position.x, position.y, position.z);

  std::string scene_message;
  if (!static_scene_loader_->loadScene(request->scene_id, scene_message)) {
    response->message = scene_message;
    return;
  }

  move_group->clearPoseTargets();
  move_group->setStartStateToCurrentState();
  moveit_msgs::msg::MoveItErrorCodes error_code;
  try {
    response->fraction = move_group->computeCartesianPath(
      {target_pose.pose}, request->eef_step_m,
      response->trajectory, true, &error_code);
  } catch (const std::exception & error) {
    response->message = std::string("Cartesian planning failed: ") + error.what();
    return;
  }

  constexpr double kCompleteCartesianFraction = 1.0 - 1.0e-6;
  if (!std::isfinite(response->fraction) ||
    response->fraction < kCompleteCartesianFraction)
  {
    response->message =
      "Cartesian path incomplete: fraction=" + std::to_string(response->fraction) +
      ", MoveIt error=" + std::to_string(error_code.val);
    RCLCPP_WARN(
      get_logger(),
      "[TIMING] plan_cartesian_path phase=planning success=false fraction=%.6f "
      "elapsed_ms=%.1f message=%s",
      response->fraction, elapsedMs(request_start), response->message.c_str());
    return;
  }
  if (response->trajectory.joint_trajectory.points.empty()) {
    response->message = "Cartesian trajectory is empty";
    return;
  }

  const auto current_state = move_group->getCurrentState(1.0);
  if (!current_state) {
    response->message = "Current robot state unavailable for Cartesian timing";
    return;
  }
  robot_trajectory::RobotTrajectory timed_trajectory(
    move_group->getRobotModel(), planning_group);
  timed_trajectory.setRobotTrajectoryMsg(*current_state, response->trajectory);
  const trajectory_processing::TimeOptimalTrajectoryGeneration time_parameterization(
    0.01, 0.02, 0.001);
  if (!time_parameterization.computeTimeStamps(
      timed_trajectory,
      request->max_velocity_scaling_factor,
      request->max_acceleration_scaling_factor))
  {
    response->message = "Failed to time-parameterize complete Cartesian path";
    return;
  }
  timed_trajectory.getRobotTrajectoryMsg(response->trajectory);

  RCLCPP_INFO(
    get_logger(),
    "[TIMING] plan_cartesian_path phase=planning success=true fraction=%.6f "
    "points=%zu elapsed_ms=%.1f",
    response->fraction, response->trajectory.joint_trajectory.points.size(),
    elapsedMs(request_start));

  if (!execute_trajectory_) {
    response->success = true;
    response->message = "Complete Cartesian planning succeeded; execution disabled";
    return;
  }
  if (cancel_requested_.load()) {
    response->message = "Execution canceled before Cartesian trajectory start";
    return;
  }
  if (!executeTrajectory(
      request->arm_name, response->trajectory, response->message))
  {
    return;
  }

  response->success = true;
  response->message = "Complete Cartesian path and trajectory execution succeeded";
  RCLCPP_INFO(
    get_logger(),
    "[TIMING] plan_cartesian_path phase=request_finish success=true fraction=%.6f "
    "total_ms=%.1f",
    response->fraction, elapsedMs(request_start));
}

void PathPlanningNode::handleJointPlanRequest(
  const std::shared_ptr<PlanToJoints::Request> request,
  std::shared_ptr<PlanToJoints::Response> response)
{
  const auto request_start = SteadyClock::now();
  response->success = false;
  if (cancel_requested_.exchange(false)) {
    response->message = "Execution canceled before planning started";
    return;
  }
  ActiveRequestGuard active_request(request_active_, cancel_requested_);

  MoveGroupInterface * move_group = nullptr;
  std::string planning_group;
  if (request->arm_name == "left") {
    move_group = left_move_group_.get();
    planning_group = left_planning_group_;
  } else if (request->arm_name == "right") {
    move_group = right_move_group_.get();
    planning_group = right_planning_group_;
  } else {
    response->message = "Invalid arm name";
    return;
  }

  if (!static_scene_loader_->hasScene(request->scene_id)) {
    response->message = "Scene ID not found";
    return;
  }
  if (!std::all_of(
      request->joint_positions.begin(), request->joint_positions.end(),
      [](const double value) {return std::isfinite(value);}))
  {
    response->message = "Joint target contains non-finite values";
    return;
  }
  const std::vector<double> target_joints(
    request->joint_positions.begin(), request->joint_positions.end());
  if (request->gripper_command &&
    (!std::isfinite(request->gripper_opening_m) ||
    request->gripper_opening_m < 0.0 ||
    request->gripper_opening_m > kMaxGripperOpening))
  {
    response->message = "Gripper opening must be finite and in [0.0, 0.1] meters";
    return;
  }

  RCLCPP_INFO(
    get_logger(),
    "[TIMING] plan_to_joints phase=request_start arm=%s scene=%u "
    "gripper=%s opening=%.4f keep_obj=%s "
    "joints=[%.9f,%.9f,%.9f,%.9f,%.9f,%.9f]",
    request->arm_name.c_str(), static_cast<unsigned int>(request->scene_id),
    request->gripper_command ? "true" : "false", request->gripper_opening_m,
    request->keep_grasp_ellipsoid ? "true" : "false",
    target_joints[0], target_joints[1], target_joints[2],
    target_joints[3], target_joints[4], target_joints[5]);

  std::string scene_message;
  auto phase_start = SteadyClock::now();
  if (!static_scene_loader_->loadScene(request->scene_id, scene_message)) {
    response->message = scene_message;
    return;
  }
  RCLCPP_INFO(
    get_logger(),
    "[TIMING] plan_to_joints phase=load_scene success=true elapsed_ms=%.1f",
    elapsedMs(phase_start));

  if (!request->keep_grasp_ellipsoid) {
    std::string grasp_object_message;
    phase_start = SteadyClock::now();
    if (!setGraspObject(request->arm_name, false, grasp_object_message)) {
      response->message =
        "Failed to clear grasp object before planning: " + grasp_object_message;
      return;
    }
    RCLCPP_INFO(
      get_logger(),
      "[TIMING] plan_to_joints phase=clear_grasp_object_before_plan "
      "success=true elapsed_ms=%.1f",
      elapsedMs(phase_start));
  }

  phase_start = SteadyClock::now();
  auto current_state = move_group->getCurrentState(1.0);
  if (!current_state) {
    response->message = "Current robot state unavailable";
    return;
  }
  const auto * joint_group = current_state->getJointModelGroup(planning_group);
  if (!joint_group || joint_group->getVariableCount() != target_joints.size()) {
    response->message = "Planning group does not contain exactly six joints";
    return;
  }
  std::vector<double> current_joints;
  current_state->copyJointGroupPositions(joint_group, current_joints);
  double max_joint_error = 0.0;
  for (std::size_t index = 0; index < target_joints.size(); ++index) {
    max_joint_error = std::max(
      max_joint_error, std::abs(target_joints[index] - current_joints[index]));
  }
  // A placement release must never generate a second arm trajectory.  The
  // caller captured these six joints only to define the hold position; the
  // trajectory bridge snapshots the measured arm state again when the gripper
  // goal starts and holds it for the entire opening action.
  const bool locked_gripper_release =
    request->keep_grasp_ellipsoid && request->gripper_command &&
    request->gripper_opening_m > gripper_contact_opening_threshold_;
  const bool arm_motion_required =
    !locked_gripper_release && max_joint_error > kJointTargetTolerance;

  moveit::core::RobotState target_state(*current_state);
  target_state.setJointGroupPositions(joint_group, target_joints);
  target_state.update();
  if (!target_state.satisfiesBounds(joint_group)) {
    response->message = "Joint target violates model bounds";
    return;
  }

  planning_scene::PlanningScene collision_scene(move_group->getRobotModel());
  collision_scene.getTransformsNonConst().setTransform(
    Eigen::Isometry3d::Identity(), world_frame_);
  collision_scene.setCurrentState(*current_state);
  for (const auto & [id, object] : planning_scene_->getObjects()) {
    (void)id;
    collision_scene.processCollisionObjectMsg(object);
  }
  for (const auto & [id, object] : planning_scene_->getAttachedObjects()) {
    (void)id;
    collision_scene.processAttachedCollisionObjectMsg(object);
  }
  if (collision_scene.isStateColliding(target_state, planning_group)) {
    response->message = "Joint target is in collision";
    return;
  }
  RCLCPP_INFO(
    get_logger(),
    "[TIMING] plan_to_joints phase=validate_target success=true elapsed_ms=%.1f",
    elapsedMs(phase_start));

  const auto finish_execution = [&](const std::string & arm_success_message) -> bool {
      if (cancel_requested_.load()) {
        response->message = "Execution canceled after arm motion";
        return false;
      }
      const bool release_grasp_object_after_gripper = locked_gripper_release;

      if (request->gripper_command) {
        std::string gripper_message;
        const double gripper_duration = release_grasp_object_after_gripper ?
          gripper_release_duration_ : gripper_move_duration_;
        phase_start = SteadyClock::now();
        if (!executeGripperCommand(
            request->arm_name, request->gripper_opening_m, gripper_duration, gripper_message))
        {
          response->message = arm_success_message + ", but " + gripper_message;
          RCLCPP_INFO(
            get_logger(),
            "[TIMING] plan_to_joints phase=execute_gripper success=false "
            "elapsed_ms=%.1f total_ms=%.1f message=%s",
            elapsedMs(phase_start), elapsedMs(request_start), response->message.c_str());
          return false;
        }
        RCLCPP_INFO(
          get_logger(),
          "[TIMING] plan_to_joints phase=execute_gripper success=true opening=%.4f "
          "duration_s=%.2f elapsed_ms=%.1f",
          request->gripper_opening_m, gripper_duration, elapsedMs(phase_start));
      }

      if (cancel_requested_.load()) {
        response->message = "Execution canceled after gripper command";
        return false;
      }
      std::string grasp_object_message;
      const bool final_grasp_object_enabled =
        request->keep_grasp_ellipsoid && !release_grasp_object_after_gripper;
      phase_start = SteadyClock::now();
      // Do not leave a released bottle in the collision world: its proxy can
      // overlap the still-open fingers at the retreat start state.
      const bool object_update_succeeded = setGraspObject(
        request->arm_name, final_grasp_object_enabled, grasp_object_message);
      if (!object_update_succeeded)
      {
        response->message = arm_success_message +
          (request->gripper_command ? ", gripper execution succeeded, but " : ", but ") +
          grasp_object_message;
        RCLCPP_INFO(
          get_logger(),
          "[TIMING] plan_to_joints phase=update_grasp_object success=false enabled=%s "
          "elapsed_ms=%.1f total_ms=%.1f message=%s",
          final_grasp_object_enabled ? "true" : "false",
          elapsedMs(phase_start), elapsedMs(request_start), response->message.c_str());
        return false;
      }

      response->success = true;
      response->message = arm_success_message;
      if (request->gripper_command) {
        response->message += ", gripper execution succeeded";
      }
      response->message += ", grasp object update succeeded";
      RCLCPP_INFO(
        get_logger(),
        "[TIMING] plan_to_joints phase=request_finish success=true total_ms=%.1f "
        "message=%s",
        elapsedMs(request_start), response->message.c_str());
      return true;
    };

  if (!arm_motion_required) {
    RCLCPP_INFO(
      get_logger(),
      "[TIMING] plan_to_joints phase=arm_motion skipped=true locked_release=%s "
      "max_joint_error=%.9f tolerance=%.9f",
      locked_gripper_release ? "true" : "false",
      max_joint_error, kJointTargetTolerance);
    if (!execute_trajectory_) {
      response->success = true;
      response->message = locked_gripper_release ?
        "Arm motion skipped for locked gripper release" :
        "Joint target already reached";
      return;
    }
    (void)finish_execution(
      locked_gripper_release ?
      "Arm held at measured position for gripper release" :
      "Joint target already reached");
    return;
  }

  move_group->clearPoseTargets();
  move_group->setStartStateToCurrentState();
  if (!move_group->setJointValueTarget(target_joints)) {
    response->message = "Failed to set joint target";
    return;
  }

  MoveGroupInterface::Plan plan;
  phase_start = SteadyClock::now();
  const auto result = move_group->plan(plan);
  if (cancel_requested_.load()) {
    response->message = "Execution canceled during planning";
    return;
  }
  if (result != moveit::core::MoveItErrorCode::SUCCESS) {
    response->message = "Joint planning failed";
    RCLCPP_INFO(
      get_logger(),
      "[TIMING] plan_to_joints phase=planning success=false elapsed_ms=%.1f "
      "total_ms=%.1f",
      elapsedMs(phase_start), elapsedMs(request_start));
    return;
  }
  if (plan.trajectory.joint_trajectory.points.empty()) {
    response->message = "Planned joint trajectory is empty";
    return;
  }
  response->trajectory = plan.trajectory;
  RCLCPP_INFO(
    get_logger(),
    "[TIMING] plan_to_joints phase=planning success=true elapsed_ms=%.1f points=%zu",
    elapsedMs(phase_start), plan.trajectory.joint_trajectory.points.size());

  if (!execute_trajectory_) {
    response->success = true;
    response->message = "Joint planning succeeded";
    return;
  }
  if (cancel_requested_.load()) {
    response->message = "Execution canceled before trajectory start";
    return;
  }
  phase_start = SteadyClock::now();
  if (!executeTrajectory(
      request->arm_name, response->trajectory, response->message))
  {
    RCLCPP_INFO(
      get_logger(),
      "[TIMING] plan_to_joints phase=execute_trajectory success=false "
      "elapsed_ms=%.1f total_ms=%.1f message=%s",
      elapsedMs(phase_start), elapsedMs(request_start), response->message.c_str());
    return;
  }

  (void)finish_execution("Joint planning and trajectory execution succeeded");
}

bool PathPlanningNode::validateTargetPose(
  MoveGroupInterface & move_group,
  const moveit::core::RobotState & current_state,
  const std::string & planning_group,
  const std::string & end_effector_link,
  const geometry_msgs::msg::PoseStamped & target_pose,
  const bool keep_grasp_ellipsoid,
  std::string & message) const
{
  planning_scene::PlanningScene collision_scene(move_group.getRobotModel());
  collision_scene.getTransformsNonConst().setTransform(
    Eigen::Isometry3d::Identity(), world_frame_);
  collision_scene.setCurrentState(current_state);

  for (const auto & [id, object] : planning_scene_->getObjects()) {
    (void)id;
    collision_scene.processCollisionObjectMsg(object);
  }
  for (const auto & [id, object] : planning_scene_->getAttachedObjects()) {
    (void)id;
    collision_scene.processAttachedCollisionObjectMsg(object);
  }

  moveit::core::RobotState target_state(collision_scene.getCurrentState());
  const auto * joint_group = target_state.getJointModelGroup(planning_group);
  if (!target_state.setFromIK(
      joint_group, target_pose.pose, end_effector_link, kIkCheckTimeout))
  {
    message = "Target pose has no IK solution";
    return false;
  }

  const std::string ellipsoid_id =
    planning_group == left_planning_group_ ?
    "left_grasp_ellipsoid" : "right_grasp_ellipsoid";
  target_state.clearAttachedBody(ellipsoid_id);

  if (keep_grasp_ellipsoid) {
    shapes::ShapeConstPtr ellipsoid_shape(
      shapes::constructShapeFromMsg(grasp_ellipsoid_mesh_));
    Eigen::Isometry3d ellipsoid_pose = Eigen::Isometry3d::Identity();
    ellipsoid_pose.translation().z() = grasp_ellipsoid_center_z_;
    const std::vector<shapes::ShapeConstPtr> ellipsoid_shapes = {ellipsoid_shape};
    const EigenSTL::vector_Isometry3d ellipsoid_shape_poses = {
      Eigen::Isometry3d::Identity()};
    const std::vector<std::string> touch_links = {
      end_effector_link,
      planning_group == left_planning_group_ ? "left_gripper_base" : "right_gripper_base",
      planning_group == left_planning_group_ ? "left_link7" : "right_link7",
      planning_group == left_planning_group_ ? "left_link8" : "right_link8"
    };
    target_state.attachBody(
      ellipsoid_id,
      ellipsoid_pose,
      ellipsoid_shapes,
      ellipsoid_shape_poses,
      touch_links,
      end_effector_link);
  }
  target_state.update();

  if (collision_scene.isStateColliding(target_state, planning_group)) {
    message = "Target pose is in collision";
    return false;
  }

  return true;
}

bool PathPlanningNode::validateTargetPoseV2(
  MoveGroupInterface & move_group,
  const moveit::core::RobotState & current_state,
  const std::string & planning_group,
  const std::string & end_effector_link,
  const geometry_msgs::msg::PoseStamped & target_pose,
  const bool keep_grasp_obj,
  std::string & message) const
{
  planning_scene::PlanningScene collision_scene(move_group.getRobotModel());
  collision_scene.getTransformsNonConst().setTransform(
    Eigen::Isometry3d::Identity(), world_frame_);
  collision_scene.setCurrentState(current_state);

  for (const auto & [id, object] : planning_scene_->getObjects()) {
    (void)id;
    collision_scene.processCollisionObjectMsg(object);
  }
  for (const auto & [id, object] : planning_scene_->getAttachedObjects()) {
    (void)id;
    collision_scene.processAttachedCollisionObjectMsg(object);
  }

  moveit::core::RobotState target_state(collision_scene.getCurrentState());
  const auto * joint_group = target_state.getJointModelGroup(planning_group);
  if (!joint_group || !target_state.setFromIK(
      joint_group, target_pose.pose, end_effector_link, kIkCheckTimeout))
  {
    message = "Target pose has no IK solution";
    return false;
  }

  const bool is_left = planning_group == left_planning_group_;
  const std::string arm_name = is_left ? "left" : "right";
  const std::string object_id = graspObjectId(arm_name);

  if (keep_grasp_obj) {
    shapes::ShapeConstPtr grasp_shape;
    Eigen::Isometry3d grasp_pose = Eigen::Isometry3d::Identity();
    if (grasp_object_type_ == GraspObjectType::Ellipsoid) {
      grasp_shape.reset(shapes::constructShapeFromMsg(grasp_ellipsoid_mesh_));
      grasp_pose.translation().z() = grasp_ellipsoid_center_z_;
    } else if (
      grasp_object_type_ == GraspObjectType::Cylinder && cylinderCache(arm_name))
    {
      const auto & cylinder = *cylinderCache(arm_name);
      grasp_shape.reset(shapes::constructShapeFromMsg(cylinder.primitive));
      grasp_pose = poseToEigen(cylinder.pose_in_link6);
    } else if (grasp_object_type_ == GraspObjectType::Mesh && meshCache(arm_name)) {
      const auto & mesh = *meshCache(arm_name);
      grasp_shape.reset(shapes::constructShapeFromMsg(mesh.mesh));
      grasp_pose = poseToEigen(mesh.pose_in_link6);
    } else if (target_state.hasAttachedBody(object_id)) {
      target_state.update();
      if (collision_scene.isStateColliding(target_state, planning_group)) {
        message = "Target pose is in collision";
        return false;
      }
      return true;
    } else {
      message = "Grasp object cache is empty";
      return false;
    }

    if (!grasp_shape) {
      message = "Failed to construct grasp collision shape";
      return false;
    }
    target_state.clearAttachedBody(object_id);
    const std::vector<shapes::ShapeConstPtr> grasp_shapes = {grasp_shape};
    const EigenSTL::vector_Isometry3d grasp_shape_poses = {
      Eigen::Isometry3d::Identity()};
    const std::vector<std::string> touch_links = {
      end_effector_link,
      is_left ? "left_gripper_base" : "right_gripper_base",
      is_left ? "left_link7" : "right_link7",
      is_left ? "left_link8" : "right_link8"
    };
    target_state.attachBody(
      object_id,
      grasp_pose,
      grasp_shapes,
      grasp_shape_poses,
      touch_links,
      end_effector_link);
  }
  target_state.update();

  if (collision_scene.isStateColliding(target_state, planning_group)) {
    message = "Target pose is in collision";
    return false;
  }
  return true;
}

bool PathPlanningNode::updateCylinderCache(
  const std::string & arm_name,
  const path_planning_interfaces::msg::PointCloudWithGraspPoint & input,
  const geometry_msgs::msg::PoseStamped & target_pose,
  std::string & message)
{
  const auto & cloud = input.point_cloud;
  if (cloud.header.frame_id.empty()) {
    message = "Point cloud frame is empty";
    return false;
  }

  const auto * field_x = findPointField(cloud, "x");
  const auto * field_y = findPointField(cloud, "y");
  const auto * field_z = findPointField(cloud, "z");
  if (!field_x || !field_y || !field_z ||
    field_x->datatype != sensor_msgs::msg::PointField::FLOAT32 ||
    field_y->datatype != sensor_msgs::msg::PointField::FLOAT32 ||
    field_z->datatype != sensor_msgs::msg::PointField::FLOAT32)
  {
    message = "Point cloud must contain FLOAT32 x, y, and z fields";
    return false;
  }

  const std::uint32_t largest_offset = std::max(
    {field_x->offset, field_y->offset, field_z->offset});
  const std::size_t required_size = cloud.height == 0U ? 0U :
    static_cast<std::size_t>(cloud.height - 1U) * cloud.row_step +
    static_cast<std::size_t>(cloud.width) * cloud.point_step;
  if (cloud.width == 0U || cloud.height == 0U || cloud.point_step == 0U ||
    largest_offset + sizeof(float) > cloud.point_step ||
    cloud.row_step < cloud.width * cloud.point_step ||
    cloud.data.size() < required_size)
  {
    message = "Point cloud layout is invalid";
    return false;
  }

  const auto & grasp_point = input.grasp_point;
  if (!std::isfinite(grasp_point.x) || !std::isfinite(grasp_point.y) ||
    !std::isfinite(grasp_point.z))
  {
    message = "Grasp point contains non-finite values";
    return false;
  }

  geometry_msgs::msg::PoseStamped frame_pose;
  frame_pose.header = cloud.header;
  frame_pose.pose.orientation.w = 1.0;
  geometry_msgs::msg::PoseStamped frame_pose_world;
  if (!transformTarget(frame_pose, frame_pose_world, message)) {
    return false;
  }
  const Eigen::Isometry3d point_frame_to_world = poseToEigen(frame_pose_world.pose);
  const Eigen::Vector3d grasp_world = point_frame_to_world * Eigen::Vector3d(
    grasp_point.x, grasp_point.y, grasp_point.z);
  Eigen::Isometry3d fit_frame_to_world = poseToEigen(target_pose.pose);
  fit_frame_to_world.translation() = grasp_world;
  const Eigen::Isometry3d world_to_fit_frame = fit_frame_to_world.inverse();

  const std::uint16_t endian_probe = 1U;
  const bool host_is_big_endian =
    *reinterpret_cast<const std::uint8_t *>(&endian_probe) == 0U;
  const bool swap_bytes = cloud.is_bigendian != host_is_big_endian;
  std::vector<CylinderPoint> points;
  points.reserve(static_cast<std::size_t>(cloud.width) * cloud.height);
  for (std::uint32_t row = 0; row < cloud.height; ++row) {
    const std::uint8_t * row_data = cloud.data.data() + row * cloud.row_step;
    for (std::uint32_t column = 0; column < cloud.width; ++column) {
      const std::uint8_t * point_data = row_data + column * cloud.point_step;
      const double x = readFloat32(point_data + field_x->offset, swap_bytes);
      const double y = readFloat32(point_data + field_y->offset, swap_bytes);
      const double z = readFloat32(point_data + field_z->offset, swap_bytes);
      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
        message = "Point cloud contains non-finite XYZ values";
        return false;
      }
      const Eigen::Vector3d point_fit =
        world_to_fit_frame * (point_frame_to_world * Eigen::Vector3d(x, y, z));
      points.push_back({point_fit.x(), point_fit.y(), point_fit.z()});
    }
  }

  FittedCylinder fitted;
  if (!fitMinimumEnclosingCylinder(points, fitted, message)) {
    return false;
  }

  CachedCylinder cached;
  cached.primitive.type = shape_msgs::msg::SolidPrimitive::CYLINDER;
  cached.primitive.dimensions.resize(2);
  cached.primitive.dimensions[shape_msgs::msg::SolidPrimitive::CYLINDER_HEIGHT] =
    fitted.height;
  cached.primitive.dimensions[shape_msgs::msg::SolidPrimitive::CYLINDER_RADIUS] =
    fitted.radius;

  Eigen::Isometry3d cylinder_in_tcp = Eigen::Isometry3d::Identity();
  cylinder_in_tcp.translation() = Eigen::Vector3d(
    fitted.center.x, fitted.center.y, fitted.center.z);
  cylinder_in_tcp.linear() = Eigen::Quaterniond(
    fitted.orientation_xyzw[3], fitted.orientation_xyzw[0],
    fitted.orientation_xyzw[1], fitted.orientation_xyzw[2]).toRotationMatrix();
  Eigen::Isometry3d tcp_in_link6 = Eigen::Isometry3d::Identity();
  tcp_in_link6.translation().z() = grasp_object_tcp_offset_z_;
  cached.pose_in_link6 = eigenToPose(tcp_in_link6 * cylinder_in_tcp);
  cylinderCache(arm_name) = cached;

  RCLCPP_INFO(
    get_logger(),
    "%s grasp cylinder cached: height %.6f m, radius %.6f m",
    arm_name.c_str(), fitted.height, fitted.radius);
  return true;
}

bool PathPlanningNode::updateMeshCache(
  const std::string & arm_name,
  const std::string & target_name,
  const path_planning_interfaces::msg::AxisGraspInput & input,
  const geometry_msgs::msg::PoseStamped & target_pose,
  std::string & message)
{
  if (!hasFiniteMeshInput(input)) {
    message = "Mesh input contains non-finite values";
    return false;
  }
  const std::string mesh_resource =
    grasp_mesh_root_resource_ + "/" +
    percentEncodeUriPathSegment(target_name) + "/collision.stl";
  const std::string visual_resource =
    grasp_mesh_root_resource_ + "/" +
    percentEncodeUriPathSegment(target_name) + "/visual.dae";
  std::unique_ptr<shapes::Mesh> mesh_shape(
    shapes::createMeshFromResource(mesh_resource));
  if (!mesh_shape) {
    message = "Failed to load grasp mesh: " + mesh_resource;
    return false;
  }

  shapes::ShapeMsg shape_message;
  if (!shapes::constructMsgFromShape(mesh_shape.get(), shape_message)) {
    message = "Failed to convert grasp mesh: " + mesh_resource;
    return false;
  }
  const auto * mesh_message = boost::get<shape_msgs::msg::Mesh>(&shape_message);
  if (!mesh_message || mesh_message->vertices.empty()) {
    message = "Grasp mesh has no vertices: " + mesh_resource;
    return false;
  }

  Eigen::Vector3d bbox_min = Eigen::Vector3d::Constant(
    std::numeric_limits<double>::infinity());
  Eigen::Vector3d bbox_max = Eigen::Vector3d::Constant(
    -std::numeric_limits<double>::infinity());
  for (const auto & vertex : mesh_message->vertices) {
    bbox_min.x() = std::min(bbox_min.x(), vertex.x);
    bbox_min.y() = std::min(bbox_min.y(), vertex.y);
    bbox_min.z() = std::min(bbox_min.z(), vertex.z);
    bbox_max.x() = std::max(bbox_max.x(), vertex.x);
    bbox_max.y() = std::max(bbox_max.y(), vertex.y);
    bbox_max.z() = std::max(bbox_max.z(), vertex.z);
  }
  const Eigen::Vector3d bbox_size = bbox_max - bbox_min;
  const double model_axis_length = bbox_size.z();

  const Eigen::Vector3d axis_start(
    input.axis_start_world.x,
    input.axis_start_world.y,
    input.axis_start_world.z);
  const Eigen::Vector3d axis_end(
    input.axis_end_world.x,
    input.axis_end_world.y,
    input.axis_end_world.z);
  const Eigen::Vector3d grasp_point(
    input.grasp_point_world.x,
    input.grasp_point_world.y,
    input.grasp_point_world.z);
  const Eigen::Vector3d observed_axis = axis_end - axis_start;
  const double observed_axis_length = observed_axis.norm();
  if (observed_axis_length <= 1e-9 || model_axis_length <= 1e-9) {
    message = "Grasp mesh axis length must be positive";
    return false;
  }

  const Eigen::Vector3d axis_unit = observed_axis / observed_axis_length;
  const Eigen::Vector3d scale_origin = axis_start +
    (grasp_point - axis_start).dot(axis_unit) * axis_unit;
  const double scale = model_axis_length / observed_axis_length;
  const Eigen::Vector3d scaled_axis_start =
    scale_origin + scale * (axis_start - scale_origin);
  const Eigen::Vector3d scaled_axis_end =
    scale_origin + scale * (axis_end - scale_origin);
  const Eigen::Vector3d scaled_axis = scaled_axis_end - scaled_axis_start;

  Eigen::Isometry3d world_model = Eigen::Isometry3d::Identity();
  world_model.linear() = Eigen::Quaterniond::FromTwoVectors(
    Eigen::Vector3d::UnitZ(), scaled_axis.normalized()).toRotationMatrix();

  const Eigen::Vector3d model_center_axis_start(
    0.5 * (bbox_min.x() + bbox_max.x()),
    0.5 * (bbox_min.y() + bbox_max.y()),
    bbox_min.z());
  world_model.translation() =
    scaled_axis_start -
    world_model.linear() * model_center_axis_start;

  Eigen::Isometry3d world_tcp = poseToEigen(target_pose.pose);
  world_tcp.translation() = grasp_point;
  const Eigen::Vector3d center_axis_point_in_model =
    world_model.inverse() * scale_origin;

  Eigen::Isometry3d tcp_model = Eigen::Isometry3d::Identity();
  tcp_model.linear() =
    Eigen::AngleAxisd(
      -M_PI_2, Eigen::Vector3d::UnitY()).toRotationMatrix();
  tcp_model.translation() =
    -tcp_model.linear() * center_axis_point_in_model;
  Eigen::Isometry3d link6_tcp = Eigen::Isometry3d::Identity();
  link6_tcp.translation().z() = grasp_object_tcp_offset_z_;

  CachedMesh cached;
  cached.target_name = target_name;
  cached.visual_resource = visual_resource;
  cached.mesh = *mesh_message;
  cached.pose_in_link6 = eigenToPose(link6_tcp * tcp_model);
  cached.pose_in_world = eigenToPose(world_model);
  meshCache(arm_name) = std::move(cached);
  publishMeshVisual(arm_name, false);

  RCLCPP_INFO(
    get_logger(),
    "%s grasp mesh cached: %s, axis length %.6f m",
    arm_name.c_str(), target_name.c_str(), model_axis_length);
  return true;
}

std::string PathPlanningNode::graspObjectId(const std::string & arm_name) const
{
  return grasp_object_type_ == GraspObjectType::Ellipsoid ?
         arm_name + "_grasp_ellipsoid" : arm_name + "_grasp_obj";
}

const std::optional<PathPlanningNode::CachedCylinder> &
PathPlanningNode::cylinderCache(const std::string & arm_name) const
{
  return arm_name == "left" ? left_cylinder_cache_ : right_cylinder_cache_;
}

std::optional<PathPlanningNode::CachedCylinder> &
PathPlanningNode::cylinderCache(const std::string & arm_name)
{
  return arm_name == "left" ? left_cylinder_cache_ : right_cylinder_cache_;
}

const std::optional<PathPlanningNode::CachedMesh> &
PathPlanningNode::meshCache(const std::string & arm_name) const
{
  return arm_name == "left" ? left_mesh_cache_ : right_mesh_cache_;
}

std::optional<PathPlanningNode::CachedMesh> &
PathPlanningNode::meshCache(const std::string & arm_name)
{
  return arm_name == "left" ? left_mesh_cache_ : right_mesh_cache_;
}

bool PathPlanningNode::setGraspObject(
  const std::string & arm_name,
  const bool enabled,
  std::string & message)
{
  if (grasp_object_type_ == GraspObjectType::Ellipsoid) {
    return setGraspEllipsoid(arm_name, enabled, message);
  }

  const std::string link_name =
    arm_name == "left" ? left_end_effector_link_ : right_end_effector_link_;
  const std::string link_prefix = arm_name + "_";
  const std::string object_id = graspObjectId(arm_name);

  moveit_msgs::msg::AttachedCollisionObject grasp_object;
  grasp_object.link_name = link_name;
  grasp_object.object.header.frame_id = link_name;
  grasp_object.object.id = object_id;

  if (enabled) {
    grasp_object.touch_links = {
      link_name,
      link_prefix + "gripper_base",
      link_prefix + "link7",
      link_prefix + "link8"
    };
    if (grasp_object_type_ == GraspObjectType::Cylinder) {
      const auto & cached = cylinderCache(arm_name);
      if (!cached) {
        message = "grasp object cache is empty";
        return false;
      }
      grasp_object.object.primitives.push_back(cached->primitive);
      grasp_object.object.primitive_poses.push_back(cached->pose_in_link6);
    } else {
      const auto & cached = meshCache(arm_name);
      if (!cached) {
        message = "grasp object cache is empty";
        return false;
      }
      grasp_object.object.meshes.push_back(cached->mesh);
      grasp_object.object.mesh_poses.push_back(cached->pose_in_link6);
    }
    grasp_object.object.operation = moveit_msgs::msg::CollisionObject::ADD;
  } else {
    grasp_object.object.operation = moveit_msgs::msg::CollisionObject::REMOVE;
  }

  if (enabled && !updateGraspObjectAllowedCollisions(
      arm_name, object_id, true, message))
  {
    return false;
  }
  if (!planning_scene_->applyAttachedCollisionObject(grasp_object)) {
    message = enabled ?
      "failed to attach grasp object" : "failed to remove grasp object";
    return false;
  }

  if (!enabled) {
    const auto known_objects = planning_scene_->getKnownObjectNames();
    if (std::find(known_objects.begin(), known_objects.end(), object_id) !=
      known_objects.end())
    {
      moveit_msgs::msg::CollisionObject world_object;
      world_object.header.frame_id = world_frame_;
      world_object.id = object_id;
      world_object.operation = moveit_msgs::msg::CollisionObject::REMOVE;
      if (!planning_scene_->applyCollisionObject(world_object)) {
        message = "failed to remove grasp object from world";
        return false;
      }
    }
    if (!updateGraspObjectAllowedCollisions(
        arm_name, object_id, false, message))
    {
      return false;
    }
  }

  if (grasp_object_type_ == GraspObjectType::Mesh) {
    if (enabled) {
      publishMeshVisual(arm_name, true);
    } else {
      deleteMeshVisual(arm_name);
    }
  }

  message = enabled ? "grasp object attached" : "grasp object removed";
  return true;
}

void PathPlanningNode::publishMeshVisual(
  const std::string & arm_name,
  const bool attached)
{
  const auto & cached = meshCache(arm_name);
  if (!cached || !grasp_visual_publisher_) {
    return;
  }

  visualization_msgs::msg::Marker marker;
  marker.header.stamp = now();
  marker.header.frame_id = attached ?
    (arm_name == "left" ? left_end_effector_link_ : right_end_effector_link_) :
    world_frame_;
  marker.ns = "grasp_object_visual";
  marker.id = arm_name == "left" ? 0 : 1;
  marker.type = visualization_msgs::msg::Marker::MESH_RESOURCE;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.pose = attached ? cached->pose_in_link6 : cached->pose_in_world;
  marker.scale.x = 1.0;
  marker.scale.y = 1.0;
  marker.scale.z = 1.0;
  marker.color.r = 1.0F;
  marker.color.g = 1.0F;
  marker.color.b = 1.0F;
  marker.color.a = 1.0F;
  marker.frame_locked = attached;
  marker.mesh_resource = cached->visual_resource;
  marker.mesh_use_embedded_materials = true;

  visualization_msgs::msg::MarkerArray markers;
  markers.markers.push_back(std::move(marker));
  grasp_visual_publisher_->publish(markers);
}

void PathPlanningNode::deleteMeshVisual(const std::string & arm_name)
{
  if (!grasp_visual_publisher_) {
    return;
  }

  visualization_msgs::msg::Marker marker;
  marker.header.stamp = now();
  marker.header.frame_id = world_frame_;
  marker.ns = "grasp_object_visual";
  marker.id = arm_name == "left" ? 0 : 1;
  marker.action = visualization_msgs::msg::Marker::DELETE;

  visualization_msgs::msg::MarkerArray markers;
  markers.markers.push_back(std::move(marker));
  grasp_visual_publisher_->publish(markers);
}

bool PathPlanningNode::setGraspEllipsoid(
  const std::string & arm_name,
  const bool enabled,
  std::string & message)
{
  const std::string link_name =
    arm_name == "left" ? left_end_effector_link_ : right_end_effector_link_;
  const std::string link_prefix = arm_name + "_";
  const std::string ellipsoid_id = arm_name + "_grasp_ellipsoid";

  moveit_msgs::msg::AttachedCollisionObject ellipsoid;
  ellipsoid.link_name = link_name;
  ellipsoid.object.header.frame_id = link_name;
  ellipsoid.object.id = ellipsoid_id;

  if (enabled) {
    ellipsoid.touch_links = {
      link_name,
      link_prefix + "gripper_base",
      link_prefix + "link7",
      link_prefix + "link8"
    };
    ellipsoid.object.meshes.push_back(grasp_ellipsoid_mesh_);

    geometry_msgs::msg::Pose pose;
    pose.orientation.w = 1.0;
    pose.position.z = grasp_ellipsoid_center_z_;
    ellipsoid.object.mesh_poses.push_back(pose);
    ellipsoid.object.operation = moveit_msgs::msg::CollisionObject::ADD;
  } else {
    ellipsoid.object.operation = moveit_msgs::msg::CollisionObject::REMOVE;
  }

  if (enabled && !updateGraspEllipsoidAllowedCollisions(arm_name, true, message)) {
    return false;
  }

  if (!planning_scene_->applyAttachedCollisionObject(ellipsoid)) {
    message = enabled ?
      "failed to attach grasp ellipsoid" :
      "failed to remove grasp ellipsoid";
    return false;
  }

  if (!enabled) {
    const auto known_objects = planning_scene_->getKnownObjectNames();
    if (std::find(known_objects.begin(), known_objects.end(), ellipsoid_id) !=
      known_objects.end())
    {
      moveit_msgs::msg::CollisionObject world_object;
      world_object.header.frame_id = world_frame_;
      world_object.id = ellipsoid_id;
      world_object.operation = moveit_msgs::msg::CollisionObject::REMOVE;
      if (!planning_scene_->applyCollisionObject(world_object)) {
        message = "failed to remove grasp ellipsoid world object";
        return false;
      }
    }
  }

  if (!enabled && !updateGraspEllipsoidAllowedCollisions(arm_name, false, message)) {
    return false;
  }

  message = enabled ?
    "grasp ellipsoid attached" :
    "grasp ellipsoid removed";
  return true;
}

bool PathPlanningNode::updateGraspEllipsoidAllowedCollisions(
  const std::string & arm_name,
  const bool allowed,
  std::string & message)
{
  return updateGraspObjectAllowedCollisions(
    arm_name, arm_name + "_grasp_ellipsoid", allowed, message);
}

bool PathPlanningNode::updateGraspObjectAllowedCollisions(
  const std::string & arm_name,
  const std::string & object_id,
  const bool allowed,
  std::string & message)
{
  moveit_msgs::msg::AllowedCollisionMatrix acm;
  if (!fetchAllowedCollisionMatrix(acm, message)) {
    return false;
  }

  const std::string link_name =
    arm_name == "left" ? left_end_effector_link_ : right_end_effector_link_;
  const std::string link_prefix = arm_name + "_";
  const std::vector<std::string> touch_links = {
    link_name,
    link_prefix + "gripper_base",
    link_prefix + "link7",
    link_prefix + "link8"
  };

  for (const auto & touch_link : touch_links) {
    setAcmPair(acm, object_id, touch_link, allowed, allowed);
  }

  moveit_msgs::msg::PlanningScene scene;
  scene.is_diff = true;
  scene.allowed_collision_matrix = acm;
  if (!planning_scene_->applyPlanningScene(scene)) {
    message = allowed ?
      "failed to allow grasp object touch links" :
      "failed to clear grasp object touch link allowances";
    return false;
  }

  message = allowed ?
    "grasp object touch links allowed" :
    "grasp object touch link allowances cleared";
  return true;
}

bool PathPlanningNode::fetchAllowedCollisionMatrix(
  moveit_msgs::msg::AllowedCollisionMatrix & acm,
  std::string & message)
{
  if (!get_planning_scene_client_->wait_for_service(kPlanningSceneServiceWaitTimeout)) {
    message = "get planning scene service unavailable";
    return false;
  }

  auto request = std::make_shared<GetPlanningScene::Request>();
  request->components.components =
    moveit_msgs::msg::PlanningSceneComponents::ALLOWED_COLLISION_MATRIX;
  auto future = get_planning_scene_client_->async_send_request(request);
  if (future.wait_for(kPlanningSceneServiceWaitTimeout) != std::future_status::ready) {
    message = "timeout fetching allowed collision matrix";
    return false;
  }

  const auto response = future.get();
  if (!response) {
    message = "get planning scene returned no response";
    return false;
  }
  acm = response->scene.allowed_collision_matrix;
  if (acm.entry_values.size() < acm.entry_names.size()) {
    acm.entry_values.resize(acm.entry_names.size());
  }
  for (auto & entry : acm.entry_values) {
    entry.enabled.resize(acm.entry_names.size(), false);
  }
  return true;
}

bool PathPlanningNode::transformTarget(
  const geometry_msgs::msg::PoseStamped & input,
  geometry_msgs::msg::PoseStamped & output,
  std::string & message) const
{
  if (input.header.frame_id == world_frame_) {
    output = input;
    return true;
  }

  try {
    output = tf_buffer_->transform(input, world_frame_, tf2::durationFromSec(1.0));
    return true;
  } catch (const tf2::TransformException & error) {
    message = std::string("Transform failed: ") + error.what();
    return false;
  }
}

bool PathPlanningNode::executeTrajectory(
  const std::string & arm_name,
  const moveit_msgs::msg::RobotTrajectory & trajectory,
  std::string & message)
{
  const auto action_client =
    arm_name == "left" ? left_trajectory_client_ : right_trajectory_client_;
  const std::string joint_prefix = arm_name + "_";

  if (!action_client->wait_for_action_server(kActionServerWaitTimeout)) {
    message = "Action server unavailable";
    return false;
  }

  FollowJointTrajectory::Goal goal;
  goal.trajectory = trajectory.joint_trajectory;
  goal.multi_dof_trajectory = trajectory.multi_dof_joint_trajectory;
  for (auto & joint_name : goal.trajectory.joint_names) {
    if (joint_name.rfind(joint_prefix, 0) == 0) {
      joint_name.erase(0, joint_prefix.size());
    }
  }

  try {
    auto goal_future = action_client->async_send_goal(goal);
    goal_future.wait();
    const auto goal_handle = goal_future.get();
    if (!goal_handle) {
      message = "Trajectory goal rejected";
      return false;
    }

    auto result_future = action_client->async_get_result(goal_handle);
    bool cancellation_sent = false;
    while (result_future.wait_for(std::chrono::milliseconds(50)) != std::future_status::ready) {
      if (cancel_requested_.load() && !cancellation_sent) {
        action_client->async_cancel_goal(goal_handle);
        cancellation_sent = true;
      }
    }
    const auto wrapped_result = result_future.get();

    if (wrapped_result.code != rclcpp_action::ResultCode::SUCCEEDED) {
      if (wrapped_result.code == rclcpp_action::ResultCode::CANCELED) {
        message = "Trajectory execution canceled";
      } else {
        message = "Trajectory execution aborted";
      }
      return false;
    }
    if (!wrapped_result.result) {
      message = "Trajectory execution aborted: Action returned no result";
      return false;
    }
    if (wrapped_result.result->error_code !=
      FollowJointTrajectory::Result::SUCCESSFUL)
    {
      message = "Trajectory execution aborted";
      if (!wrapped_result.result->error_string.empty()) {
        message += ": " + wrapped_result.result->error_string;
      }
      return false;
    }

    message = "Planning and trajectory execution succeeded";
    return true;
  } catch (const std::exception & error) {
    message = std::string("Trajectory execution aborted: ") + error.what();
    return false;
  }
}

bool PathPlanningNode::executeGripperCommand(
  const std::string & arm_name,
  double opening,
  double move_duration,
  std::string & message)
{
  const auto action_client =
    arm_name == "left" ? left_gripper_client_ : right_gripper_client_;

  if (!action_client->wait_for_action_server(kActionServerWaitTimeout)) {
    message = "gripper action server is unavailable";
    return false;
  }

  FollowJointTrajectory::Goal goal;
  goal.trajectory.joint_names = {"joint7"};
  goal.trajectory.points.resize(1);
  goal.trajectory.points.front().positions = {opening / 2.0};
  goal.trajectory.points.front().effort = {gripper_effort_};
  goal.trajectory.points.front().time_from_start =
    rclcpp::Duration::from_seconds(move_duration);
  const bool contact_close_command =
    opening <= gripper_contact_opening_threshold_;
  if (contact_close_command) {
    goal.goal_time_tolerance =
      rclcpp::Duration::from_seconds(gripper_contact_goal_time_tolerance_);
  }
  if (contact_close_command && gripper_contact_goal_tolerance_ > 0.0) {
    control_msgs::msg::JointTolerance tolerance;
    tolerance.name = "joint7";
    tolerance.position = gripper_contact_goal_tolerance_;
    tolerance.velocity = 0.0;
    tolerance.acceleration = 0.0;
    goal.goal_tolerance = {tolerance};
  }

  try {
    auto goal_future = action_client->async_send_goal(goal);
    goal_future.wait();
    const auto goal_handle = goal_future.get();
    if (!goal_handle) {
      message = "gripper trajectory goal was rejected";
      return false;
    }

    auto result_future = action_client->async_get_result(goal_handle);
    bool cancellation_sent = false;
    while (result_future.wait_for(std::chrono::milliseconds(50)) != std::future_status::ready) {
      if (cancel_requested_.load() && !cancellation_sent) {
        action_client->async_cancel_goal(goal_handle);
        cancellation_sent = true;
      }
    }
    const auto wrapped_result = result_future.get();

    if (wrapped_result.code != rclcpp_action::ResultCode::SUCCEEDED) {
      if (contact_close_command && treat_gripper_contact_abort_as_success_ &&
        wrapped_result.code == rclcpp_action::ResultCode::ABORTED)
      {
        message = "gripper close reached contact; treating aborted goal tolerance as success";
        return true;
      }
      if (wrapped_result.code == rclcpp_action::ResultCode::CANCELED) {
        message = "gripper trajectory execution was canceled";
      } else {
        message = "gripper trajectory execution was aborted";
      }
      return false;
    }
    if (!wrapped_result.result) {
      message = "gripper trajectory execution returned no result";
      return false;
    }
    if (wrapped_result.result->error_code !=
      FollowJointTrajectory::Result::SUCCESSFUL)
    {
      if (contact_close_command && treat_gripper_contact_abort_as_success_ &&
        wrapped_result.result->error_code ==
        FollowJointTrajectory::Result::GOAL_TOLERANCE_VIOLATED)
      {
        message = "gripper close reached contact; treating goal tolerance violation as success";
        return true;
      }
      message = "gripper trajectory execution was aborted";
      if (!wrapped_result.result->error_string.empty()) {
        message += ": " + wrapped_result.result->error_string;
      }
      return false;
    }

    message = "gripper execution succeeded";
    return true;
  } catch (const std::exception & error) {
    message = std::string("gripper trajectory execution was aborted: ") +
      error.what();
    return false;
  }
}

}  // namespace path_planning_server
