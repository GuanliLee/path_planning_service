"""Start only MoveGroup with the 0819 waypoint planning pipelines.

This launch file intentionally does not start robot drivers, publishers, the path
planning server, or RViz.  It is used to replace an older MoveGroup in a running
hardware stack without disturbing the CAN-controlled arms.
"""

from copy import deepcopy

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder
import yaml


def _runtime_planning_pipelines(moveit_config, package_share):
    optimizer_settings = {}
    for filename in (
        "chomp_optimizer.yaml",
        "stomp_optimizer.yaml",
        "trajectory_preference_optimizer.yaml",
    ):
        with open(
            f"{package_share}/config/{filename}",
            encoding="utf-8",
        ) as stream:
            optimizer_settings.update(
                yaml.safe_load(stream)["trajectory_optimizer"]
            )

    ompl_pipeline = moveit_config.planning_pipelines["ompl"]
    runtime_pipelines = {
        "planning_pipelines": [
            "ompl",
            "ompl_none",
            "ompl_chomp",
            "ompl_stomp",
            "ompl_none_preference",
            "ompl_chomp_preference",
            "ompl_stomp_preference",
            "ompl_waypoint_none",
            "ompl_waypoint_chomp",
            "ompl_waypoint_stomp",
            "ompl_waypoint_none_preference",
            "ompl_waypoint_chomp_preference",
            "ompl_waypoint_stomp_preference",
        ],
        "default_planning_pipeline": "ompl",
        "ompl": deepcopy(ompl_pipeline),
    }

    optimizer_adapter = "path_planning_server/TrajectoryOptimizerAdapter"
    preference_adapter = "path_planning_server/TrajectoryPreferenceAdapter"
    time_parameterization_adapter = (
        "default_planning_response_adapters/AddTimeOptimalParameterization"
    )
    waypoint_plugin = "path_planning_server/MultiSegmentOmplPlanner"

    for optimizer_mode in ("none", "chomp", "stomp"):
        pipeline = deepcopy(ompl_pipeline)
        response_adapters = pipeline["response_adapters"]
        response_adapters.insert(
            response_adapters.index(time_parameterization_adapter),
            optimizer_adapter,
        )
        pipeline["trajectory_optimizer"] = deepcopy(optimizer_settings)
        pipeline["trajectory_optimizer"]["mode"] = optimizer_mode
        runtime_pipelines[f"ompl_{optimizer_mode}"] = pipeline

        preference_pipeline = deepcopy(pipeline)
        preference_pipeline["response_adapters"].insert(
            preference_pipeline["response_adapters"].index(optimizer_adapter) + 1,
            preference_adapter,
        )
        runtime_pipelines[
            f"ompl_{optimizer_mode}_preference"
        ] = preference_pipeline

        waypoint_pipeline = deepcopy(pipeline)
        waypoint_pipeline["planning_plugins"] = [waypoint_plugin]
        runtime_pipelines[f"ompl_waypoint_{optimizer_mode}"] = waypoint_pipeline

        waypoint_preference_pipeline = deepcopy(preference_pipeline)
        waypoint_preference_pipeline["planning_plugins"] = [waypoint_plugin]
        runtime_pipelines[
            f"ompl_waypoint_{optimizer_mode}_preference"
        ] = waypoint_preference_pipeline

    return runtime_pipelines


def generate_launch_description():
    package_share = get_package_share_directory("path_planning_server")
    moveit_config = (
        MoveItConfigsBuilder(
            "agilex_dual_piper_rviz",
            package_name="path_planning_server",
        )
        .robot_description(file_path="generated/dual_piper.urdf")
        .robot_description_semantic(file_path="config/dual_piper.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .planning_pipelines(
            default_planning_pipeline="ompl",
            pipelines=["ompl"],
            load_all=False,
        )
        .trajectory_execution(
            file_path="config/moveit_controllers.yaml",
            moveit_manage_controllers=False,
        )
        .planning_scene_monitor(
            publish_planning_scene=True,
            publish_geometry_updates=True,
            publish_state_updates=True,
            publish_transforms_updates=True,
            publish_robot_description=True,
            publish_robot_description_semantic=True,
        )
        .to_moveit_configs()
    )
    moveit_config.planning_pipelines = _runtime_planning_pipelines(
        moveit_config,
        package_share,
    )

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        name="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {"allow_trajectory_execution": False},
        ],
    )
    return LaunchDescription([move_group])
