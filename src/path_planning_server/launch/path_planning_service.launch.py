from ament_index_python.packages import get_package_share_directory
from copy import deepcopy
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from moveit_configs_utils import MoveItConfigsBuilder
import yaml


def generate_launch_description():
    package_share = get_package_share_directory("path_planning_server")

    with open(
        f"{package_share}/config/chomp_optimizer.yaml",
        encoding="utf-8",
    ) as stream:
        chomp_optimizer = yaml.safe_load(stream)
    with open(
        f"{package_share}/config/stomp_optimizer.yaml",
        encoding="utf-8",
    ) as stream:
        stomp_optimizer = yaml.safe_load(stream)
    with open(
        f"{package_share}/config/trajectory_preference_optimizer.yaml",
        encoding="utf-8",
    ) as stream:
        trajectory_preference_optimizer = yaml.safe_load(stream)

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

    optimizer_settings = {}
    optimizer_settings.update(chomp_optimizer["trajectory_optimizer"])
    optimizer_settings.update(stomp_optimizer["trajectory_optimizer"])
    optimizer_settings.update(
        trajectory_preference_optimizer["trajectory_optimizer"]
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
    time_parameterization_adapter = (
        "default_planning_response_adapters/AddTimeOptimalParameterization"
    )
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
        preference_adapters = preference_pipeline["response_adapters"]
        preference_adapters.insert(
            preference_adapters.index(optimizer_adapter) + 1,
            "path_planning_server/TrajectoryPreferenceAdapter",
        )
        runtime_pipelines[
            f"ompl_{optimizer_mode}_preference"
        ] = preference_pipeline
        waypoint_pipeline = deepcopy(pipeline)
        waypoint_pipeline["planning_plugins"] = [
            "path_planning_server/MultiSegmentOmplPlanner"
        ]
        runtime_pipelines[f"ompl_waypoint_{optimizer_mode}"] = waypoint_pipeline
        waypoint_preference_pipeline = deepcopy(preference_pipeline)
        waypoint_preference_pipeline["planning_plugins"] = [
            "path_planning_server/MultiSegmentOmplPlanner"
        ]
        runtime_pipelines[
            f"ompl_waypoint_{optimizer_mode}_preference"
        ] = waypoint_preference_pipeline
    moveit_config.planning_pipelines = runtime_pipelines

    scene_id = LaunchConfiguration("scene_id")
    use_rviz = LaunchConfiguration("use_rviz")
    use_virtual_joint_states = LaunchConfiguration("use_virtual_joint_states")
    execute_trajectory = LaunchConfiguration("execute_trajectory")
    grasp_object_type = LaunchConfiguration("grasp_object_type")

    static_world_transform = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_to_dual_piper",
        output="screen",
        arguments=[
            "--x", "0",
            "--y", "0",
            "--z", "0",
            "--qx", "0",
            "--qy", "0",
            "--qz", "0",
            "--qw", "1",
            "--frame-id", "world",
            "--child-frame-id", "rviz_piper_world",
        ],
    )

    joint_state_publisher = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="dual_piper_joint_state_publisher",
        output="screen",
        condition=IfCondition(use_virtual_joint_states),
        parameters=[{
            "rate": 30,
            "source_list": ["/path_planning_test_joint_states"],
        }],
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="dual_piper_robot_state_publisher",
        output="screen",
        parameters=[
            moveit_config.robot_description,
            {"publish_frequency": 30.0},
        ],
    )

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        name="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {
                "allow_trajectory_execution": False,
            },
        ],
    )

    planning_server = Node(
        package="path_planning_server",
        executable="path_planning_server_node",
        name="path_planning_server",
        output="screen",
        parameters=[
            f"{package_share}/config/planner.yaml",
            f"{package_share}/config/grasp_ellipsoid.yaml",
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
            {
                "default_scene_id": ParameterValue(
                    scene_id,
                    value_type=int,
                ),
                "execute_trajectory": ParameterValue(
                    execute_trajectory,
                    value_type=bool,
                ),
                "grasp_object.type": ParameterValue(
                    grasp_object_type,
                    value_type=str,
                ),
            },
        ],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="path_planning_rviz",
        output="screen",
        condition=IfCondition(use_rviz),
        arguments=[
            "-d",
            f"{package_share}/config/moveit.rviz",
        ],
        parameters=[
            moveit_config.planning_pipelines,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument("scene_id", default_value="1"),
        DeclareLaunchArgument("use_rviz", default_value="true"),
        DeclareLaunchArgument(
            "use_virtual_joint_states",
            default_value="true",
            description="Publish virtual joint states for RViz-only validation.",
        ),
        DeclareLaunchArgument(
            "execute_trajectory",
            default_value="false",
            description="Execute successful plans through Piper trajectory actions.",
        ),
        DeclareLaunchArgument(
            "grasp_object_type",
            default_value="mesh",
            choices=["ellipsoid", "cylinder", "mesh"],
            description="Collision model used by /plan_to_pose_v2 and /plan_to_pose_v3.",
        ),
        static_world_transform,
        joint_state_publisher,
        robot_state_publisher,
        move_group,
        planning_server,
        rviz,
    ])
