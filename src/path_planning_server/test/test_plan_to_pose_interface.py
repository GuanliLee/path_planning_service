from pathlib import Path

import yaml


SERVICE_FILE = (
    Path(__file__).parents[2]
    / 'path_planning_interfaces'
    / 'srv'
    / 'PlanToPose.srv'
)
V2_SERVICE_FILE = SERVICE_FILE.with_name('PlanToPoseV2.srv')
CARTESIAN_SERVICE_FILE = SERVICE_FILE.with_name('PlanCartesianPath.srv')
JOINT_SERVICE_FILE = (
    Path(__file__).parents[2]
    / "path_planning_interfaces"
    / "srv"
    / "PlanToJoints.srv"
)
SERVER_SOURCE = Path(__file__).parents[1] / 'src' / 'path_planning_node.cpp'
PLANNER_CONFIG = Path(__file__).parents[1] / 'config' / 'planner.yaml'


def service_fields(service_file=SERVICE_FILE):
    request, response = service_file.read_text(encoding='utf-8').split('---')

    def fields(section):
        return [
            line.strip()
            for line in section.splitlines()
            if line.strip() and not line.lstrip().startswith('#')
        ]

    return fields(request), fields(response)


def test_plan_to_pose_exposes_optional_gripper_command():
    request, response = service_fields()

    assert request == [
        'string arm_name',
        'uint32 scene_id',
        'geometry_msgs/PoseStamped target_pose',
        'bool gripper_command',
        'float64 gripper_opening_m',
        'bool keep_grasp_ellipsoid',
    ]
    assert response == [
        'bool success',
        'string message',
        'moveit_msgs/RobotTrajectory trajectory',
    ]


def test_plan_to_joints_exposes_exact_six_axis_target():
    request, response = service_fields(JOINT_SERVICE_FILE)

    assert request == [
        "string arm_name",
        "uint32 scene_id",
        "float64[6] joint_positions",
        "bool gripper_command",
        "float64 gripper_opening_m",
        "bool keep_grasp_ellipsoid",
    ]
    assert response == [
        "bool success",
        "string message",
        "moveit_msgs/RobotTrajectory trajectory",
    ]

    source = SERVER_SOURCE.read_text(encoding="utf-8")
    assert 'create_service<PlanToJoints>(' in source
    assert '"/plan_to_joints"' in source
    assert "setJointValueTarget(target_joints)" in source


def test_cartesian_service_requires_a_complete_collision_checked_path():
    request, response = service_fields(CARTESIAN_SERVICE_FILE)

    assert request == [
        'string arm_name',
        'uint32 scene_id',
        'geometry_msgs/PoseStamped target_pose',
        'float64 max_velocity_scaling_factor',
        'float64 max_acceleration_scaling_factor',
        'float64 eef_step_m',
    ]
    assert response == [
        'bool success',
        'string message',
        'float64 fraction',
        'moveit_msgs/RobotTrajectory trajectory',
    ]

    source = SERVER_SOURCE.read_text(encoding='utf-8')
    assert 'create_service<PlanCartesianPath>(' in source
    assert '"/plan_cartesian_path"' in source
    assert 'computeCartesianPath(' in source
    assert 'kCompleteCartesianFraction' in source
    assert 'TimeOptimalTrajectoryGeneration' in source


def test_plan_to_pose_v2_exposes_grasp_object_fields():
    request, response = service_fields(V2_SERVICE_FILE)

    assert request == [
        'string arm_name',
        'uint32 scene_id',
        'geometry_msgs/PoseStamped target_pose',
        'string target_name',
        'bool gripper_command',
        'float64 gripper_opening_m',
        'bool keep_grasp_obj',
        'path_planning_interfaces/GraspObjectInput grasp_object',
        'bool remove_obj',
    ]
    assert response == [
        'bool success',
        'string message',
        'moveit_msgs/RobotTrajectory trajectory',
    ]

    source = SERVER_SOURCE.read_text(encoding='utf-8')
    assert 'create_service<PlanToPoseV2>(' in source
    assert '"/plan_to_pose_v2"' in source


def test_v2_grasp_object_messages():
    message_dir = SERVICE_FILE.parents[1] / 'msg'

    assert (message_dir / 'PointCloudWithGraspPoint.msg').read_text(
        encoding='utf-8'
    ).splitlines() == [
        'sensor_msgs/PointCloud2 point_cloud',
        'geometry_msgs/Point grasp_point',
    ]
    assert (message_dir / 'AxisGraspInput.msg').read_text(
        encoding='utf-8'
    ).splitlines() == [
        'geometry_msgs/Point axis_start_world',
        'geometry_msgs/Point axis_end_world',
        'geometry_msgs/Point grasp_point_world',
    ]
    assert (message_dir / 'GraspObjectInput.msg').read_text(
        encoding='utf-8'
    ).splitlines() == [
        'path_planning_interfaces/PointCloudWithGraspPoint cylinder_input',
        'path_planning_interfaces/AxisGraspInput mesh_input',
    ]


def test_planning_start_state_preserves_attached_collision_objects():
    source = SERVER_SOURCE.read_text(encoding='utf-8')

    assert 'move_group->setStartStateToCurrentState();' in source
    assert 'move_group->setStartState(*current_state);' not in source


def test_final_place_release_uses_dedicated_slow_duration():
    source = SERVER_SOURCE.read_text(encoding='utf-8')
    config = yaml.safe_load(PLANNER_CONFIG.read_text(encoding='utf-8'))
    parameters = config['path_planning_server']['ros__parameters']

    assert parameters['gripper_move_duration'] == 1.0
    assert parameters['gripper_release_duration'] == 3.0
    assert 'release_grasp_object_after_gripper ?' in source
    assert 'gripper_release_duration_ : gripper_move_duration_' in source


def test_joint_service_updates_the_configured_grasp_object_type():
    source = SERVER_SOURCE.read_text(encoding='utf-8')

    assert 'setGraspObject(request->arm_name, false' in source
    assert 'final_grasp_object_enabled, grasp_object_message' in source


def test_release_is_gripper_only_and_never_plans_a_second_arm_motion():
    source = SERVER_SOURCE.read_text(encoding='utf-8')

    assert 'const bool locked_gripper_release =' in source
    assert '!locked_gripper_release && max_joint_error > kJointTargetTolerance' in source
    assert 'Arm held at measured position for gripper release' in source


def test_release_removes_the_object_before_cartesian_retreat():
    source = SERVER_SOURCE.read_text(encoding='utf-8')
    config = yaml.safe_load(PLANNER_CONFIG.read_text(encoding='utf-8'))
    parameters = config['path_planning_server']['ros__parameters']

    assert 'grasp_object.placed_proxy_padding_m' not in parameters
    assert 'placeGraspObjectInWorld(' not in source
    assert 'placed_proxy' not in source
    assert 'setGraspObject(\n        request->arm_name, final_grasp_object_enabled' in source
