from pathlib import Path


SERVICE_FILE = (
    Path(__file__).parents[2]
    / 'path_planning_interfaces'
    / 'srv'
    / 'PlanToPoseV3.srv'
)


def service_fields():
    request, response = SERVICE_FILE.read_text(encoding='utf-8').split('---')

    def fields(section):
        return [
            line.strip()
            for line in section.splitlines()
            if line.strip() and not line.lstrip().startswith('#')
        ]

    return fields(request), fields(response)


def test_plan_to_pose_v3_matches_0819_interface():
    request, response = service_fields()

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
        'string trajectory_optimizer_mode',
        'bool use_preference_optimizer',
        'bool use_tcp_position_constraint',
        'string ompl_waypoint_config',
        'bool adjust_grasp_mesh_min_z',
        'bool allow_grasp_mesh_z_lowering',
        'float64 grasp_mesh_min_z_m',
        'float64 max_grasp_mesh_z_adjustment_m',
    ]
    assert response == [
        'bool success',
        'string message',
        'moveit_msgs/RobotTrajectory trajectory',
    ]
