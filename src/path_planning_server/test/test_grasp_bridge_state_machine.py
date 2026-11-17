import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

from grasp_bridge_state_machine import (
    GraspBridgeStateMachine,
    MeshGeometry,
    _HYBRID_CARTESIAN_STEPS,
    _PLACE_ORIENTATION_TARGET_STEPS,
    _PLACE_TRANSPORT_STEP_ORDER,
    _active_ompl_route_config,
    _arm_joint_positions_from_state,
    _bridge_task_type_for_step,
    _build_ik_orientation_candidates,
    _candidate_indexes_for_step,
    _capture_joint_target_for_profile,
    _elevated_place_waypoint,
    _grasp_approach_vertical_deviation_degrees,
    _grasp_orientation_mode_for_profile,
    _grasp_retraction_offset_for_target,
    _grasp_shallow_offset_for_profile,
    _handoff_post_place_retreat_to_return_route,
    _is_no_ik_failure,
    _link6_pose_from_tcp_pose,
    _matrix_from_pose,
    _mesh_geometry_for_grasp_pose,
    _mesh_request_semantics_for_step,
    _normalized_target_key,
    _object_axis_in_tcp,
    _object_upright_tilt_degrees,
    _place_completion_steps,
    _post_place_cartesian_retreat_plan,
    _place_pose_with_grasp_orientation_compensation,
    _place_pose_with_limited_upright_axis_compensation,
    _place_pose_with_upright_axis_compensation,
    _phase_ompl_routes_apply_to_task,
    _pose_with_local_orientation_offset,
    _parse_mesh_geometry_payload,
    _parse_scene_float_overrides,
    _return_home_step,
    _retreat_after_grasp_required,
    _retreat_after_place_required,
    _route_config_for_step,
    _parse_scene_string_overrides,
    _parse_slot_route_overrides,
    _parse_scene_support_surfaces,
    _scene_value,
    _slot_route_value,
    _support_surface_for_place_z,
    _rotation_aligning_vectors,
    _tcp_pose_delta,
    _transform_mesh_geometry,
    _upright_target_is_selected,
    _upright_place_compensation_is_active,
    _upright_object_axis_in_tcp,
    _validated_ik_fallback_degrees,
    _vertical_post_place_retreat_required,
)


def test_scene_specific_routes_preserve_legacy_default() -> None:
    overrides = _parse_scene_string_overrides(
        ["3=right_route_to_place_scene_003.yaml"],
        "routes",
    )
    assert _scene_value("right_route_to_place.yaml", overrides, 1) == (
        "right_route_to_place.yaml"
    )
    assert _scene_value("right_route_to_place.yaml", overrides, 3) == (
        "right_route_to_place_scene_003.yaml"
    )
    with pytest.raises(ValueError, match="duplicate"):
        _parse_scene_string_overrides(["3=a.yaml", "3=b.yaml"], "routes")


def test_slot_routes_are_selected_by_scene_arm_layer_and_position() -> None:
    scene_overrides = {3: "scene_fallback.yaml"}
    slot_overrides = _parse_slot_route_overrides(
        ["3:right:1:5=scene_003_right_layer_1_position_5.yaml"],
        "slot_routes",
    )
    assert _slot_route_value(
        "legacy.yaml", scene_overrides, slot_overrides, 3, "right", 1, 5
    ) == "scene_003_right_layer_1_position_5.yaml"
    assert _slot_route_value(
        "legacy.yaml", scene_overrides, slot_overrides, 3, "right", 1, 4
    ) == "scene_fallback.yaml"
    outer_row_overrides = _parse_slot_route_overrides(
        ["3:right:1:12=position_6_route.yaml"],
        "slot_routes",
    )
    assert outer_row_overrides[(3, "right", 1, 12)] == "position_6_route.yaml"
    assert _slot_route_value(
        "legacy.yaml", scene_overrides, slot_overrides, 1, "right", 1, 5
    ) == "legacy.yaml"
    with pytest.raises(ValueError, match="duplicate"):
        _parse_slot_route_overrides(
            ["3:right:1:5=a.yaml", "3:right:1:5=b.yaml"], "slot_routes"
        )
    with pytest.raises(ValueError, match="position 1..12"):
        _parse_slot_route_overrides(["3:right:1:13=a.yaml"], "slot_routes")


def test_scene_support_surface_is_selected_below_place_tcp() -> None:
    surfaces = _parse_scene_support_surfaces(
        ["3=0.13,0.46,0.80"],
        "surfaces",
    )
    assert _support_surface_for_place_z(0.23, surfaces, 1, 0.30) == pytest.approx(
        0.23
    )
    assert _support_surface_for_place_z(0.23, surfaces, 3, 0.2224) == pytest.approx(
        0.13
    )
    assert _support_surface_for_place_z(0.23, surfaces, 3, 0.55) == pytest.approx(
        0.46
    )
    assert _support_surface_for_place_z(0.23, surfaces, 3, 0.90) == pytest.approx(
        0.80
    )
    with pytest.raises(ValueError, match="no support surface"):
        _support_surface_for_place_z(0.23, surfaces, 3, 0.10)


def test_scene_mesh_clearance_override_is_bounded_and_isolated() -> None:
    clearances = _parse_scene_float_overrides(
        ["3=0.010"],
        "clearances",
        minimum=0.0,
        maximum=0.05,
    )

    assert clearances.get(1, 0.002) == pytest.approx(0.002)
    assert clearances.get(3, 0.002) == pytest.approx(0.010)
    with pytest.raises(ValueError, match="must be numeric"):
        _parse_scene_float_overrides(
            ["3=high"],
            "clearances",
            minimum=0.0,
            maximum=0.05,
        )
    with pytest.raises(ValueError, match=r"\[0.0, 0.05\]"):
        _parse_scene_float_overrides(
            ["3=0.051"],
            "clearances",
            minimum=0.0,
            maximum=0.05,
        )


def test_hybrid_cartesian_steps_only_cover_short_task_space_motions() -> None:
    assert _HYBRID_CARTESIAN_STEPS == {
        "GRASP_APPROACH",
        "GRASP",
        "RETREAT_AFTER_GRASP",
        "PLACE_APPROACH",
    }
    assert "PRE_GRASP" not in _HYBRID_CARTESIAN_STEPS
    assert "PLACE_WAYPOINT" not in _HYBRID_CARTESIAN_STEPS
    assert "RETURN_HOME" not in _HYBRID_CARTESIAN_STEPS


def test_cartesian_service_call_forwards_hybrid_motion_profile() -> None:
    class FakeResponse:
        success = True
        message = "complete"
        fraction = 1.0

    class FakeFuture:
        def add_done_callback(self, callback) -> None:
            callback(self)

        def cancel(self) -> None:
            raise AssertionError("completed future must not be canceled")

        def result(self) -> FakeResponse:
            return FakeResponse()

    class FakeClient:
        request = None

        def wait_for_service(self, timeout_sec: float) -> bool:
            assert timeout_sec == pytest.approx(2.0)
            return True

        def call_async(self, request):
            self.request = request
            return FakeFuture()

    machine = object.__new__(GraspBridgeStateMachine)
    machine._cartesian_plan_client = FakeClient()
    machine._cartesian_plan_service_name = "/plan_cartesian_path"
    machine._service_timeout_s = 2.0
    machine._is_cancel_requested = lambda: False

    target = PoseStamped()
    target.header.frame_id = "world"
    target.pose.orientation.w = 1.0
    success, detail = GraspBridgeStateMachine._call_cartesian_plan(
        machine,
        arm_name="right",
        scene_id=1,
        link6_pose=target,
        velocity_scaling=0.1,
        acceleration_scaling=0.2,
        eef_step_m=0.005,
    )

    assert success
    assert "fraction=1.000000" in detail
    request = machine._cartesian_plan_client.request
    assert request.arm_name == "right"
    assert request.scene_id == 1
    assert request.target_pose.header.frame_id == "world"
    assert request.max_velocity_scaling_factor == pytest.approx(0.1)
    assert request.max_acceleration_scaling_factor == pytest.approx(0.2)
    assert request.eef_step_m == pytest.approx(0.005)


def test_v3_route_service_call_forwards_route_and_mesh_geometry() -> None:
    class FakeResponse:
        success = True
        message = "complete"

    class FakeFuture:
        def add_done_callback(self, callback) -> None:
            callback(self)

        def cancel(self) -> None:
            raise AssertionError("completed future must not be canceled")

        def result(self) -> FakeResponse:
            return FakeResponse()

    class FakeClient:
        request = None

        def wait_for_service(self, timeout_sec: float) -> bool:
            assert timeout_sec == pytest.approx(2.0)
            return True

        def call_async(self, request):
            self.request = request
            return FakeFuture()

    machine = object.__new__(GraspBridgeStateMachine)
    machine._pose_v3_client = FakeClient()
    machine._pose_v3_service_name = "/plan_to_pose_v3"
    machine._service_timeout_s = 2.0
    machine._is_cancel_requested = lambda: False
    machine._route_trajectory_optimizer_mode = "none"
    machine._route_use_preference_optimizer = False
    machine._route_use_tcp_position_constraint = False
    machine._grasp_object_type = "mesh"
    machine._world_frame = "world"

    target = PoseStamped()
    target.header.frame_id = "world"
    target.pose.orientation.w = 1.0
    geometry = MeshGeometry(
        frame_id="world",
        target_name="Sprite",
        axis_start=(0.1, 0.2, 0.3),
        axis_end=(0.1, 0.2, 0.5),
        grasp_point=(0.1, 0.2, 0.4),
    )
    success, detail = GraspBridgeStateMachine._call_v3_route_plan(
        machine,
        arm_name="right",
        scene_id=1,
        target_name="Sprite",
        link6_pose=target,
        route_config="right_route_to_pregrasp.yaml",
        mesh_geometry=geometry,
        keep_grasp_obj=True,
        adjust_grasp_mesh_min_z=True,
        allow_grasp_mesh_z_lowering=True,
        grasp_mesh_min_z_m=0.232,
        max_grasp_mesh_z_adjustment_m=0.05,
    )

    assert success
    assert detail == "complete"
    request = machine._pose_v3_client.request
    assert request.arm_name == "right"
    assert request.scene_id == 1
    assert request.target_pose.header.frame_id == "world"
    assert request.target_name == "Sprite"
    assert request.ompl_waypoint_config == "right_route_to_pregrasp.yaml"
    assert request.trajectory_optimizer_mode == "none"
    assert request.use_preference_optimizer is False
    assert request.use_tcp_position_constraint is False
    assert request.gripper_command is False
    assert request.keep_grasp_obj is True
    assert request.adjust_grasp_mesh_min_z is True
    assert request.allow_grasp_mesh_z_lowering is True
    assert request.grasp_mesh_min_z_m == pytest.approx(0.232)
    assert request.max_grasp_mesh_z_adjustment_m == pytest.approx(0.05)
    mesh_input = request.grasp_object.mesh_input
    assert (
        mesh_input.axis_start_world.x,
        mesh_input.axis_start_world.y,
        mesh_input.axis_start_world.z,
    ) == pytest.approx(geometry.axis_start)
    assert (
        mesh_input.axis_end_world.x,
        mesh_input.axis_end_world.y,
        mesh_input.axis_end_world.z,
    ) == pytest.approx(geometry.axis_end)
    assert (
        mesh_input.grasp_point_world.x,
        mesh_input.grasp_point_world.y,
        mesh_input.grasp_point_world.z,
    ) == pytest.approx(geometry.grasp_point)


def test_phase_route_mapping_only_selects_global_ompl_steps() -> None:
    routes = ("pre.yaml", "place.yaml", "photo.yaml")

    assert _route_config_for_step("PRE_GRASP", *routes) == "pre.yaml"
    assert _route_config_for_step("PLACE_WAYPOINT", *routes) == "place.yaml"
    assert _route_config_for_step("RETURN_HOME", *routes) == "photo.yaml"
    assert _route_config_for_step("GRASP", *routes) == ""
    assert _route_config_for_step("PLACE_APPROACH", *routes) == ""


@pytest.mark.parametrize(
    ("arm_name", "task_profile", "capture_profile", "expected"),
    (
        ("right", "grasp", None, True),
        ("right", "pick", None, False),
        ("left", "grasp", None, False),
        ("left", "grasp", "full_flow_restock", True),
        ("right", "grasp", "full_flow_restock", False),
        ("right", "pick", "full_flow_pick", True),
        ("left", "pick", "full_flow_pick", False),
    ),
)
def test_phase_routes_are_scoped_to_their_intended_workflows(
    arm_name: str,
    task_profile: str,
    capture_profile: str | None,
    expected: bool,
) -> None:
    assert (
        _phase_ompl_routes_apply_to_task(arm_name, task_profile, capture_profile)
        is expected
    )


@pytest.mark.parametrize(
    ("arm_name", "capture_profile", "expected"),
    (
        ("right", "full_flow_pick", True),
        ("left", "full_flow_pick", False),
        ("right", "full_flow_restock", False),
        ("right", None, False),
    ),
)
def test_vertical_post_place_retreat_is_scoped_to_full_flow_right_pick(
    arm_name: str,
    capture_profile: str | None,
    expected: bool,
) -> None:
    assert (
        _vertical_post_place_retreat_required(arm_name, capture_profile) is expected
    )


def test_active_place_route_skips_cartesian_retreat_after_grasp() -> None:
    assert not _retreat_after_grasp_required(
        True,
        0.06,
        "right_route_to_place.yaml",
    )


def test_legacy_flows_keep_configured_retreat_after_grasp() -> None:
    assert _retreat_after_grasp_required(True, 0.06, "")
    assert not _retreat_after_grasp_required(False, 0.06, "")
    assert not _retreat_after_grasp_required(True, 0.0, "")


def test_active_photo_route_keeps_cartesian_retreat_after_place() -> None:
    assert _retreat_after_place_required(
        True,
        0.08,
        "right_route_to_photo.yaml",
        True,
    )


def test_legacy_flows_keep_configured_retreat_after_place() -> None:
    assert _retreat_after_place_required(True, 0.08, "", True)
    assert _retreat_after_place_required(
        True,
        0.08,
        "right_route_to_photo.yaml",
        False,
    )
    assert not _retreat_after_place_required(False, 0.08, "", True)
    assert not _retreat_after_place_required(True, 0.0, "", True)


def test_disabled_route_template_preserves_existing_planning(tmp_path: Path) -> None:
    route_path = tmp_path / "right_route.yaml"
    route_path.write_text(
        yaml.safe_dump(
            {
                "enabled": False,
                "arm_name": "right",
                "frame_id": "world",
                "waypoints": [],
            }
        ),
        encoding="utf-8",
    )

    assert _active_ompl_route_config(tmp_path, route_path.name, "right") == ""


def test_enabled_route_is_selected_only_for_the_matching_arm(tmp_path: Path) -> None:
    route_path = tmp_path / "right_route.yaml"
    route_path.write_text(
        yaml.safe_dump(
            {
                "enabled": True,
                "arm_name": "right",
                "frame_id": "world",
                "waypoints": [
                    {
                        "position": [0.1, -0.2, 0.3],
                        "orientation": [0.0, 0.0, 0.0, 1.0],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert (
        _active_ompl_route_config(tmp_path, route_path.name, "right")
        == route_path.name
    )
    with pytest.raises(ValueError, match="arm_name must be 'left'"):
        _active_ompl_route_config(tmp_path, route_path.name, "left")


def test_route_config_rejects_path_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="bare YAML filename"):
        _active_ompl_route_config(tmp_path, "../right_route.yaml", "right")


@pytest.mark.parametrize(
    "step",
    (
        "RETREAT_AFTER_GRASP",
        "PLACE_WAYPOINT",
        "PLACE_XY",
        "PLACE_ORIENT",
        "PLACE_APPROACH",
    ),
)
def test_place_transport_steps_keep_attached_grasp_geometry(step: str) -> None:
    assert _bridge_task_type_for_step(step) == "CARRY"


def test_place_transport_orients_before_horizontal_motion() -> None:
    assert _PLACE_TRANSPORT_STEP_ORDER == (
        "PLACE_LIFT",
        "PLACE_ORIENT",
        "PLACE_XY",
    )


def test_all_explicit_place_orientation_targets_receive_grasp_delta() -> None:
    assert _PLACE_ORIENTATION_TARGET_STEPS == {
        "PLACE_ORIENT",
        "PLACE_WAYPOINT",
        "PLACE_APPROACH",
    }


@pytest.mark.parametrize(
    ("step", "expected"),
    (
        ("OPEN", "MOVE"),
        ("PRE_GRASP", "MOVE"),
        ("GRASP_APPROACH", "MOVE"),
        ("RETREAT_AFTER_PLACE", "MOVE"),
        ("LIFT_AFTER_GRASP", "LIFT"),
        ("PLACE_LIFT", "LIFT"),
        ("GRASP", "GRASP"),
        ("PLACE", "PLACE"),
        ("RETURN_HOME", "RETURN_HOME"),
    ),
)
def test_other_steps_preserve_existing_bridge_semantics(
    step: str,
    expected: str,
) -> None:
    assert _bridge_task_type_for_step(step) == expected


def test_mesh_lifecycle_caches_at_approach_and_attaches_until_place_release() -> None:
    assert _mesh_request_semantics_for_step("PRE_GRASP") == (
        True,
        False,
        False,
    )
    assert _mesh_request_semantics_for_step("GRASP") == (True, True, False)
    assert _mesh_request_semantics_for_step("LIFT_AFTER_GRASP") == (
        False,
        True,
        False,
    )
    assert _mesh_request_semantics_for_step("PLACE") == (False, False, False)
    assert _mesh_request_semantics_for_step("RETREAT_AFTER_PLACE") == (
        False,
        False,
        False,
    )


def test_mesh_response_contract_transforms_all_points_with_capture_tf() -> None:
    geometry = _parse_mesh_geometry_payload(
        {
            "schema": "axis_grasp_input_v1",
            "valid": True,
            "frame_id": "cam_right_color_optical_frame",
            "units": "m",
            "target_name": "Sprite",
            "axis_start_xyz": [0.0, 0.0, 0.2],
            "axis_end_xyz": [0.0, 0.0, 0.4],
            "grasp_point_xyz": [0.01, -0.02, 0.3],
            "axis_direction": "bottom_to_top",
        },
        "cam_right_color_optical_frame",
        "Sprite",
    )
    world_from_camera = np.eye(4, dtype=np.float64)
    world_from_camera[:3, 3] = [0.2, -0.1, 0.05]

    world = _transform_mesh_geometry(
        geometry,
        "world",
        world_from_camera,
    )

    assert world.frame_id == "world"
    assert world.axis_start == pytest.approx((0.2, -0.1, 0.25))
    assert world.axis_end == pytest.approx((0.2, -0.1, 0.45))
    assert world.grasp_point == pytest.approx((0.21, -0.12, 0.35))


def test_mesh_response_contract_rejects_invalid_or_cross_frame_geometry() -> None:
    invalid = {
        "schema": "axis_grasp_input_v1",
        "valid": False,
        "reason": "object height is unavailable",
    }
    with pytest.raises(ValueError, match="object height is unavailable"):
        _parse_mesh_geometry_payload(
            invalid,
            "cam_right_color_optical_frame",
            "Sprite",
        )

    cross_frame = {
        "schema": "axis_grasp_input_v1",
        "valid": True,
        "frame_id": "right_base_link",
        "units": "m",
        "target_name": "Sprite",
        "axis_start_xyz": [0.0, 0.0, 0.2],
        "axis_end_xyz": [0.0, 0.0, 0.4],
        "grasp_point_xyz": [0.0, 0.0, 0.3],
        "axis_direction": "bottom_to_top",
    }
    with pytest.raises(ValueError, match="same frame"):
        _parse_mesh_geometry_payload(
            cross_frame,
            "cam_right_color_optical_frame",
            "Sprite",
        )


def test_mesh_cache_uses_the_tcp_candidate_that_will_actually_execute() -> None:
    geometry = MeshGeometry(
        frame_id="world",
        target_name="Sprite",
        axis_start=(0.2, -0.1, 0.1),
        axis_end=(0.2, -0.1, 0.3),
        grasp_point=(0.2, -0.1, 0.2),
    )
    pose = PoseStamped()
    pose.header.frame_id = "world"
    pose.pose.position.x = 0.21
    pose.pose.position.y = -0.12
    pose.pose.position.z = 0.23

    selected = _mesh_geometry_for_grasp_pose(geometry, pose)

    assert selected.axis_start == geometry.axis_start
    assert selected.axis_end == geometry.axis_end
    assert selected.grasp_point == pytest.approx((0.21, -0.12, 0.23))


def test_measured_mesh_axis_is_recorded_in_the_grasp_tcp_frame() -> None:
    pose = PoseStamped()
    pose.header.frame_id = "world"
    pose.pose.orientation.w = 1.0

    axis_tcp = _object_axis_in_tcp(pose, (0.0, 1.0, 0.0))

    assert axis_tcp == pytest.approx((0.0, 1.0, 0.0))


def test_return_home_step_has_no_gripper_command() -> None:
    initial_pose = PoseStamped()
    initial_pose.header.frame_id = "world"

    task_type, pose, gripper_command, gripper_opening = _return_home_step(
        initial_pose
    )

    assert task_type == "RETURN_HOME"
    assert pose is initial_pose
    assert not gripper_command
    assert gripper_opening == 0.0


def test_place_completion_retreats_before_returning_to_photo_position() -> None:
    final_place_pose = PoseStamped()
    final_place_pose.header.frame_id = "world"
    photo_pose = PoseStamped()
    photo_pose.header.frame_id = "world"

    steps = _place_completion_steps(
        final_place_pose,
        0.08,
        verify_upright_before_release=True,
        retreat_after_place=True,
        return_home_pose=photo_pose,
    )

    assert [step[0] for step in steps] == [
        "PLACE_APPROACH",
        "WAIT_PLACE_STABLE",
        "VERIFY_PLACE_UPRIGHT",
        "PLACE",
        "RETREAT_AFTER_PLACE",
        "RETURN_HOME",
    ]
    assert steps[0] == ("PLACE_APPROACH", final_place_pose, False, 0.0)
    assert steps[1] == ("WAIT_PLACE_STABLE", None, False, 0.0)
    assert steps[2] == ("VERIFY_PLACE_UPRIGHT", None, False, 0.0)
    assert steps[3] == ("PLACE", final_place_pose, True, 0.08)
    assert steps[4] == ("RETREAT_AFTER_PLACE", None, False, 0.0)
    assert steps[5] == ("RETURN_HOME", photo_pose, False, 0.0)


def test_slot_routed_scene_hands_retreat_off_after_clearing_bottle() -> None:
    required_scene_ids = {3}

    assert _handoff_post_place_retreat_to_return_route(
        3,
        "right_route_to_photo_scene_003_layer_1_position_6.yaml",
        True,
        required_scene_ids,
    )
    assert not _handoff_post_place_retreat_to_return_route(
        1,
        "right_route_to_photo.yaml",
        True,
        required_scene_ids,
    )
    assert not _handoff_post_place_retreat_to_return_route(
        3,
        "",
        True,
        required_scene_ids,
    )
    assert not _handoff_post_place_retreat_to_return_route(
        3,
        "right_route_to_photo_scene_003_layer_1_position_6.yaml",
        False,
        required_scene_ids,
    )


def test_place_completion_always_separates_motion_stability_and_release() -> None:
    final_place_pose = PoseStamped()

    steps = _place_completion_steps(
        final_place_pose,
        0.08,
        verify_upright_before_release=False,
        retreat_after_place=False,
        return_home_pose=None,
    )

    assert steps == [
        ("PLACE_APPROACH", final_place_pose, False, 0.0),
        ("WAIT_PLACE_STABLE", None, False, 0.0),
        ("PLACE", final_place_pose, True, 0.08),
    ]


def test_routed_place_releases_then_returns_without_cartesian_steps() -> None:
    final_place_pose = PoseStamped()
    photo_pose = PoseStamped()

    steps = _place_completion_steps(
        final_place_pose,
        0.08,
        verify_upright_before_release=True,
        retreat_after_place=False,
        return_home_pose=photo_pose,
        approach_before_place=False,
    )

    assert [step[0] for step in steps] == [
        "WAIT_PLACE_STABLE",
        "VERIFY_PLACE_UPRIGHT",
        "PLACE",
        "RETURN_HOME",
    ]
    assert "PLACE_APPROACH" not in {step[0] for step in steps}
    assert "RETREAT_AFTER_PLACE" not in {step[0] for step in steps}


def test_tcp_pose_delta_measures_translation_and_shortest_rotation() -> None:
    reference = PoseStamped()
    reference.pose.orientation.w = 1.0
    sample = PoseStamped()
    sample.pose.position.x = 0.0006
    sample.pose.position.y = 0.0008
    half_angle = np.deg2rad(0.4) / 2.0
    sample.pose.orientation.z = float(np.sin(half_angle))
    sample.pose.orientation.w = float(np.cos(half_angle))

    translation_m, rotation_deg = _tcp_pose_delta(reference, sample)

    assert translation_m == pytest.approx(0.001)
    assert rotation_deg == pytest.approx(0.4)


def test_dual_arm_capture_selects_each_configured_six_axis_target() -> None:
    restock = {
        "left": (-1.0, 0.5, -0.7, 0.2, 0.8, -0.4),
        "right": (1.0, 0.4, -0.8, -0.2, 0.9, 0.1),
    }
    pick = {"left": restock["left"], "right": None}
    full_flow_pick = {
        "left": None,
        "right": (0.8, 0.9, -1.4, -0.3, 0.9, 0.4),
    }

    assert _capture_joint_target_for_profile(
        "grasp", "right", restock, pick
    ) == restock["right"]
    assert _capture_joint_target_for_profile(
        "pick", "right", restock, pick
    ) is None
    assert _capture_joint_target_for_profile(
        "grasp", "left", restock, pick
    ) == restock["left"]
    assert _capture_joint_target_for_profile(
        "pick", "left", restock, pick
    ) == pick["left"]
    assert _capture_joint_target_for_profile(
        "pick",
        "right",
        restock,
        pick,
        full_flow_pick,
        capture_profile="full_flow_pick",
    ) == full_flow_pick["right"]


def test_full_flow_pick_uses_remote_orientation_without_changing_other_profiles() -> None:
    assert _grasp_orientation_mode_for_profile(
        "grasp",
        None,
        "remote",
        "capture_tcp",
        "remote",
    ) == "remote"
    assert _grasp_orientation_mode_for_profile(
        "pick",
        None,
        "remote",
        "capture_tcp",
        "remote",
    ) == "capture_tcp"
    assert _grasp_orientation_mode_for_profile(
        "pick",
        "full_flow_pick",
        "remote",
        "capture_tcp",
        "remote",
    ) == "remote"


def test_guarded_release_extracts_live_six_axis_joint_target() -> None:
    joint_state = JointState()
    joint_state.name = [
        "right_joint4",
        "left_joint1",
        "right_joint1",
        "right_joint6",
        "right_joint2",
        "right_joint5",
        "right_joint3",
        "right_joint7",
    ]
    joint_state.position = [0.4, -1.0, 0.1, 0.6, 0.2, 0.5, 0.3, 0.08]

    assert _arm_joint_positions_from_state(joint_state, "right") == pytest.approx(
        (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    )


def test_guarded_release_rejects_incomplete_joint_state() -> None:
    joint_state = JointState()
    joint_state.name = ["right_joint1"]
    joint_state.position = [0.1]

    with pytest.raises(RuntimeError, match="joint_state_missing:right_joint2"):
        _arm_joint_positions_from_state(joint_state, "right")


def test_restock_final_pose_is_raised_two_centimeters_before_waypoint() -> None:
    nominal_pose = PoseStamped()
    nominal_pose.header.frame_id = "world"
    nominal_pose.pose.position.x = 0.489322903785962
    nominal_pose.pose.position.y = -0.020938377004554136
    nominal_pose.pose.position.z = 0.30619623218684644
    nominal_pose.pose.orientation.x = -0.11241760487545066
    nominal_pose.pose.orientation.y = 0.6525535082516751
    nominal_pose.pose.orientation.z = 0.17339098968769825
    nominal_pose.pose.orientation.w = 0.7290211009824191

    final_pose = _elevated_place_waypoint(nominal_pose, 0.02)
    waypoint = _elevated_place_waypoint(final_pose, 0.08)

    assert nominal_pose.pose.position.z == pytest.approx(0.30619623218684644)
    assert final_pose.pose.position.z == pytest.approx(0.32619623218684644)
    assert waypoint.header.frame_id == "world"
    assert waypoint.pose.position.x == pytest.approx(final_pose.pose.position.x)
    assert waypoint.pose.position.y == pytest.approx(final_pose.pose.position.y)
    assert waypoint.pose.position.z == pytest.approx(0.40619623218684643)
    assert waypoint.pose.orientation == final_pose.pose.orientation


def test_post_place_retreat_uses_world_minus_x_and_holds_y_z_orientation() -> None:
    current_pose = PoseStamped()
    current_pose.header.frame_id = "world"
    current_pose.pose.position.x = 0.487780
    current_pose.pose.position.y = -0.009511
    current_pose.pose.position.z = 0.322724
    current_pose.pose.orientation.x = -0.1595
    current_pose.pose.orientation.y = 0.7286
    current_pose.pose.orientation.z = 0.1645
    current_pose.pose.orientation.w = 0.6455

    plan = _post_place_cartesian_retreat_plan(
        current_pose,
        slow_distance_m=0.05,
        min_total_distance_m=0.08,
        safe_world_x_m=0.3375,
        max_total_distance_m=0.22,
    )

    assert plan.direction_world == (-1.0, 0.0, 0.0)
    assert plan.slow_distance_m == pytest.approx(0.05)
    assert plan.total_distance_m == pytest.approx(0.150280, abs=1e-6)
    assert plan.slow_target.pose.position.x == pytest.approx(0.437780, abs=1e-6)
    assert plan.slow_target.pose.position.y == pytest.approx(-0.009511, abs=1e-9)
    assert plan.slow_target.pose.position.z == pytest.approx(0.322724, abs=1e-9)
    assert plan.slow_target.pose.orientation == current_pose.pose.orientation
    assert plan.final_target.pose.position.x == pytest.approx(0.3375, abs=1e-9)
    assert plan.final_target.pose.position.y == pytest.approx(-0.009511, abs=1e-9)
    assert plan.final_target.pose.position.z == pytest.approx(0.322724, abs=1e-9)
    assert plan.final_target.pose.orientation == current_pose.pose.orientation


def test_post_place_retreat_rejects_distance_beyond_configured_maximum() -> None:
    current_pose = PoseStamped()
    current_pose.header.frame_id = "world"
    current_pose.pose.position.x = 0.58
    current_pose.pose.position.z = 0.32
    current_pose.pose.orientation.w = 1.0

    with pytest.raises(ValueError, match="exceeds configured maximum"):
        _post_place_cartesian_retreat_plan(
            current_pose,
            slow_distance_m=0.05,
            min_total_distance_m=0.08,
            safe_world_x_m=0.3375,
            max_total_distance_m=0.22,
        )


def test_slot_return_handoff_only_plans_slow_bottle_clearance() -> None:
    current_pose = PoseStamped()
    current_pose.header.frame_id = "world"
    current_pose.pose.position.x = 0.603468
    current_pose.pose.position.y = -0.1578
    current_pose.pose.position.z = 0.2304
    current_pose.pose.orientation.w = 1.0

    plan = _post_place_cartesian_retreat_plan(
        current_pose,
        slow_distance_m=0.05,
        min_total_distance_m=0.08,
        safe_world_x_m=0.3375,
        max_total_distance_m=0.22,
        handoff_after_slow=True,
    )

    assert plan.slow_distance_m == pytest.approx(0.05)
    assert plan.total_distance_m == pytest.approx(0.05)
    assert plan.slow_target.pose.position.x == pytest.approx(0.553468)
    assert plan.final_target.pose.position.x == pytest.approx(0.553468)


def test_cartesian_retreat_converts_tcp_target_to_link6_target() -> None:
    tcp_pose = PoseStamped()
    tcp_pose.header.frame_id = "world"
    tcp_pose.header.stamp.sec = 12
    tcp_pose.header.stamp.nanosec = 345
    tcp_pose.pose.position.x = 0.50
    tcp_pose.pose.position.y = -0.02
    tcp_pose.pose.position.z = 0.33
    tcp_pose.pose.orientation.y = math.sin(math.pi / 8.0)
    tcp_pose.pose.orientation.w = math.cos(math.pi / 8.0)

    link6_pose = _link6_pose_from_tcp_pose(tcp_pose, 0.1358)
    tcp_from_link6 = np.eye(4, dtype=np.float64)
    tcp_from_link6[2, 3] = -0.1358

    assert link6_pose.header == tcp_pose.header
    assert np.allclose(
        _matrix_from_pose(link6_pose),
        _matrix_from_pose(tcp_pose) @ tcp_from_link6,
        atol=1.0e-12,
    )


class _NullLogger:
    def info(self, _message: str) -> None:
        pass


class _PoseHelper:
    def get_logger(self) -> _NullLogger:
        return _NullLogger()

    def _copy_pose(self, pose: PoseStamped) -> PoseStamped:
        return GraspBridgeStateMachine._copy_pose(self, pose)


def test_retracted_pose_moves_against_approach_axis() -> None:
    grasp_pose = PoseStamped()
    grasp_pose.pose.position.x = 0.50
    grasp_pose.pose.position.y = -0.20
    grasp_pose.pose.position.z = 0.40

    retreat = GraspBridgeStateMachine._retracted_pose_from_axis(
        _PoseHelper(),
        grasp_pose,
        np.asarray([0.0, 1.0, 0.0]),
        0.06,
        log_label="test_retreat",
        min_z_m=0.0,
    )

    assert retreat.pose.position.x == pytest.approx(0.50)
    assert retreat.pose.position.y == pytest.approx(-0.26)
    assert retreat.pose.position.z == pytest.approx(0.40)


def _arbitrary_grasp_pose() -> PoseStamped:
    pose = PoseStamped()
    pose.header.frame_id = "world"
    pose.pose.position.x = 0.41
    pose.pose.position.y = -0.17
    pose.pose.position.z = 0.36
    quaternion = np.asarray([0.2, -0.3, 0.1, 0.9], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    pose.pose.orientation.x = float(quaternion[0])
    pose.pose.orientation.y = float(quaternion[1])
    pose.pose.orientation.z = float(quaternion[2])
    pose.pose.orientation.w = float(quaternion[3])
    return pose


def test_local_y_fallback_keeps_opening_axis_and_changes_approach_axis() -> None:
    original = _arbitrary_grasp_pose()
    rotated = _pose_with_local_orientation_offset(original, "y", 5.0)
    original_matrix = _matrix_from_pose(original)
    rotated_matrix = _matrix_from_pose(rotated)

    assert rotated_matrix[:3, 3] == pytest.approx(original_matrix[:3, 3])
    assert rotated_matrix[:3, 1] == pytest.approx(original_matrix[:3, 1])
    assert float(np.dot(rotated_matrix[:3, 2], original_matrix[:3, 2])) == pytest.approx(
        np.cos(np.deg2rad(5.0))
    )


def test_local_z_fallback_rotates_opening_axis_with_full_tcp_pose() -> None:
    original = _arbitrary_grasp_pose()
    rotated = _pose_with_local_orientation_offset(original, "z", -5.0)
    original_matrix = _matrix_from_pose(original)
    rotated_matrix = _matrix_from_pose(rotated)

    assert rotated_matrix[:3, 3] == pytest.approx(original_matrix[:3, 3])
    assert rotated_matrix[:3, 2] == pytest.approx(original_matrix[:3, 2])
    assert float(np.dot(rotated_matrix[:3, 1], original_matrix[:3, 1])) == pytest.approx(
        np.cos(np.deg2rad(5.0))
    )
    assert not np.allclose(rotated_matrix[:3, 1], original_matrix[:3, 1])
    rotated_quaternion = np.asarray(
        [
            rotated.pose.orientation.x,
            rotated.pose.orientation.y,
            rotated.pose.orientation.z,
            rotated.pose.orientation.w,
        ]
    )
    assert np.linalg.norm(rotated_quaternion) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("local_axis", "degrees"),
    (("y", 5.0), ("z", -3.0)),
)
def test_place_compensation_restores_nominal_object_transform(
    local_axis: str,
    degrees: float,
) -> None:
    original_grasp = _arbitrary_grasp_pose()
    selected_grasp = _pose_with_local_orientation_offset(
        original_grasp,
        local_axis,
        degrees,
    )
    nominal_place = _arbitrary_grasp_pose()
    nominal_place.pose.position.x = 0.52
    nominal_place.pose.position.y = 0.08
    nominal_place.pose.position.z = 0.30
    place_quaternion = np.asarray(
        [0.0163, 0.7252, -0.0906, 0.6824],
        dtype=np.float64,
    )
    place_quaternion /= np.linalg.norm(place_quaternion)
    nominal_place.pose.orientation.x = float(place_quaternion[0])
    nominal_place.pose.orientation.y = float(place_quaternion[1])
    nominal_place.pose.orientation.z = float(place_quaternion[2])
    nominal_place.pose.orientation.w = float(place_quaternion[3])

    compensated_place = _place_pose_with_grasp_orientation_compensation(
        nominal_place,
        original_grasp,
        selected_grasp,
    )

    object_at_grasp = np.eye(4, dtype=np.float64)
    object_at_grasp[:3, 3] = np.asarray([0.31, -0.02, 0.08])
    nominal_object_at_place = (
        _matrix_from_pose(nominal_place)
        @ np.linalg.inv(_matrix_from_pose(original_grasp))
        @ object_at_grasp
    )
    compensated_object_at_place = (
        _matrix_from_pose(compensated_place)
        @ np.linalg.inv(_matrix_from_pose(selected_grasp))
        @ object_at_grasp
    )

    assert compensated_place.pose.position == nominal_place.pose.position
    assert compensated_object_at_place == pytest.approx(nominal_object_at_place)


def _pose_with_quaternion(quaternion: tuple[float, float, float, float]) -> PoseStamped:
    pose = PoseStamped()
    pose.header.frame_id = "world"
    pose.pose.position.x = 0.52
    pose.pose.position.y = 0.08
    pose.pose.position.z = 0.30
    pose.pose.orientation.x = quaternion[0]
    pose.pose.orientation.y = quaternion[1]
    pose.pose.orientation.z = quaternion[2]
    pose.pose.orientation.w = quaternion[3]
    return pose


def test_upright_compensation_corrects_latest_original_grasp_case() -> None:
    actual_grasp_tcp = _pose_with_quaternion(
        (-0.3493, 0.7510, 0.2134, 0.5180)
    )
    nominal_place = _pose_with_quaternion(
        (-0.1124, 0.6526, 0.1734, 0.7290)
    )
    object_axis_tcp = _upright_object_axis_in_tcp(actual_grasp_tcp)

    compensated_place, correction_degrees = (
        _place_pose_with_upright_axis_compensation(
            nominal_place,
            object_axis_tcp,
        )
    )

    assert _object_upright_tilt_degrees(
        nominal_place,
        object_axis_tcp,
    ) == pytest.approx(29.54, abs=0.05)
    assert correction_degrees == pytest.approx(29.54, abs=0.05)
    assert _object_upright_tilt_degrees(
        compensated_place,
        object_axis_tcp,
    ) == pytest.approx(0.0, abs=1e-6)
    assert compensated_place.pose.position == nominal_place.pose.position

    nominal_rotation = _matrix_from_pose(nominal_place)[:3, :3]
    compensated_rotation = _matrix_from_pose(compensated_place)[:3, :3]
    world_correction = compensated_rotation @ nominal_rotation.T
    # The full basis is transformed, including the TCP opening direction Y.
    assert compensated_rotation[:, 1] == pytest.approx(
        world_correction @ nominal_rotation[:, 1]
    )
    assert not np.allclose(compensated_rotation[:, 1], nominal_rotation[:, 1])


def test_upright_compensation_disambiguates_opposite_pca_axis_sign() -> None:
    """Regression for the position-4 159.865-degree PLACE_WAYPOINT failure."""
    nominal_place = _pose_with_quaternion(
        (-0.0887, 0.7610, 0.0698, 0.6388)
    )
    nominal_place.pose.position.x = 0.6062
    nominal_place.pose.position.y = -0.1615
    nominal_place.pose.position.z = 0.219082
    reported_axis_tcp = np.asarray(
        [0.9844777494, -0.0002184551, -0.1755092965],
        dtype=np.float64,
    )

    reported_result, reported_correction = (
        _place_pose_with_upright_axis_compensation(
            nominal_place,
            reported_axis_tcp,
        )
    )
    reversed_result, reversed_correction = (
        _place_pose_with_upright_axis_compensation(
            nominal_place,
            -reported_axis_tcp,
        )
    )

    assert reported_correction == pytest.approx(20.13694, abs=1e-4)
    assert reversed_correction == pytest.approx(reported_correction, abs=1e-9)
    assert _matrix_from_pose(reported_result) == pytest.approx(
        _matrix_from_pose(reversed_result),
        abs=1e-9,
    )
    assert _object_upright_tilt_degrees(
        reported_result,
        reported_axis_tcp,
    ) == pytest.approx(0.0, abs=1e-6)

    link6_target = _link6_pose_from_tcp_pose(reported_result, 0.1358)
    assert link6_target.pose.position.x == pytest.approx(0.475874, abs=1e-5)

    limited_reported, limited_reported_correction = (
        _place_pose_with_limited_upright_axis_compensation(
            nominal_place,
            reported_axis_tcp,
            15.0,
        )
    )
    limited_reversed, limited_reversed_correction = (
        _place_pose_with_limited_upright_axis_compensation(
            nominal_place,
            -reported_axis_tcp,
            15.0,
        )
    )
    assert limited_reported_correction == pytest.approx(15.0)
    assert limited_reversed_correction == pytest.approx(15.0)
    assert _matrix_from_pose(limited_reported) == pytest.approx(
        _matrix_from_pose(limited_reversed),
        abs=1e-9,
    )


@pytest.mark.parametrize(
    ("local_axis", "degrees"),
    (("y", 0.0), ("y", 5.0), ("z", -5.0)),
)
def test_upright_compensation_levels_original_and_fallback_grasps(
    local_axis: str,
    degrees: float,
) -> None:
    actual_grasp_tcp = _arbitrary_grasp_pose()
    if degrees:
        actual_grasp_tcp = _pose_with_local_orientation_offset(
            actual_grasp_tcp,
            local_axis,
            degrees,
        )
    nominal_place = _pose_with_quaternion(
        (0.0163, 0.7252, -0.0906, 0.6824)
    )
    object_axis_tcp = _upright_object_axis_in_tcp(actual_grasp_tcp)

    compensated_place, _ = _place_pose_with_upright_axis_compensation(
        nominal_place,
        object_axis_tcp,
    )

    assert _object_upright_tilt_degrees(
        compensated_place,
        object_axis_tcp,
    ) == pytest.approx(0.0, abs=1e-6)


def test_vector_alignment_handles_opposite_vertical_axes() -> None:
    rotation = _rotation_aligning_vectors([0.0, 0.0, -1.0], [0.0, 0.0, 1.0])

    assert rotation @ np.asarray([0.0, 0.0, -1.0]) == pytest.approx(
        [0.0, 0.0, 1.0]
    )
    assert rotation @ rotation.T == pytest.approx(np.eye(3))
    assert np.linalg.det(rotation) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("approach_axis", "expected_deviation_deg"),
    (
        ((1.0, 0.0, 0.0), 0.0),
        ((1.0, 0.0, -1.0), 45.0),
        ((0.0, 0.0, -1.0), 90.0),
    ),
)
def test_side_grasp_deviation_is_measured_from_the_horizontal_plane(
    approach_axis: tuple[float, float, float],
    expected_deviation_deg: float,
) -> None:
    assert _grasp_approach_vertical_deviation_degrees(
        approach_axis
    ) == pytest.approx(expected_deviation_deg)


def test_upright_target_wildcard_selects_all_current_and_future_products() -> None:
    selected = {_normalized_target_key("*")}

    assert _upright_target_is_selected("AD Calcium Milk", selected)
    assert _upright_target_is_selected("Aojiru", selected)
    assert _upright_target_is_selected("Future Upright Product", selected)


def test_explicit_upright_target_selection_remains_case_and_space_insensitive() -> None:
    selected = {_normalized_target_key("HK Orange Fanta")}

    assert _upright_target_is_selected("  HK   ORANGE fanta ", selected)
    assert not _upright_target_is_selected("AD Calcium Milk", selected)


def test_scene_scoped_upright_compensation_can_override_side_grasp_gate() -> None:
    assert _upright_place_compensation_is_active(True, False, 3, {3})
    assert not _upright_place_compensation_is_active(True, False, 1, {3})
    assert _upright_place_compensation_is_active(True, True, 1, {3})
    assert not _upright_place_compensation_is_active(False, True, 3, {3})


def test_upright_place_compensation_is_enabled_for_right_grasp_flow() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "grasp_bridge_state_machine.right.yaml"
    )
    parameters = yaml.safe_load(config_path.read_text(encoding="utf-8"))[
        "grasp_bridge_state_machine"
    ]["ros__parameters"]

    assert parameters["compensate_place_orientation_from_grasp_delta"] is True
    assert parameters["place_upright_axis_compensation_enabled"] is True
    assert parameters["place_upright_axis_compensation_force_scene_ids"] == [3]
    assert parameters["place_upright_axis_compensation_profiles"] == ["grasp"]
    assert parameters["place_upright_axis_compensation_arms"] == ["left", "right"]
    assert parameters["place_upright_axis_compensation_targets"] == ["*"]
    assert parameters[
        "place_upright_side_grasp_max_vertical_deviation_deg"
    ] == pytest.approx(45.0)
    assert parameters["place_upright_tilt_limit_deg"] == pytest.approx(4.0)


def test_ik_fallback_candidate_order_is_original_then_y_then_z() -> None:
    candidates = _build_ik_orientation_candidates(
        _arbitrary_grasp_pose(),
        [3.0, -3.0, 5.0, -5.0, 8.0, -8.0],
        [3.0, -3.0, 5.0, -5.0],
    )

    assert [candidate.name for candidate in candidates] == [
        "original",
        "local_y_+3deg",
        "local_y_-3deg",
        "local_y_+5deg",
        "local_y_-5deg",
        "local_y_+8deg",
        "local_y_-8deg",
        "local_z_+3deg",
        "local_z_-3deg",
        "local_z_+5deg",
        "local_z_-5deg",
    ]


def test_grasp_candidate_lock_is_reused_by_coupled_steps() -> None:
    assert _candidate_indexes_for_step("PRE_GRASP", 11, None) == tuple(range(11))
    assert _candidate_indexes_for_step("GRASP", 11, 4) == (4,)
    assert _candidate_indexes_for_step("RETREAT_AFTER_GRASP", 11, 4) == (4,)
    assert _candidate_indexes_for_step("PLACE", 1, 4) == (0,)


def test_full_approach_selection_retries_locked_candidate_first() -> None:
    assert _candidate_indexes_for_step(
        "GRASP",
        5,
        2,
        reselect_locked_for_full_approach=True,
    ) == (2, 0, 1, 3, 4)
    assert _candidate_indexes_for_step(
        "PRE_GRASP",
        5,
        2,
        reselect_locked_for_full_approach=True,
    ) == (2,)


def test_full_flow_grasp_shallow_offset_is_isolated_from_existing_flows() -> None:
    assert _grasp_shallow_offset_for_profile("full_flow_pick", 0.002, 0.0) == 0.0
    assert _grasp_shallow_offset_for_profile("grasp", 0.002, 0.0) == pytest.approx(
        0.002
    )
    assert _grasp_shallow_offset_for_profile(None, 0.002, 0.0) == pytest.approx(
        0.002
    )

def test_aojiru_grasp_moves_five_mm_forward_from_the_existing_target() -> None:
    assert _grasp_retraction_offset_for_target("Aojiru", 0.002) == pytest.approx(
        -0.003
    )
    assert _grasp_retraction_offset_for_target(" aojiru ", 0.0) == pytest.approx(-0.005)
    assert _grasp_retraction_offset_for_target("Sprite", 0.002) == pytest.approx(0.002)


def test_left_aojiru_grasp_adds_another_five_mm_forward() -> None:
    assert _grasp_retraction_offset_for_target(
        "Aojiru",
        0.002,
        arm_name="left",
    ) == pytest.approx(-0.008)
    assert _grasp_retraction_offset_for_target(
        "Aojiru",
        0.002,
        arm_name="right",
    ) == pytest.approx(-0.003)



def test_only_explicit_no_ik_detail_matches_ik_fallback_condition() -> None:
    assert _is_no_ik_failure("Target pose has no IK solution")
    assert _is_no_ik_failure("PLANNING FAILED: target pose has NO IK SOLUTION")
    assert not _is_no_ik_failure("Planning failed due to collision")
    assert not _is_no_ik_failure("Trajectory execution failed")


def test_incomplete_cartesian_path_is_retryable_planning_failure() -> None:
    from grasp_bridge_state_machine import _is_planning_failure

    assert _is_planning_failure("Cartesian path incomplete: fraction=0.750000")
    assert _is_planning_failure("Planning failed due to collision")
    assert not _is_planning_failure("Trajectory execution aborted")


def test_ik_fallback_angles_are_bounded_and_deduplicated() -> None:
    assert _validated_ik_fallback_degrees(
        [3.0, -3.0, 3.0],
        "angles",
    ) == (3.0, -3.0)
    assert _validated_ik_fallback_degrees([30.0], "angles") == (30.0,)
    with pytest.raises(ValueError, match="must not exceed"):
        _validated_ik_fallback_degrees([31.0], "angles")


def test_dual_arm_config_uses_shared_grasp_flow_and_capture_targets() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "grasp_bridge_state_machine.right.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    parameters = config["grasp_bridge_state_machine"]["ros__parameters"]

    assert parameters["pre_grasp_offset_m"] == pytest.approx(0.08)
    assert parameters["grasp_shallow_offset_m"] == pytest.approx(0.002)
    assert parameters["full_flow_pick_grasp_shallow_offset_m"] == pytest.approx(0.0)
    assert "rebase_right_full_flow_pick_121_gripper_end_to_planning_tcp" not in parameters
    assert parameters["hybrid_cartesian_motion_enabled"] is True
    assert parameters["hybrid_cartesian_eef_step_m"] == pytest.approx(0.005)
    assert parameters["hybrid_cartesian_velocity_scaling"] == pytest.approx(0.10)
    assert parameters["hybrid_cartesian_acceleration_scaling"] == pytest.approx(0.10)
    assert parameters["retreat_after_grasp"] is True
    assert parameters["retreat_after_grasp_offset_m"] == pytest.approx(0.06)
    assert parameters["lift_after_grasp"] is False
    assert parameters["place_lift_before_place"] is False
    assert "place_lift_height_m" not in parameters
    assert parameters["restock_place_final_z_offset_m"] == pytest.approx(0.01)
    assert parameters["restock_place_dynamic_mesh_z_enabled"] is True
    assert parameters["restock_place_support_surface_z_m"] == pytest.approx(0.23)
    assert parameters["restock_place_mesh_clearance_m"] == pytest.approx(0.002)
    assert parameters["restock_place_mesh_clearances_by_scene"] == ["3=0.008"]
    assert parameters[
        "restock_place_max_dynamic_z_adjustment_m"
    ] == pytest.approx(0.05)
    assert parameters["place_descend_offset_m"] == pytest.approx(0.08)
    assert parameters["place_release_stability_window_s"] == pytest.approx(0.4)
    assert parameters["place_release_stability_timeout_s"] == pytest.approx(3.0)
    assert parameters["place_release_stability_sample_period_s"] == pytest.approx(0.05)
    assert parameters["place_release_stability_position_tolerance_m"] == pytest.approx(
        0.001
    )
    assert parameters[
        "place_release_stability_orientation_tolerance_deg"
    ] == pytest.approx(0.5)
    assert parameters["retreat_after_place"] is True
    assert parameters["retreat_after_place_offset_m"] == pytest.approx(0.08)
    assert parameters["retreat_after_place_release_settle_s"] == pytest.approx(0.2)
    assert parameters["retreat_after_place_slow_distance_m"] == pytest.approx(0.05)
    assert parameters["retreat_after_place_safe_world_x_m"] == pytest.approx(0.3375)
    assert parameters["retreat_after_place_max_distance_m"] == pytest.approx(0.22)
    assert "retreat_after_place_min_axis_x_component" not in parameters
    assert parameters["retreat_after_place_eef_step_m"] == pytest.approx(0.005)
    assert parameters["retreat_after_place_slow_velocity_scaling"] == pytest.approx(0.15)
    assert parameters["retreat_after_place_fast_velocity_scaling"] == pytest.approx(0.35)
    assert parameters["planning_retry_attempts"] == 2
    assert parameters["joint_plan_service_name"] == "/plan_to_joints"
    assert parameters["cartesian_plan_service_name"] == "/plan_cartesian_path"
    assert parameters["pose_v3_service_name"] == "/plan_to_pose_v3"
    assert parameters["route_to_pregrasp_config"] == (
        "right_route_to_pregrasp.yaml"
    )
    assert parameters["route_to_place_config"] == "right_route_to_place.yaml"
    assert parameters["route_to_photo_config"] == "right_route_to_photo.yaml"
    assert parameters["full_flow_restock_route_to_place_config"] == (
        "left_full_flow_route_to_place.yaml"
    )
    assert parameters["full_flow_pick_route_to_place_config"] == (
        "right_full_flow_pick_route_to_place.yaml"
    )
    assert parameters["route_to_place_scene_overrides"] == [
        "3=right_route_to_place_scene_003.yaml"
    ]
    assert parameters["full_flow_restock_route_to_place_scene_overrides"] == [
        "3=left_full_flow_route_to_place_scene_003.yaml"
    ]
    assert parameters["full_flow_pick_route_to_place_scene_overrides"] == [
        "3=right_full_flow_pick_route_to_place_scene_003.yaml"
    ]
    assert parameters["restock_place_support_surfaces_by_scene"] == [
        "3=0.13,0.46,0.80"
    ]
    assert parameters["route_trajectory_optimizer_mode"] == "none"
    assert parameters["route_use_preference_optimizer"] is False
    assert parameters["route_use_tcp_position_constraint"] is False
    assert parameters["grasp_object_type"] == "mesh"
    assert parameters["right_capture_joint_enabled"] is True
    assert parameters["right_capture_joint_positions"] == pytest.approx([
        1.096756612,
        0.376842732,
        -0.773397184,
        -0.213252900,
        0.928927888,
        0.106338624,
    ])
    assert parameters["left_capture_joint_enabled"] is True
    assert parameters["left_capture_joint_positions"] == pytest.approx([
        -1.148251300,
        0.480582200,
        -0.704144504,
        0.253827644,
        0.853500032,
        -0.448258468,
    ])
    assert parameters["pick_left_capture_joint_enabled"] is True
    assert parameters["pick_left_capture_joint_positions"] == pytest.approx(
        parameters["left_capture_joint_positions"]
    )
    assert parameters["full_flow_pick_right_capture_joint_enabled"] is True
    assert parameters["full_flow_pick_right_capture_joint_positions"] == pytest.approx([
        0.872025560,
        0.909094060,
        -1.470564088,
        -0.300158908,
        0.919176692,
        0.449549324,
    ])
    assert parameters["grasp_orientation_mode"] == "remote"
    assert parameters["pick_grasp_orientation_mode"] == "capture_tcp"
    assert parameters["full_flow_pick_grasp_orientation_mode"] == "remote"
    assert parameters["left_camera_frame"] == "cam_left_color_optical_frame"
    assert parameters["right_camera_frame"] == "cam_right_color_optical_frame"
    assert parameters["grasp_ik_fallback_enabled"] is True
    assert parameters["grasp_ik_fallback_profiles"] == ["grasp"]
    assert parameters["grasp_ik_fallback_relaxed_scene_ids"] == [3]
    assert parameters["grasp_ik_fallback_relaxed_local_y_degrees"] == pytest.approx(
        [
            3.0,
            -3.0,
            5.0,
            -5.0,
            8.0,
            10.0,
            12.0,
            15.0,
            18.0,
            20.0,
            25.0,
            30.0,
            -8.0,
            -10.0,
            -12.0,
            -15.0,
            -18.0,
            -20.0,
            -25.0,
            -30.0,
        ]
    )
    assert parameters["grasp_ik_fallback_local_y_degrees"] == pytest.approx(
        [3.0, -3.0, 5.0, -5.0, 8.0, -8.0]
    )
    assert parameters["grasp_ik_fallback_local_z_degrees"] == pytest.approx(
        [3.0, -3.0, 5.0, -5.0]
    )


def test_simple_place_transport_uses_one_eight_cm_waypoint() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "config"
    parameters = yaml.safe_load(
        (config_dir / "grasp_bridge_state_machine.right.yaml").read_text(
            encoding="utf-8"
        )
    )["grasp_bridge_state_machine"]["ros__parameters"]
    assert parameters["lift_after_grasp"] is False
    assert parameters["place_lift_before_place"] is False
    assert parameters["place_descend_before_place"] is True
    assert parameters["place_descend_offset_m"] == pytest.approx(0.08)


def test_phase_route_configs_are_safe_and_calibrated_as_expected() -> None:
    route_dir = Path(__file__).resolve().parents[1] / "config" / "ompl_waypoints"
    route_expectations = {
        "right_route_to_pregrasp.yaml": (False, "right", 1),
        "right_route_to_place.yaml": (True, "right", 2),
        "right_route_to_photo.yaml": (True, "right", 2),
        "left_full_flow_route_to_place.yaml": (True, "left", 2),
        "right_full_flow_pick_route_to_place.yaml": (True, "right", 2),
    }
    for file_name, (expected_enabled, expected_arm, expected_waypoint_count) in (
        route_expectations.items()
    ):
        route = yaml.safe_load((route_dir / file_name).read_text(encoding="utf-8"))
        assert route["enabled"] is expected_enabled
        assert route["arm_name"] == expected_arm
        assert route["frame_id"] == "world"
        assert len(route["waypoints"]) == expected_waypoint_count
        for waypoint in route["waypoints"]:
            assert len(waypoint["position"]) == 3
            assert len(waypoint["orientation"]) == 4
            assert all(
                math.isfinite(float(value)) for value in waypoint["position"]
            )
            assert all(
                math.isfinite(float(value)) for value in waypoint["orientation"]
            )
            assert math.sqrt(
                sum(float(value) ** 2 for value in waypoint["orientation"])
            ) == pytest.approx(1.0, abs=1.0e-3)


def test_full_flow_place_routes_use_mirrored_and_reversed_corridors() -> None:
    route_dir = Path(__file__).resolve().parents[1] / "config" / "ompl_waypoints"
    left = yaml.safe_load(
        (route_dir / "left_full_flow_route_to_place.yaml").read_text(encoding="utf-8")
    )
    right = yaml.safe_load(
        (route_dir / "right_full_flow_pick_route_to_place.yaml").read_text(encoding="utf-8")
    )

    assert [waypoint["position"] for waypoint in left["waypoints"]] == [
        [0.171, 0.110, 0.384],
        [0.350, 0.085, 0.424],
    ]
    assert [waypoint["position"] for waypoint in right["waypoints"]] == [
        [0.350, -0.085, 0.424],
        [0.171, -0.110, 0.384],
    ]
    assert [waypoint["orientation"] for waypoint in left["waypoints"]] == [
        [0.315, 0.729, -0.252, 0.553],
        [0.258, 0.751, -0.115, 0.597],
    ]
    assert [waypoint["orientation"] for waypoint in right["waypoints"]] == [
        [-0.258, 0.751, 0.115, 0.597],
        [-0.315, 0.729, 0.252, 0.553],
    ]
