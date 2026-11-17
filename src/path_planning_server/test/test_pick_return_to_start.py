"""Offline workflow tests: all planner, inference and hardware calls are mocked."""
import copy
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import rclpy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

from grasp_bridge_state_machine import (
    ExecuteNamedGraspTask, GraspBridgeStateMachine, MeshGeometry,
    _pick_returns_to_start,
)


@pytest.mark.parametrize("enabled,arm,profile,capture,expected", [
    (True, "right", "pick", None, True),
    (False, "right", "pick", None, False),
    (True, "left", "pick", None, False),
    (True, "right", "grasp", None, False),
    (True, "right", "pick", "full_flow_pick", False),
])
def test_workflow_scope(enabled, arm, profile, capture, expected):
    assert _pick_returns_to_start(enabled, arm, profile, capture) is expected


@pytest.fixture
def workflow():
    # Never join the hardware's ROS domain, even when these tests run on robot.
    os.environ["ROS_DOMAIN_ID"] = "199"
    config = Path(__file__).resolve().parent.parent / "config/grasp_bridge_state_machine.right.yaml"
    rclpy.init(args=["--ros-args", "--params-file", str(config)])
    machine = GraspBridgeStateMachine()
    calls = []
    start = PoseStamped()
    start.header.frame_id = "world"
    start.pose.position.x = 0.2
    start.pose.position.z = 0.5
    start.pose.orientation.w = 1.0
    state = JointState()
    state.name = [f"right_joint{i}" for i in range(1, 7)]
    state.position = [0.1, 0.2, -0.3, 0.4, 0.5, -0.6]
    machine._store_joint_state(state)
    machine._current_tcp_pose_world = lambda arm: copy.deepcopy(start)
    machine._clear_target_markers = Mock()
    machine._publish_target_markers = Mock()
    machine._publish_collection_event = Mock()
    machine._sleep_interruptibly = lambda seconds: True
    machine._call_v3_route_plan = Mock(side_effect=AssertionError("unexpected route"))

    def motion(kind, **kwargs):
        calls.append((kind, kwargs))
        if kind == "joint":
            state.position = list(kwargs["joint_positions"])
            machine._store_joint_state(state)
        return True, "mock completed"

    machine._call_joint_plan = lambda **kw: motion("joint", **kw)
    machine._call_bridge_task = lambda **kw: motion("bridge", **kw)
    machine._call_cartesian_plan = lambda **kw: motion("cartesian", **kw)
    bundle = SimpleNamespace(rgb=None, depth=None, info=SimpleNamespace(width=640, height=480))
    machine._capture_sensor_snapshot = lambda *a, **kw: (bundle, {"world_from_camera": np.eye(4)})
    target = copy.deepcopy(start)
    target.pose.position.x = 0.4
    target.header.frame_id = "cam_right_color_optical_frame"
    async def inference(*args):
        return SimpleNamespace(tcp_pose=copy.deepcopy(target), mesh_geometry=MeshGeometry(
            frame_id=target.header.frame_id, target_name="Sprite",
            axis_start=(0.4, 0.0, 0.4), axis_end=(0.4, 0.0, 0.6),
            grasp_point=(0.4, 0.0, 0.5),
        ))
    machine._request_remote_grasp = inference
    def to_world(arm, pose, matrix):
        result = copy.deepcopy(pose)
        result.header.frame_id = "world"
        return result
    machine._camera_tcp_pose_to_world_pose_at_capture = to_world
    request = ExecuteNamedGraspTask.Request()
    request.arm_name = "right"
    request.target_name = "Sprite"
    request.scene_id = 1
    try:
        yield machine, request, calls, machine._configured_pick_capture_joints["right"]
    finally:
        machine.destroy_node()
        rclpy.shutdown()


def test_pick_visits_fixed_start_before_inference_and_returns_without_release(workflow):
    machine, request, calls, initial_joints = workflow
    success, detail = machine._execute_named(request, task_profile="pick")
    assert success, detail
    assert [kind for kind, _ in calls] == ["joint", "joint", "bridge", "cartesian", "joint", "cartesian", "joint"]
    moved_to_start, opened, pregrasp, grasp, closed, retreat, returned = [kw for _, kw in calls]
    assert tuple(moved_to_start["joint_positions"]) == initial_joints
    assert not moved_to_start.get("gripper_command", False)
    assert tuple(opened["joint_positions"]) == initial_joints
    assert opened["gripper_command"] and opened["gripper_opening_m"] > 0
    assert closed["gripper_command"] and closed["gripper_opening_m"] == 0
    assert returned["keep_grasp_ellipsoid"] is True
    assert not returned.get("gripper_command", False)
    assert tuple(returned["joint_positions"]) == initial_joints
    a, b = grasp["link6_pose"].pose.position, retreat["link6_pose"].pose.position
    assert np.linalg.norm([a.x-b.x, a.y-b.y, a.z-b.z]) >= 0.0799
    assert "gripper closed" in detail


def test_stale_start_feedback_stops_before_any_command(workflow):
    machine, request, calls, _ = workflow
    machine._latest_joint_state_received_at = time.monotonic() - 2
    success, detail = machine._execute_named(request, task_profile="pick")
    assert not success and "stale" in detail
    assert not calls


def test_inference_failure_stops_before_grasp(workflow):
    machine, request, calls, _ = workflow
    async def fail(*args):
        raise RuntimeError("object not found")
    machine._request_remote_grasp = fail
    success, detail = machine._execute_named(request, task_profile="pick")
    assert not success and "object not found" in detail
    assert [kind for kind, _ in calls] == ["joint", "joint"]


@pytest.mark.parametrize("failure", ["retreat", "return", "cancel"])
def test_failed_or_canceled_carry_never_releases(workflow, failure):
    machine, request, calls, _ = workflow
    cartesian = machine._call_cartesian_plan
    joint = machine._call_joint_plan
    def cartesian_result(**kwargs):
        result = cartesian(**kwargs)
        if sum(kind == "cartesian" for kind, _ in calls) == 2:
            if failure == "retreat":
                return False, "blocked retreat"
            if failure == "cancel":
                machine._is_cancel_requested = lambda: True
        return result
    def joint_result(**kwargs):
        result = joint(**kwargs)
        if len(calls) > 1 and not kwargs.get("gripper_command", False) and failure == "return":
            return False, "blocked return"
        return result
    machine._call_cartesian_plan = cartesian_result
    machine._call_joint_plan = joint_result
    success, detail = machine._execute_named(request, task_profile="pick")
    assert not success, detail
    gripper = [kw for kind, kw in calls if kind == "joint" and kw.get("gripper_command")]
    assert len(gripper) == 2
    assert gripper[-1]["gripper_opening_m"] == 0
    if failure != "return":
        assert not any(kind == "joint" and not kw.get("gripper_command", False) for kind, kw in calls[1:])


@pytest.mark.parametrize("outcome", ["success", "return_failed", "cancel", "execution_failed"])
def test_rejected_approach_recovers_only_before_execution(workflow, outcome):
    machine, request, calls, initial_joints = workflow
    def failed_approach(**kwargs):
        calls.append(("cartesian", kwargs))
        if outcome == "cancel":
            machine._is_cancel_requested = lambda: True
        detail = ("trajectory execution failed" if outcome == "execution_failed"
                  else "Cartesian path incomplete: fraction=0.588235, MoveIt error=1")
        return False, detail
    machine._call_cartesian_plan = failed_approach
    joint = machine._call_joint_plan
    def joint_result(**kwargs):
        result = joint(**kwargs)
        if len(calls) > 1 and outcome == "return_failed" and not kwargs.get("gripper_command", False):
            return False, "no collision-free return"
        return result
    machine._call_joint_plan = joint_result
    success, detail = machine._execute_named(request, task_profile="pick")
    assert not success
    returns = [kw for kind, kw in calls[1:] if kind == "joint" and not kw.get("gripper_command", False)]
    closes = [kw for kind, kw in calls if kind == "joint" and kw.get("gripper_command", False) and kw["gripper_opening_m"] == 0]
    assert not closes
    if outcome in ("cancel", "execution_failed"):
        assert not returns
    else:
        assert len(returns) == 1
        assert tuple(returns[0]["joint_positions"]) == initial_joints
        assert returns[0]["keep_grasp_ellipsoid"] is False
        assert ("已返回固定" if outcome == "success" else "返回起始位置也失败") in detail


def test_fixed_start_never_uses_current_joints_as_target(workflow):
    machine, request, calls, fixed = workflow
    machine._latest_joint_state.position = [0.7, 0.3, -0.8, 0.1, 0.6, 0.2]
    success, detail = machine._execute_named(request, task_profile="pick")
    assert success, detail
    assert tuple(calls[0][1]["joint_positions"]) == fixed
    assert tuple(calls[-1][1]["joint_positions"]) == fixed


def test_fixed_start_missing_configuration_fails_before_motion(workflow):
    machine, request, calls, _ = workflow
    machine._configured_pick_capture_joints["right"] = None
    success, detail = machine._execute_named(request, task_profile="pick")
    assert not success and "not configured" in detail
    assert not calls


def test_fixed_start_must_be_reached_before_inference(workflow):
    machine, request, calls, _ = workflow
    def not_reached(**kwargs):
        calls.append(("joint", kwargs))
        return True, "mock success without reaching target"
    machine._call_joint_plan = not_reached
    success, detail = machine._execute_named(request, task_profile="pick")
    assert not success and "not reached" in detail
    assert len(calls) == 1
