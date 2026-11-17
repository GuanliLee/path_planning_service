import math

import pytest
from sensor_msgs_py import point_cloud2

import test_plan_to_pose_v2 as client


def dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def subtract(left, right):
    return tuple(a - b for a, b in zip(left, right))


def test_generated_cylinder_matches_fitter_convention():
    args = client.build_parser().parse_args(["--mode", "cylinder"])
    request = client.make_request(args)
    cloud = request.grasp_object.cylinder_input.point_cloud
    grasp = request.grasp_object.cylinder_input.grasp_point

    assert cloud.width * cloud.height == 168
    quaternion = client.normalized_quaternion(
        (args.target_qx, args.target_qy, args.target_qz, args.target_qw)
    )
    axis = client.rotate((1.0, 0.0, 0.0), quaternion)
    grasp_values = (grasp.x, grasp.y, grasp.z)
    axial_values = []
    radial_values = []
    for point in point_cloud2.read_points_list(cloud, field_names=("x", "y", "z")):
        offset = subtract(tuple(point), grasp_values)
        axial = dot(offset, axis)
        radial = math.sqrt(max(0.0, dot(offset, offset) - axial * axial))
        axial_values.append(axial)
        radial_values.append(radial)

    assert max(axial_values) - min(axial_values) == pytest.approx(0.10, abs=1e-6)
    assert max(radial_values) == pytest.approx(0.025, abs=1e-6)
    assert abs(dot(axis, client.rotate((0.0, 0.0, 1.0), quaternion))) < 1e-12


def test_generated_mesh_axis_and_empty_cache_request():
    args = client.build_parser().parse_args(["--mode", "mesh"])
    request = client.make_request(args)
    mesh = request.grasp_object.mesh_input
    axis = (
        mesh.axis_end_world.x - mesh.axis_start_world.x,
        mesh.axis_end_world.y - mesh.axis_start_world.y,
        mesh.axis_end_world.z - mesh.axis_start_world.z,
    )
    assert math.sqrt(dot(axis, axis)) == pytest.approx(0.10)

    reuse_args = client.build_parser().parse_args(
        ["--mode", "mesh", "--reuse-cache"]
    )
    reuse_request = client.make_request(reuse_args)
    reuse_mesh = reuse_request.grasp_object.mesh_input
    assert reuse_mesh.axis_start_world.x == 0.0
    assert reuse_mesh.axis_end_world.x == 0.0
    assert reuse_mesh.grasp_point_world.x == 0.0
