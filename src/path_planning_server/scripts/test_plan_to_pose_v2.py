#!/usr/bin/env python3
"""Call /plan_to_pose_v2 with generated ellipsoid, cylinder, or mesh input."""

import argparse
import math
import sys

import rclpy
from rclpy.node import Node
from rclpy.parameter_client import AsyncParameterClient
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from path_planning_interfaces.srv import PlanToPoseV2


def normalized_quaternion(values):
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("target quaternion must be finite and non-zero")
    return tuple(value / norm for value in values)


def rotate(vector, quaternion):
    """Rotate a vector by an (x, y, z, w) unit quaternion."""
    x, y, z = vector
    qx, qy, qz, qw = quaternion
    tx = 2.0 * (qy * z - qz * y)
    ty = 2.0 * (qz * x - qx * z)
    tz = 2.0 * (qx * y - qy * x)
    return (
        x + qw * tx + qy * tz - qz * ty,
        y + qw * ty + qz * tx - qx * tz,
        z + qw * tz + qx * ty - qy * tx,
    )


def add(left, right):
    return tuple(a + b for a, b in zip(left, right))


def scale(vector, factor):
    return tuple(value * factor for value in vector)


def set_point(message, values):
    message.x, message.y, message.z = values


def build_parser():
    parser = argparse.ArgumentParser(
        description="Send a generated test request to /plan_to_pose_v2."
    )
    parser.add_argument(
        "--mode", choices=("ellipsoid", "cylinder", "mesh"), default="ellipsoid"
    )
    parser.add_argument("--arm", choices=("left", "right"), default="right")
    parser.add_argument("--scene-id", type=int, default=1)
    parser.add_argument("--target-name", default="Sprite")
    parser.add_argument("--frame", default="world")
    parser.add_argument("--target-x", type=float, default=0.25)
    parser.add_argument("--target-y", type=float, default=-0.30)
    parser.add_argument("--target-z", type=float, default=0.30)
    parser.add_argument("--target-qx", type=float, default=0.0)
    parser.add_argument("--target-qy", type=float, default=0.676)
    parser.add_argument("--target-qz", type=float, default=0.0)
    parser.add_argument("--target-qw", type=float, default=0.737)
    parser.add_argument("--tcp-offset-z", type=float, default=0.1358)
    parser.add_argument("--object-height", type=float, default=0.10)
    parser.add_argument("--object-radius", type=float, default=0.025)
    parser.add_argument(
        "--keep-grasp-obj",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--remove-obj", action="store_true")
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="Send an empty grasp-object input and reuse the server-side cache.",
    )
    parser.add_argument("--gripper-opening", type=float, default=None)
    parser.add_argument("--service-timeout", type=float, default=20.0)
    parser.add_argument("--response-timeout", type=float, default=30.0)
    parser.add_argument("--server-node", default="/path_planning_server")
    return parser


def make_request(args):
    request = PlanToPoseV2.Request()
    request.arm_name = args.arm
    request.scene_id = args.scene_id
    request.target_name = args.target_name
    request.target_pose.header.frame_id = args.frame
    request.target_pose.pose.position.x = args.target_x
    request.target_pose.pose.position.y = args.target_y
    request.target_pose.pose.position.z = args.target_z

    quaternion = normalized_quaternion(
        (args.target_qx, args.target_qy, args.target_qz, args.target_qw)
    )
    orientation = request.target_pose.pose.orientation
    orientation.x, orientation.y, orientation.z, orientation.w = quaternion

    request.keep_grasp_obj = args.keep_grasp_obj
    request.remove_obj = args.remove_obj
    request.gripper_command = args.gripper_opening is not None
    request.gripper_opening_m = args.gripper_opening or 0.0

    if args.remove_obj or args.reuse_cache or args.mode == "ellipsoid":
        return request

    target_origin = (args.target_x, args.target_y, args.target_z)
    local_z_world = rotate((0.0, 0.0, 1.0), quaternion)
    grasp_point = add(target_origin, scale(local_z_world, args.tcp_offset_z))

    if args.mode == "cylinder":
        points = []
        radial_samples = 24
        height_samples = 7
        for height_index in range(height_samples):
            local_axis = args.object_height * (
                height_index / (height_samples - 1) - 0.5
            )
            for angle_index in range(radial_samples):
                angle = 2.0 * math.pi * angle_index / radial_samples
                # The 0817 fitter constrains the cylinder axis to the TCP XY
                # plane, i.e. perpendicular to the target pose's local +Z.
                local_point = (
                    local_axis,
                    args.object_radius * math.sin(angle),
                    args.object_radius * math.cos(angle),
                )
                points.append(add(grasp_point, rotate(local_point, quaternion)))

        header = Header()
        header.frame_id = args.frame
        cylinder_input = request.grasp_object.cylinder_input
        cylinder_input.point_cloud = point_cloud2.create_cloud_xyz32(header, points)
        set_point(cylinder_input.grasp_point, grasp_point)
    else:
        half_axis = scale(local_z_world, args.object_height / 2.0)
        mesh_input = request.grasp_object.mesh_input
        set_point(mesh_input.axis_start_world, add(grasp_point, scale(half_axis, -1.0)))
        set_point(mesh_input.axis_end_world, add(grasp_point, half_axis))
        set_point(mesh_input.grasp_point_world, grasp_point)

    return request


def main():
    args = build_parser().parse_args()
    if args.scene_id < 0:
        raise SystemExit("--scene-id must be non-negative")
    if args.object_height <= 0.0 or args.object_radius <= 0.0:
        raise SystemExit("--object-height and --object-radius must be positive")

    request = make_request(args)
    rclpy.init()
    node = Node("test_plan_to_pose_v2")
    client = node.create_client(PlanToPoseV2, "/plan_to_pose_v2")
    try:
        if not client.wait_for_service(timeout_sec=args.service_timeout):
            print("/plan_to_pose_v2 is unavailable", file=sys.stderr)
            return 2

        parameter_client = AsyncParameterClient(node, args.server_node)
        if not parameter_client.wait_for_services(timeout_sec=args.service_timeout):
            print(
                f"parameter services for {args.server_node} are unavailable",
                file=sys.stderr,
            )
            return 5
        mode_future = parameter_client.get_parameters(["grasp_object.type"])
        rclpy.spin_until_future_complete(
            node, mode_future, timeout_sec=args.service_timeout
        )
        if not mode_future.done() or mode_future.exception() is not None:
            print("failed to read server grasp-object mode", file=sys.stderr)
            return 5
        server_mode = mode_future.result().values[0].string_value
        if server_mode != args.mode:
            print(
                f"mode mismatch: server={server_mode}, client={args.mode}; "
                "restart with grasp_object_type matching --mode",
                file=sys.stderr,
            )
            return 5

        future = client.call_async(request)
        rclpy.spin_until_future_complete(
            node, future, timeout_sec=args.response_timeout
        )
        if not future.done():
            print("service response timed out", file=sys.stderr)
            return 3
        if future.exception() is not None:
            print(f"service call failed: {future.exception()}", file=sys.stderr)
            return 4

        response = future.result()
        cloud = request.grasp_object.cylinder_input.point_cloud
        point_count = cloud.width * cloud.height
        trajectory_points = len(response.trajectory.joint_trajectory.points)
        print(
            f"mode={args.mode} success={response.success} "
            f"cloud_points={point_count} trajectory_points={trajectory_points}"
        )
        print(f"message={response.message}")
        return 0 if response.success else 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
