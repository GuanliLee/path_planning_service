cmake_minimum_required(VERSION 3.16)
project(path_planning_server)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(chomp_motion_planner REQUIRED)
find_package(control_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(geometric_shapes REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(moveit_planners_chomp REQUIRED)
find_package(moveit_planners_ompl REQUIRED)
find_package(moveit_planners_stomp REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(ompl REQUIRED)
find_package(path_planning_interfaces REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(shape_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(yaml_cpp_vendor REQUIRED)
find_package(yaml-cpp REQUIRED)

find_path(MOVEIT_OMPL_INTERFACE_INCLUDE_DIR
  moveit/ompl_interface/ompl_interface.hpp
  PATH_SUFFIXES include/moveit_planners
  REQUIRED
)
find_library(MOVEIT_OMPL_INTERFACE_LIBRARY
  moveit_ompl_interface
  PATH_SUFFIXES lib
  REQUIRED
)

add_library(grasp_cylinder_fitter STATIC
  src/grasp_cylinder_fitter.cpp
)
target_include_directories(grasp_cylinder_fitter PUBLIC
  include
)

add_library(attached_mesh_geometry STATIC
  src/attached_mesh_geometry.cpp
)
target_include_directories(attached_mesh_geometry PUBLIC
  include
)
ament_target_dependencies(attached_mesh_geometry
  shape_msgs
)
target_link_libraries(attached_mesh_geometry
  Eigen3::Eigen
)

add_library(ompl_waypoint_loader STATIC
  src/ompl_waypoint_loader.cpp
)
target_include_directories(ompl_waypoint_loader PUBLIC
  include
)
ament_target_dependencies(ompl_waypoint_loader
  ament_index_cpp
  geometry_msgs
  yaml_cpp_vendor
)
target_link_libraries(ompl_waypoint_loader
  yaml-cpp
)

add_library(multi_segment_ompl_planner SHARED
  src/multi_segment_ompl_planner.cpp
)
target_include_directories(multi_segment_ompl_planner PUBLIC
  include
  ${MOVEIT_OMPL_INTERFACE_INCLUDE_DIR}
)
ament_target_dependencies(multi_segment_ompl_planner
  moveit_core
  moveit_msgs
  moveit_ros_planning
  pluginlib
  rclcpp
)
target_link_libraries(multi_segment_ompl_planner
  ${MOVEIT_OMPL_INTERFACE_LIBRARY}
  ompl::ompl
)
pluginlib_export_plugin_description_file(
  moveit_core
  multi_segment_ompl_planner_plugin.xml
)

add_library(trajectory_preference_optimizer STATIC
  src/trajectory_preference_optimizer.cpp
)
set_target_properties(trajectory_preference_optimizer PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)
target_include_directories(trajectory_preference_optimizer PUBLIC
  include
)
ament_target_dependencies(trajectory_preference_optimizer
  moveit_core
  rclcpp
)

add_library(trajectory_preference_adapter SHARED
  src/trajectory_preference_adapter.cpp
)
target_include_directories(trajectory_preference_adapter PUBLIC
  include
)
ament_target_dependencies(trajectory_preference_adapter
  moveit_core
  moveit_msgs
  moveit_ros_planning
  pluginlib
  rclcpp
)
target_link_libraries(trajectory_preference_adapter
  trajectory_preference_optimizer
)
pluginlib_export_plugin_description_file(
  moveit_core
  trajectory_preference_adapter_plugin.xml
)

add_library(trajectory_optimizer_adapter SHARED
  src/trajectory_optimizer_adapter.cpp
)
target_include_directories(trajectory_optimizer_adapter PUBLIC
  include
)
ament_target_dependencies(trajectory_optimizer_adapter
  chomp_motion_planner
  moveit_core
  moveit_msgs
  moveit_planners_stomp
  moveit_ros_planning
  pluginlib
  rclcpp
)
pluginlib_export_plugin_description_file(
  moveit_core
  trajectory_optimizer_adapter_plugin.xml
)

add_executable(path_planning_server_node
  src/main.cpp
  src/path_planning_node.cpp
  src/static_scene_loader.cpp
)

target_include_directories(path_planning_server_node PRIVATE
  include
)

ament_target_dependencies(path_planning_server_node
  ament_index_cpp
  control_msgs
  geometric_shapes
  geometry_msgs
  moveit_core
  moveit_msgs
  moveit_ros_planning_interface
  path_planning_interfaces
  rclcpp
  rclcpp_action
  sensor_msgs
  shape_msgs
  std_srvs
  tf2
  tf2_geometry_msgs
  tf2_ros
  visualization_msgs
  yaml_cpp_vendor
)
target_link_libraries(path_planning_server_node
  attached_mesh_geometry
  grasp_cylinder_fitter
  ompl_waypoint_loader
  yaml-cpp
)

install(TARGETS
  path_planning_server_node
  attached_mesh_geometry
  ompl_waypoint_loader
  multi_segment_ompl_planner
  trajectory_preference_adapter
  trajectory_preference_optimizer
  trajectory_optimizer_adapter
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(PROGRAMS
  scripts/dual_joint_state_adapter.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME dual_joint_state_adapter
)
install(PROGRAMS
  scripts/grasp_bridge_state_machine.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME grasp_bridge_state_machine
)
install(PROGRAMS
  scripts/test_plan_to_pose_v2.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME test_plan_to_pose_v2
)
install(FILES
  scripts/capture_sync.py
  scripts/handeye_transform.py
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/
  DESTINATION include
)
install(DIRECTORY config generated launch meshes
  DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
)
install(FILES .setup_assistant
  DESTINATION share/${PROJECT_NAME}
)
if(BUILD_TESTING)
  enable_testing()
  find_package(ament_cmake_pytest REQUIRED)
  add_executable(test_attached_mesh_geometry
    test/test_attached_mesh_geometry.cpp
  )
  target_link_libraries(test_attached_mesh_geometry
    attached_mesh_geometry
  )
  add_test(
    NAME test_attached_mesh_geometry
    COMMAND test_attached_mesh_geometry
  )
  add_executable(test_grasp_cylinder_fitter
    test/test_grasp_cylinder_fitter.cpp
  )
  target_link_libraries(test_grasp_cylinder_fitter
    grasp_cylinder_fitter
  )
  add_test(
    NAME test_grasp_cylinder_fitter
    COMMAND test_grasp_cylinder_fitter
  )
  add_executable(test_ompl_waypoint_loader
    test/test_ompl_waypoint_loader.cpp
  )
  target_link_libraries(test_ompl_waypoint_loader
    ompl_waypoint_loader
  )
  add_test(
    NAME test_ompl_waypoint_loader
    COMMAND test_ompl_waypoint_loader
  )
  add_executable(test_multi_segment_ompl_planner
    test/test_multi_segment_ompl_planner.cpp
  )
  target_include_directories(test_multi_segment_ompl_planner PRIVATE
    include
  )
  ament_target_dependencies(test_multi_segment_ompl_planner
    geometry_msgs
    moveit_core
    moveit_msgs
    rclcpp
  )
  target_link_libraries(test_multi_segment_ompl_planner
    multi_segment_ompl_planner
    moveit_core::moveit_test_utils
  )
  add_test(
    NAME test_multi_segment_ompl_planner
    COMMAND test_multi_segment_ompl_planner
  )
  add_executable(test_trajectory_optimizer_adapter
    test/test_trajectory_optimizer_adapter.cpp
  )
  target_include_directories(test_trajectory_optimizer_adapter PRIVATE
    include
  )
  ament_target_dependencies(test_trajectory_optimizer_adapter
    moveit_core
    moveit_msgs
    rclcpp
  )
  target_link_libraries(test_trajectory_optimizer_adapter
    trajectory_optimizer_adapter
  )
  add_test(
    NAME test_trajectory_optimizer_adapter
    COMMAND test_trajectory_optimizer_adapter
  )
  add_executable(test_trajectory_preference_adapter
    test/test_trajectory_preference_adapter.cpp
  )
  target_include_directories(test_trajectory_preference_adapter PRIVATE
    include
  )
  ament_target_dependencies(test_trajectory_preference_adapter
    moveit_core
    moveit_msgs
    rclcpp
  )
  target_link_libraries(test_trajectory_preference_adapter
    trajectory_preference_adapter
  )
  add_test(
    NAME test_trajectory_preference_adapter
    COMMAND test_trajectory_preference_adapter
  )
  add_executable(test_trajectory_preference_optimizer
    test/test_trajectory_preference_optimizer.cpp
  )
  target_include_directories(test_trajectory_preference_optimizer PRIVATE
    include
  )
  ament_target_dependencies(test_trajectory_preference_optimizer
    geometry_msgs
    moveit_core
  )
  target_link_libraries(test_trajectory_preference_optimizer
    trajectory_preference_optimizer
    moveit_core::moveit_test_utils
  )
  add_test(
    NAME test_trajectory_preference_optimizer
    COMMAND test_trajectory_preference_optimizer
  )
  ament_add_pytest_test(
    test_dual_joint_state_adapter
    test/test_dual_joint_state_adapter.py
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/scripts
  )
  ament_add_pytest_test(
    test_scene_001_geometry
    test/test_scene_001_geometry.py
  )
  ament_add_pytest_test(
    test_dual_piper_urdf
    test/test_dual_piper_urdf.py
  )
  ament_add_pytest_test(
    test_gripper_urdf
    test/test_gripper_urdf.py
  )
  ament_add_pytest_test(
    test_plan_to_pose_interface
    test/test_plan_to_pose_interface.py
  )
  ament_add_pytest_test(
    test_plan_to_pose_v3_interface
    test/test_plan_to_pose_v3_interface.py
  )
  ament_add_pytest_test(
    test_plan_to_pose_v2_client
    test/test_plan_to_pose_v2_client.py
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/scripts
  )
  ament_add_pytest_test(
    test_beverage_visual_assets
    test/test_beverage_visual_assets.py
  )
  ament_add_pytest_test(
    test_capture_sync
    test/test_capture_sync.py
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/scripts
  )
  ament_add_pytest_test(
    test_handeye_transform
    test/test_handeye_transform.py
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/scripts
  )
  ament_add_pytest_test(
    test_pick_return_to_start
    test/test_pick_return_to_start.py
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/scripts
  )
  ament_add_pytest_test(
    test_grasp_bridge_state_machine
    test/test_grasp_bridge_state_machine.py
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/scripts
  )
endif()

ament_package()
