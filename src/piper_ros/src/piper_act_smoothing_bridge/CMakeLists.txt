cmake_minimum_required(VERSION 3.16)
project(piper_act_smoothing_bridge)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_srvs REQUIRED)

set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_PYTHON_MODULE OFF CACHE BOOL "" FORCE)
set(BUILD_CLOUD_CLIENT OFF CACHE BOOL "" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/ruckig EXCLUDE_FROM_ALL)

add_executable(piper_act_smoothing_bridge src/act_smoothing_bridge.cpp)
target_include_directories(piper_act_smoothing_bridge PRIVATE include)
target_link_libraries(piper_act_smoothing_bridge ruckig::ruckig)
ament_target_dependencies(piper_act_smoothing_bridge rclcpp sensor_msgs std_srvs)

install(TARGETS piper_act_smoothing_bridge DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_smoothing_core test/test_smoothing_core.cpp)
  target_include_directories(test_smoothing_core PRIVATE include)
endif()

ament_package()
