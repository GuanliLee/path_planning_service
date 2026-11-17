#include <ruckig/ruckig.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_srvs/srv/set_bool.hpp"

#include "piper_act_smoothing_bridge/smoothing_core.hpp"

namespace smoothing = piper_act_smoothing_bridge;
using SteadyClock = std::chrono::steady_clock;

class ActSmoothingBridge : public rclcpp::Node
{
public:
  ActSmoothingBridge()
  : Node("act_smoothing_bridge"),
    command_rate_(declare_parameter<double>("command_rate", 180.0)),
    period_(1.0 / command_rate_),
    target_timeout_(declare_parameter<double>("target_timeout", 0.5)),
    state_timeout_(declare_parameter<double>("state_timeout", 0.12)),
    state_abort_timeout_(declare_parameter<double>("state_abort_timeout", 0.5)),
    start_tolerance_(declare_parameter<double>("start_tolerance", 0.005)),
    velocity_alpha_(declare_parameter<double>("target_velocity_alpha", 0.25)),
    gripper_opening_speed_(declare_parameter<double>("gripper_opening_speed", 0.015)),
    gripper_closing_speed_(declare_parameter<double>("gripper_closing_speed", 0.045)),
    arm_speed_(declare_parameter<double>("arm_speed", 100.0)),
    gripper_effort_(declare_parameter<double>("gripper_effort", 5.0)),
    shadow_mode_(declare_parameter<bool>("shadow_mode", false)),
    target_topic_(declare_parameter<std::string>("target_topic", "/act/right/target")),
    state_topic_(declare_parameter<std::string>("state_topic", "/joint_states_right")),
    command_topic_(declare_parameter<std::string>("command_topic", "/joint_ctrl_cmd_right")),
    mirror_topic_(declare_parameter<std::string>("mirror_topic", "/act/right/smoothed_command")),
    otg_(period_)
  {
    if (!(std::isfinite(command_rate_) && command_rate_ > 0.0 &&
      std::isfinite(target_timeout_) && target_timeout_ > 0.0 &&
      std::isfinite(state_timeout_) && state_timeout_ > 0.0 &&
      std::isfinite(state_abort_timeout_) && state_abort_timeout_ > state_timeout_ &&
      std::isfinite(start_tolerance_) && start_tolerance_ > 0.0))
    {
      throw std::invalid_argument("invalid ACT smoothing timing or start limits");
    }
    velocity_limits_ = vector_parameter(
      "max_velocity", {0.27, 0.60, 0.30, 0.27, 0.72, 0.27});
    acceleration_limits_ = vector_parameter(
      "max_acceleration", {1.5, 1.5, 1.5, 1.5, 1.5, 1.5});
    jerk_limits_ = vector_parameter(
      "max_jerk", {30.0, 30.0, 30.0, 30.0, 30.0, 30.0});
    target_gripper_effort_ = gripper_effort_;

    input_.max_velocity = velocity_limits_;
    input_.max_acceleration = acceleration_limits_;
    input_.max_jerk = jerk_limits_;

    target_subscription_ = create_subscription<sensor_msgs::msg::JointState>(
      target_topic_, rclcpp::QoS(rclcpp::KeepLast(1)).reliable(),
      std::bind(&ActSmoothingBridge::target_callback, this, std::placeholders::_1));
    state_subscription_ = create_subscription<sensor_msgs::msg::JointState>(
      state_topic_, rclcpp::QoS(rclcpp::KeepLast(1)).reliable(),
      std::bind(&ActSmoothingBridge::state_callback, this, std::placeholders::_1));
    mirror_publisher_ = create_publisher<sensor_msgs::msg::JointState>(
      mirror_topic_, rclcpp::QoS(rclcpp::KeepLast(1)).reliable());
    arm_service_ = create_service<std_srvs::srv::SetBool>(
      "act_smoothing_arm",
      std::bind(
        &ActSmoothingBridge::arm_callback, this,
        std::placeholders::_1, std::placeholders::_2));
    control_thread_ = std::thread(&ActSmoothingBridge::control_loop, this);
    RCLCPP_INFO(
      get_logger(),
      "ACT smoothing bridge ready: %s (30 Hz) -> %s (%.1f Hz); "
      "feedback hold=%.3fs abort=%.3fs; hardware publisher disarmed",
      target_topic_.c_str(), command_topic_.c_str(), command_rate_, state_timeout_,
      state_abort_timeout_);
  }

  ~ActSmoothingBridge() override
  {
    stop_.store(true);
    if (control_thread_.joinable()) {
      control_thread_.join();
    }
    std::lock_guard<std::mutex> lock(mutex_);
    command_publisher_.reset();
  }

private:
  static constexpr std::array<const char *, 7> kJointNames{
    "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"};

  smoothing::ArmVector vector_parameter(
    const std::string & name, const std::vector<double> & defaults)
  {
    const auto values = declare_parameter<std::vector<double>>(name, defaults);
    if (values.size() != smoothing::kArmDofs) {
      throw std::invalid_argument(name + " must contain six values");
    }
    smoothing::ArmVector result{};
    std::copy(values.begin(), values.end(), result.begin());
    if (!smoothing::finite(result) ||
      std::any_of(result.begin(), result.end(), [](double value) {return value <= 0.0;}))
    {
      throw std::invalid_argument(name + " values must be positive and finite");
    }
    return result;
  }

  static std::optional<std::array<double, 7>> ordered_positions(
    const sensor_msgs::msg::JointState & message)
  {
    if (message.name.size() != message.position.size()) {
      return std::nullopt;
    }
    std::unordered_map<std::string, double> positions;
    for (std::size_t index = 0; index < message.name.size(); ++index) {
      positions[message.name[index]] = message.position[index];
    }
    std::array<double, 7> result{};
    for (std::size_t index = 0; index < kJointNames.size(); ++index) {
      const auto item = positions.find(kJointNames[index]);
      if (item == positions.end() || !std::isfinite(item->second)) {
        return std::nullopt;
      }
      result[index] = item->second;
    }
    if (std::any_of(result.begin(), result.begin() + 6, [](double value) {
      return std::abs(value) > 3.5;
    }) || result[6] < 0.0 || result[6] > 0.0400001)
    {
      return std::nullopt;
    }
    return result;
  }

  void target_callback(const sensor_msgs::msg::JointState::SharedPtr message)
  {
    const auto ordered = ordered_positions(*message);
    if (!ordered.has_value()) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 1000, "Rejected malformed ACT target");
      return;
    }
    const auto now = SteadyClock::now();
    std::lock_guard<std::mutex> lock(mutex_);
    smoothing::ArmVector arm{};
    std::copy_n(ordered->begin(), 6, arm.begin());
    if (target_received_) {
      const double elapsed = std::chrono::duration<double>(now - target_time_).count();
      if (elapsed >= 0.01 && elapsed <= 0.2) {
        target_velocity_ = smoothing::estimate_target_velocity(
          target_position_, arm, target_velocity_, velocity_limits_, elapsed, velocity_alpha_);
      } else {
        target_velocity_.fill(0.0);
      }
    } else {
      target_velocity_.fill(0.0);
    }
    target_position_ = arm;
    target_gripper_ = (*ordered)[6];
    if (message->effort.size() >= 7 && std::isfinite(message->effort[6])) {
      target_gripper_effort_ = std::clamp(message->effort[6], 0.5, 5.0);
    }
    target_time_ = now;
    target_received_ = true;
  }

  void state_callback(const sensor_msgs::msg::JointState::SharedPtr message)
  {
    const auto ordered = ordered_positions(*message);
    if (!ordered.has_value()) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    hardware_position_ = *ordered;
    state_time_ = SteadyClock::now();
    state_received_ = true;
  }

  void arm_callback(
    const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
    std::shared_ptr<std_srvs::srv::SetBool::Response> response)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!request->data) {
      armed_ = false;
      command_publisher_.reset();
      response->success = true;
      response->message = "ACT smoothing bridge disarmed";
      return;
    }
    const auto now = SteadyClock::now();
    if (!target_received_ || !state_received_ ||
      std::chrono::duration<double>(now - target_time_).count() > target_timeout_ ||
      std::chrono::duration<double>(now - state_time_).count() > state_timeout_)
    {
      response->success = false;
      response->message = "fresh ACT target and hardware state are required";
      return;
    }
    double start_error = 0.0;
    for (std::size_t index = 0; index < 6; ++index) {
      start_error = std::max(
        start_error, std::abs(target_position_[index] - hardware_position_[index]));
    }
    if (start_error > start_tolerance_ ||
      std::abs(target_gripper_ - hardware_position_[6]) > start_tolerance_)
    {
      response->success = false;
      response->message = "ACT target does not match fresh hardware start state";
      return;
    }
    const auto existing = get_publishers_info_by_topic(command_topic_);
    if (!shadow_mode_ && !existing.empty()) {
      response->success = false;
      response->message = "another hardware command publisher already exists";
      return;
    }
    input_.current_position = {
      hardware_position_[0], hardware_position_[1], hardware_position_[2],
      hardware_position_[3], hardware_position_[4], hardware_position_[5]};
    input_.current_velocity.fill(0.0);
    input_.current_acceleration.fill(0.0);
    input_.target_position = target_position_;
    input_.target_velocity.fill(0.0);
    input_.target_acceleration.fill(0.0);
    gripper_command_ = hardware_position_[6];
    target_velocity_.fill(0.0);
    if (!shadow_mode_) {
      command_publisher_ = create_publisher<sensor_msgs::msg::JointState>(command_topic_, 10);
    }
    armed_ = true;
    response->success = true;
    response->message = shadow_mode_ ?
      "ACT smoothing bridge armed in shadow mode" : "ACT smoothing bridge armed";
  }

  sensor_msgs::msg::JointState command_message(bool mirror) const
  {
    sensor_msgs::msg::JointState message;
    message.header.stamp = get_clock()->now();
    message.name.assign(kJointNames.begin(), kJointNames.end());
    message.position.assign(input_.current_position.begin(), input_.current_position.end());
    message.position.push_back(gripper_command_);
    if (mirror) {
      message.velocity.assign(input_.current_velocity.begin(), input_.current_velocity.end());
      message.velocity.push_back(0.0);
      message.effort.assign(input_.current_acceleration.begin(), input_.current_acceleration.end());
      message.effort.push_back(target_gripper_effort_);
    } else {
      message.velocity.assign(6, 0.0);
      message.velocity.push_back(arm_speed_);
      message.effort.assign(6, 0.0);
      message.effort.push_back(target_gripper_effort_);
    }
    return message;
  }

  void fail_closed_locked(const std::string & reason)
  {
    if (armed_) {
      RCLCPP_ERROR(get_logger(), "ACT smoothing bridge disarmed: %s", reason.c_str());
    }
    armed_ = false;
    feedback_hold_active_ = false;
    command_publisher_.reset();
  }

  void control_loop()
  {
    const auto period = std::chrono::duration_cast<SteadyClock::duration>(
      std::chrono::duration<double>(period_));
    auto deadline = SteadyClock::now() + period;
    while (rclcpp::ok() && !stop_.load()) {
      std::optional<sensor_msgs::msg::JointState> command;
      std::optional<sensor_msgs::msg::JointState> mirror;
      rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr publisher;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (armed_) {
          const auto now = SteadyClock::now();
          const double target_age =
            std::chrono::duration<double>(now - target_time_).count();
          const double state_age =
            std::chrono::duration<double>(now - state_time_).count();
          if (target_age > target_timeout_) {
            fail_closed_locked("ACT target timeout");
          } else {
            const auto feedback_freshness = smoothing::classify_feedback_age(
              state_age, state_timeout_, state_abort_timeout_);
            if (feedback_freshness == smoothing::FeedbackFreshness::kAbort) {
              fail_closed_locked("persistent hardware feedback timeout");
            } else if (feedback_freshness == smoothing::FeedbackFreshness::kHold) {
              if (!feedback_hold_active_) {
                feedback_hold_active_ = true;
                feedback_hold_started_ = now;
                input_.current_velocity.fill(0.0);
                input_.current_acceleration.fill(0.0);
                RCLCPP_WARN(
                  get_logger(),
                  "Hardware feedback age %.3fs exceeded %.3fs; holding the last safe command",
                  state_age, state_timeout_);
              }
              command = command_message(false);
              mirror = command_message(true);
              publisher = command_publisher_;
            } else {
              if (feedback_hold_active_) {
                const double held_for =
                  std::chrono::duration<double>(now - feedback_hold_started_).count();
                RCLCPP_WARN(
                  get_logger(),
                  "Hardware feedback recovered after %.3fs; resuming the bounded trajectory",
                  held_for);
                feedback_hold_active_ = false;
              }
              input_.target_position = target_position_;
              input_.target_velocity = target_velocity_;
              input_.target_acceleration.fill(0.0);
              auto result = otg_.update(input_, output_);
              if (static_cast<int>(result) < 0) {
                input_.target_velocity.fill(0.0);
                result = otg_.update(input_, output_);
              }
              if (static_cast<int>(result) < 0) {
                fail_closed_locked("online trajectory generation failed");
              } else {
                output_.pass_to_input(input_);
                gripper_command_ = smoothing::next_gripper_position(
                  gripper_command_, target_gripper_, gripper_opening_speed_,
                  gripper_closing_speed_, period_);
                command = command_message(false);
                mirror = command_message(true);
                publisher = command_publisher_;
              }
            }
          }
        }
      }
      if (publisher && command.has_value()) {
        publisher->publish(*command);
      }
      if (mirror.has_value()) {
        mirror_publisher_->publish(*mirror);
      }
      std::this_thread::sleep_until(deadline);
      const auto now = SteadyClock::now();
      deadline = now > deadline + period ? now + period : deadline + period;
    }
  }

  double command_rate_;
  double period_;
  double target_timeout_;
  double state_timeout_;
  double state_abort_timeout_;
  double start_tolerance_;
  double velocity_alpha_;
  double gripper_opening_speed_;
  double gripper_closing_speed_;
  double arm_speed_;
  double gripper_effort_;
  bool shadow_mode_;
  std::string target_topic_;
  std::string state_topic_;
  std::string command_topic_;
  std::string mirror_topic_;
  smoothing::ArmVector velocity_limits_{};
  smoothing::ArmVector acceleration_limits_{};
  smoothing::ArmVector jerk_limits_{};
  smoothing::ArmVector target_position_{};
  smoothing::ArmVector target_velocity_{};
  std::array<double, 7> hardware_position_{};
  double target_gripper_{0.04};
  double gripper_command_{0.04};
  double target_gripper_effort_{5.0};
  bool target_received_{false};
  bool state_received_{false};
  bool armed_{false};
  bool feedback_hold_active_{false};
  SteadyClock::time_point target_time_{};
  SteadyClock::time_point state_time_{};
  SteadyClock::time_point feedback_hold_started_{};
  ruckig::Ruckig<6> otg_;
  ruckig::InputParameter<6> input_;
  ruckig::OutputParameter<6> output_;
  std::mutex mutex_;
  std::atomic<bool> stop_{false};
  std::thread control_thread_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr target_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr state_subscription_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr command_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr mirror_publisher_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr arm_service_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ActSmoothingBridge>());
  rclcpp::shutdown();
  return 0;
}
