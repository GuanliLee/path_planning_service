#ifndef PIPER_ACT_SMOOTHING_BRIDGE__SMOOTHING_CORE_HPP_
#define PIPER_ACT_SMOOTHING_BRIDGE__SMOOTHING_CORE_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace piper_act_smoothing_bridge
{

inline constexpr std::size_t kArmDofs = 6;
using ArmVector = std::array<double, kArmDofs>;

enum class FeedbackFreshness
{
  kRun,
  kHold,
  kAbort,
};

inline FeedbackFreshness classify_feedback_age(
  double age,
  double hold_timeout,
  double abort_timeout)
{
  if (!(std::isfinite(age) && age >= 0.0 &&
    std::isfinite(hold_timeout) && hold_timeout > 0.0 &&
    std::isfinite(abort_timeout) && abort_timeout > hold_timeout))
  {
    throw std::invalid_argument("invalid feedback freshness timing");
  }
  if (age > abort_timeout) {
    return FeedbackFreshness::kAbort;
  }
  if (age > hold_timeout) {
    return FeedbackFreshness::kHold;
  }
  return FeedbackFreshness::kRun;
}

inline bool finite(const ArmVector & values)
{
  return std::all_of(values.begin(), values.end(), [](double value) {
    return std::isfinite(value);
  });
}

inline ArmVector estimate_target_velocity(
  const ArmVector & previous_position,
  const ArmVector & position,
  const ArmVector & previous_velocity,
  const ArmVector & velocity_limits,
  double elapsed,
  double alpha)
{
  if (!(std::isfinite(elapsed) && elapsed > 0.0)) {
    throw std::invalid_argument("target elapsed time must be positive and finite");
  }
  if (!(std::isfinite(alpha) && alpha > 0.0 && alpha <= 1.0)) {
    throw std::invalid_argument("velocity filter alpha must be in (0, 1]");
  }
  ArmVector result{};
  for (std::size_t index = 0; index < kArmDofs; ++index) {
    const double raw = (position[index] - previous_position[index]) / elapsed;
    const double filtered = alpha * raw + (1.0 - alpha) * previous_velocity[index];
    result[index] = std::clamp(
      filtered, -velocity_limits[index], velocity_limits[index]);
  }
  return result;
}

inline double next_gripper_position(
  double current,
  double target,
  double opening_speed,
  double closing_speed,
  double period)
{
  if (!(std::isfinite(current) && std::isfinite(target) &&
    std::isfinite(opening_speed) && std::isfinite(closing_speed) &&
    std::isfinite(period) && opening_speed > 0.0 && closing_speed > 0.0 &&
    period > 0.0))
  {
    throw std::invalid_argument("invalid gripper smoothing input");
  }
  const double delta = target - current;
  const double limit = (delta >= 0.0 ? opening_speed : closing_speed) * period;
  return current + std::clamp(delta, -limit, limit);
}

}  // namespace piper_act_smoothing_bridge

#endif
