#include <gtest/gtest.h>

#include "piper_act_smoothing_bridge/smoothing_core.hpp"

namespace smoothing = piper_act_smoothing_bridge;

TEST(SmoothingCore, EstimatesFilteredClampedTargetVelocity)
{
  const smoothing::ArmVector previous{0, 0, 0, 0, 0, 0};
  const smoothing::ArmVector target{1, -1, 0.01, 0, 0, 0};
  const smoothing::ArmVector old_velocity{0, 0, 0, 0, 0, 0};
  const smoothing::ArmVector limits{0.27, 0.60, 0.30, 0.27, 0.72, 0.27};
  const auto velocity = smoothing::estimate_target_velocity(
    previous, target, old_velocity, limits, 1.0 / 30.0, 0.25);
  EXPECT_DOUBLE_EQ(velocity[0], 0.27);
  EXPECT_DOUBLE_EQ(velocity[1], -0.60);
  EXPECT_NEAR(velocity[2], 0.075, 1e-12);
}

TEST(SmoothingCore, GripperUsesDirectionSpecificLimits)
{
  EXPECT_NEAR(
    smoothing::next_gripper_position(0.02, 0.04, 0.015, 0.045, 1.0 / 180.0),
    0.02 + 0.015 / 180.0, 1e-12);
  EXPECT_NEAR(
    smoothing::next_gripper_position(0.02, 0.0, 0.015, 0.045, 1.0 / 180.0),
    0.02 - 0.045 / 180.0, 1e-12);
}

TEST(SmoothingCore, RejectsInvalidElapsedTime)
{
  const smoothing::ArmVector values{};
  EXPECT_THROW(
    smoothing::estimate_target_velocity(values, values, values, values, 0.0, 0.25),
    std::invalid_argument);
}

TEST(SmoothingCore, FeedbackGapHoldsBeforeHardAbort)
{
  EXPECT_EQ(
    smoothing::classify_feedback_age(0.10, 0.12, 0.50),
    smoothing::FeedbackFreshness::kRun);
  EXPECT_EQ(
    smoothing::classify_feedback_age(0.20, 0.12, 0.50),
    smoothing::FeedbackFreshness::kHold);
  EXPECT_EQ(
    smoothing::classify_feedback_age(0.51, 0.12, 0.50),
    smoothing::FeedbackFreshness::kAbort);
}

TEST(SmoothingCore, RejectsOverlappingFeedbackTimeouts)
{
  EXPECT_THROW(
    smoothing::classify_feedback_age(0.10, 0.50, 0.50),
    std::invalid_argument);
}
