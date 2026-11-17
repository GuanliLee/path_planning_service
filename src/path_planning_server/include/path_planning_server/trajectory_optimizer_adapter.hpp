#pragma once

#include <moveit/planning_interface/planning_response_adapter.hpp>
#include <rclcpp/node.hpp>

#include <cstdint>
#include <memory>
#include <string>

namespace path_planning_server
{
class TrajectoryOptimizerAdapter : public planning_interface::PlanningResponseAdapter
{
public:
  TrajectoryOptimizerAdapter();
  ~TrajectoryOptimizerAdapter() override;

  void initialize(const rclcpp::Node::SharedPtr& node, const std::string& parameter_namespace) override;
  std::string getDescription() const override;
  void adapt(const planning_scene::PlanningSceneConstPtr& planning_scene,
             const planning_interface::MotionPlanRequest& request,
             planning_interface::MotionPlanResponse& response) const override;

private:
  struct StompPlugin;

  enum class Mode
  {
    NONE,
    CHOMP,
    STOMP,
  };

  struct ChompSettings
  {
    double planning_time_limit{ 1.0 };
    std::int64_t max_iterations{ 200 };
    std::int64_t max_iterations_after_collision_free{ 5 };
    double smoothness_cost_weight{ 0.1 };
    double obstacle_cost_weight{ 1.0 };
    double learning_rate{ 0.01 };
    double ridge_factor{ 0.01 };
    double min_clearance{ 0.01 };
    double joint_update_limit{ 0.1 };
  };

  struct StompSettings
  {
    std::int64_t num_timesteps{ 60 };
    std::int64_t num_iterations{ 40 };
    std::int64_t num_iterations_after_valid{ 0 };
    std::int64_t num_rollouts{ 30 };
    std::int64_t max_rollouts{ 30 };
    double exponentiated_cost_sensitivity{ 0.8 };
    double control_cost_weight{ 0.1 };
    double delta_t{ 0.1 };
  };

  void runChomp(const planning_scene::PlanningSceneConstPtr& planning_scene,
                const planning_interface::MotionPlanRequest& request,
                planning_interface::MotionPlanResponse& response) const;
  void runStomp(const planning_scene::PlanningSceneConstPtr& planning_scene,
                const planning_interface::MotionPlanRequest& request,
                planning_interface::MotionPlanResponse& response) const;

  rclcpp::Node::SharedPtr node_;
  std::string parameter_namespace_;
  Mode mode_{ Mode::NONE };
  ChompSettings chomp_;
  StompSettings stomp_;
  mutable std::unique_ptr<StompPlugin> stomp_plugin_;
};
}  // namespace path_planning_server
