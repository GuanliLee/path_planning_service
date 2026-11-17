#include "path_planning_server/grasp_cylinder_fitter.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <utility>

namespace path_planning_server
{

namespace
{

constexpr double kPi = 3.14159265358979323846;
constexpr int kCoarseAngleCount = 180;
constexpr int kRefinementIterations = 32;

struct Point2
{
  double x;
  double y;
};

struct Circle
{
  Point2 center{0.0, 0.0};
  double radius{-1.0};
};

struct CylinderCandidate
{
  double angle{0.0};
  double height{0.0};
  double radius{0.0};
  double axis_center{0.0};
  Point2 radial_center{0.0, 0.0};
  double volume{std::numeric_limits<double>::infinity()};
};

double squaredDistance(const Point2 & lhs, const Point2 & rhs)
{
  const double dx = lhs.x - rhs.x;
  const double dy = lhs.y - rhs.y;
  return dx * dx + dy * dy;
}

bool contains(const Circle & circle, const Point2 & point)
{
  if (circle.radius < 0.0) {
    return false;
  }
  const double radius_squared = circle.radius * circle.radius;
  const double tolerance = 1e-12 * std::max(1.0, radius_squared);
  return squaredDistance(circle.center, point) <= radius_squared + tolerance;
}

Circle diameterCircle(const Point2 & first, const Point2 & second)
{
  Circle circle;
  circle.center = {
    (first.x + second.x) * 0.5,
    (first.y + second.y) * 0.5};
  circle.radius = std::sqrt(squaredDistance(first, second)) * 0.5;
  return circle;
}

Circle circleThroughThree(
  const Point2 & first,
  const Point2 & second,
  const Point2 & third)
{
  const double determinant = 2.0 * (
    first.x * (second.y - third.y) +
    second.x * (third.y - first.y) +
    third.x * (first.y - second.y));

  if (std::abs(determinant) < 1e-15) {
    Circle best;
    for (const auto & candidate : {
        diameterCircle(first, second),
        diameterCircle(first, third),
        diameterCircle(second, third)})
    {
      if (contains(candidate, first) && contains(candidate, second) &&
        contains(candidate, third) &&
        (best.radius < 0.0 || candidate.radius < best.radius))
      {
        best = candidate;
      }
    }
    return best;
  }

  const double first_squared = first.x * first.x + first.y * first.y;
  const double second_squared = second.x * second.x + second.y * second.y;
  const double third_squared = third.x * third.x + third.y * third.y;

  Circle circle;
  circle.center.x = (
    first_squared * (second.y - third.y) +
    second_squared * (third.y - first.y) +
    third_squared * (first.y - second.y)) / determinant;
  circle.center.y = (
    first_squared * (third.x - second.x) +
    second_squared * (first.x - third.x) +
    third_squared * (second.x - first.x)) / determinant;
  circle.radius = std::sqrt(squaredDistance(circle.center, first));
  return circle;
}

Circle minimumEnclosingCircle(const std::vector<Point2> & points)
{
  Circle circle;
  for (std::size_t i = 0; i < points.size(); ++i) {
    if (contains(circle, points[i])) {
      continue;
    }
    circle = {points[i], 0.0};
    for (std::size_t j = 0; j < i; ++j) {
      if (contains(circle, points[j])) {
        continue;
      }
      circle = diameterCircle(points[i], points[j]);
      for (std::size_t k = 0; k < j; ++k) {
        if (!contains(circle, points[k])) {
          circle = circleThroughThree(points[i], points[j], points[k]);
        }
      }
    }
  }
  return circle;
}

CylinderCandidate evaluateCandidate(
  const std::vector<CylinderPoint> & points,
  const double angle)
{
  const double cosine = std::cos(angle);
  const double sine = std::sin(angle);
  double minimum_axis = std::numeric_limits<double>::infinity();
  double maximum_axis = -std::numeric_limits<double>::infinity();
  std::vector<Point2> radial_points;
  radial_points.reserve(points.size());

  for (const auto & point : points) {
    const double axis_projection = cosine * point.x + sine * point.y;
    minimum_axis = std::min(minimum_axis, axis_projection);
    maximum_axis = std::max(maximum_axis, axis_projection);
    radial_points.push_back({-sine * point.x + cosine * point.y, point.z});
  }

  const Circle radial_circle = minimumEnclosingCircle(radial_points);
  CylinderCandidate candidate;
  candidate.angle = angle;
  candidate.height = maximum_axis - minimum_axis;
  candidate.radius = radial_circle.radius;
  candidate.axis_center = (minimum_axis + maximum_axis) * 0.5;
  candidate.radial_center = radial_circle.center;
  candidate.volume = kPi * candidate.radius * candidate.radius * candidate.height;
  return candidate;
}

}  // namespace

bool fitMinimumEnclosingCylinder(
  const std::vector<CylinderPoint> & input_points,
  FittedCylinder & cylinder,
  std::string & message)
{
  if (input_points.size() < 3U) {
    message = "point cloud must contain at least three points";
    return false;
  }

  std::vector<CylinderPoint> points = input_points;
  std::mt19937 generator(0);
  std::shuffle(points.begin(), points.end(), generator);

  const double coarse_step = kPi / static_cast<double>(kCoarseAngleCount);
  CylinderCandidate best;
  for (int index = 0; index < kCoarseAngleCount; ++index) {
    const auto candidate = evaluateCandidate(points, index * coarse_step);
    if (candidate.volume < best.volume) {
      best = candidate;
    }
  }

  double left = best.angle - coarse_step;
  double right = best.angle + coarse_step;
  constexpr double golden_ratio = 0.6180339887498948482;
  double first_angle = right - golden_ratio * (right - left);
  double second_angle = left + golden_ratio * (right - left);
  CylinderCandidate first = evaluateCandidate(points, first_angle);
  CylinderCandidate second = evaluateCandidate(points, second_angle);

  for (int iteration = 0; iteration < kRefinementIterations; ++iteration) {
    if (first.volume <= second.volume) {
      right = second_angle;
      second_angle = first_angle;
      second = first;
      first_angle = right - golden_ratio * (right - left);
      first = evaluateCandidate(points, first_angle);
    } else {
      left = first_angle;
      first_angle = second_angle;
      first = second;
      second_angle = left + golden_ratio * (right - left);
      second = evaluateCandidate(points, second_angle);
    }
  }

  for (const auto & candidate : {first, second}) {
    if (candidate.volume < best.volume) {
      best = candidate;
    }
  }

  if (!std::isfinite(best.height) || !std::isfinite(best.radius) ||
    best.height <= 0.0 || best.radius <= 0.0)
  {
    message = "point cloud does not define a positive cylinder";
    return false;
  }

  const double cosine = std::cos(best.angle);
  const double sine = std::sin(best.angle);
  cylinder.height = best.height;
  cylinder.radius = best.radius;
  cylinder.center = {
    cosine * best.axis_center - sine * best.radial_center.x,
    sine * best.axis_center + cosine * best.radial_center.x,
    best.radial_center.y};

  const double half_sqrt_two = std::sqrt(0.5);
  cylinder.orientation_xyzw = {
    -sine * half_sqrt_two,
    cosine * half_sqrt_two,
    0.0,
    half_sqrt_two};
  message = "minimum enclosing cylinder fitted";
  return true;
}

}  // namespace path_planning_server
