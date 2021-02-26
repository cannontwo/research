#include <chrono>
#include <thread>

#include <ompl/base/spaces/SO2StateSpace.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>

#define TINYCOLORMAP_WITH_EIGEN
#include <tinycolormap.hpp>

#include <cannon/research/parl/runner.hpp>
#include <cannon/research/parl/envs/inverted_pendulum.hpp>
#include <cannon/research/parl_stability/voronoi.hpp>
#include <cannon/research/parl_stability/transition_map.hpp>
#include <cannon/research/parl_stability/lyapunov_finding.hpp>
#include <cannon/plot/plotter.hpp>
#include <cannon/graphics/random_color.hpp>

using namespace cannon::research::parl;
using namespace cannon::plot;
using namespace cannon::graphics;

int main() {

  // Create transition map using CGAL polygon affine mapping
  PWAFunc simple_pwa;

  // Example 3.3 of https://ieeexplore.ieee.org/document/1017553
  // Region 1 (CCW)
  Polygon_2 r1_poly;
  r1_poly.push_back(K::Point_2(-1.0, 4.0));
  r1_poly.push_back(K::Point_2(-4.0, 4.0));
  r1_poly.push_back(K::Point_2(-4.0, -4.0));
  r1_poly.push_back(K::Point_2(-1.0, -4.0));

  Matrix2d r1_A;
  r1_A << 0.9, -0.1,
          0.1, 1;

  Vector2d r1_c;
  r1_c << 0.0,
          -0.02;

  simple_pwa.push_back(std::make_pair(r1_poly, AutonomousLinearParams(r1_A, r1_c, 0)));
  
  // Region 2
  Polygon_2 r2_poly;
  r2_poly.push_back(K::Point_2(1.0, 0.0));
  r2_poly.push_back(K::Point_2(1.0, 4.0));
  r2_poly.push_back(K::Point_2(-1.0, 4.0));
  r2_poly.push_back(K::Point_2(-1.0, 0.0));

  Matrix2d r2_A;
  r2_A << 1.0, -0.02,
          0.02, 0.9;

  Vector2d r2_c = Vector2d::Zero();

  simple_pwa.push_back(std::make_pair(r2_poly, AutonomousLinearParams(r2_A, r2_c, 0)));
  
  // Region 3
  Polygon_2 r3_poly;
  r3_poly.push_back(K::Point_2(1.0, 0.0));
  r3_poly.push_back(K::Point_2(-1.0, 0.0));
  r3_poly.push_back(K::Point_2(-1.0, -4.0));
  r3_poly.push_back(K::Point_2(1.0, -4.0));

  Matrix2d r3_A;
  r3_A << 1.0, -0.02,
          0.02, 0.9;

  Vector2d r3_c = Vector2d::Zero();

  simple_pwa.push_back(std::make_pair(r3_poly, AutonomousLinearParams(r3_A, r3_c, 0)));

  // Region 4
  Polygon_2 r4_poly;
  r4_poly.push_back(K::Point_2(4.0, -4.0));
  r4_poly.push_back(K::Point_2(4.0, 4.0));
  r4_poly.push_back(K::Point_2(1.0, 4.0));
  r4_poly.push_back(K::Point_2(1.0, -4.0));

  Matrix2d r4_A;
  r4_A << 0.9, -0.1,
          0.1, 1;

  Vector2d r4_c;
  r4_c << 0.0,
          0.2;

  simple_pwa.push_back(std::make_pair(r4_poly, AutonomousLinearParams(r4_A, r4_c, 0)));

  auto transition_map_pair = compute_transition_map(simple_pwa);
  
  std::vector<LyapunovComponent> lyap;
  PWAFunc refined_pwa;
  double theta;
  std::tie(lyap, refined_pwa, theta) = find_lyapunov(simple_pwa,
      transition_map_pair.first, transition_map_pair.second, 10);

  Vector2d zero = Vector2d::Zero();
  log_info("Value of Lyapunov function at 0 is", evaluate_lyap(lyap, zero));

  Vector2d ones = Vector2d::Ones();
  log_info("Value of Lyapunov function at (1, 1) is", evaluate_lyap(lyap, ones));

  Vector2d neg_ones = Vector2d::Ones();
  neg_ones[0] = -1;
  log_info("Value of Lyapunov function at (-1, 1) is", evaluate_lyap(lyap, neg_ones));
}
