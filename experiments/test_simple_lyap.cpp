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

  // Example 3.2 of https://ieeexplore.ieee.org/document/1017553
  // Region 1 (CCW)
  Polygon_2 r1_poly;
  r1_poly.push_back(K::Point_2(-4.0, -4.0));
  r1_poly.push_back(K::Point_2(0.0, 0.0));
  r1_poly.push_back(K::Point_2(-4.0, 4.0));

  Matrix2d r1_A;
  r1_A << 1.0, 0.01,
          -0.05, 0.99;

  Vector2d r1_c = Vector2d::Zero();

  simple_pwa.push_back(std::make_pair(r1_poly, AutonomousLinearParams(r1_A, r1_c, 0)));
  
  // Region 2
  Polygon_2 r2_poly;
  r2_poly.push_back(K::Point_2(-4.0, 4.0));
  r2_poly.push_back(K::Point_2(0.0, 0.0));
  r2_poly.push_back(K::Point_2(4.0, 4.0));

  Matrix2d r2_A;
  r2_A << 1.0, 0.05,
          -0.01, 0.99;

  Vector2d r2_c = Vector2d::Zero();

  simple_pwa.push_back(std::make_pair(r2_poly, AutonomousLinearParams(r2_A, r2_c, 0)));
  
  // Region 3
  Polygon_2 r3_poly;
  r3_poly.push_back(K::Point_2(4.0, 4.0));
  r3_poly.push_back(K::Point_2(0.0, 0.0));
  r3_poly.push_back(K::Point_2(4.0, -4.0));

  Matrix2d r3_A;
  r3_A << 1.0, 0.01,
          -0.05, 0.99;

  Vector2d r3_c = Vector2d::Zero();

  simple_pwa.push_back(std::make_pair(r3_poly, AutonomousLinearParams(r3_A, r3_c, 0)));

  // Region 4
  Polygon_2 r4_poly;
  r4_poly.push_back(K::Point_2(4.0, -4.0));
  r4_poly.push_back(K::Point_2(0.0, 0.0));
  r4_poly.push_back(K::Point_2(-4.0, -4.0));

  Matrix2d r4_A;
  r4_A << 1.0, 0.05,
          -0.01, 0.99;

  Vector2d r4_c = Vector2d::Zero();

  simple_pwa.push_back(std::make_pair(r4_poly, AutonomousLinearParams(r4_A, r4_c, 0)));

  auto transition_map_pair = compute_transition_map(simple_pwa);
  
  std::vector<LyapunovComponent> lyap;
  PWAFunc refined_pwa;
  double theta;
  std::tie(lyap, refined_pwa, theta) = find_lyapunov(simple_pwa,
      transition_map_pair.first, transition_map_pair.second, 10);

  unsigned int num_lower_regions = 0;
  Plotter p;
  for (unsigned int i = 0; i < refined_pwa.size(); i++) {
    bool all_lower = true;
    auto poly = refined_pwa[i].first;

    for (auto it = refined_pwa[i].first.vertices_begin(); it < refined_pwa[i].first.vertices_end(); it++) {
      Vector2d query = Vector2d::Zero();
      query[0] = CGAL::to_double(it->x());
      query[1] = CGAL::to_double(it->y());

      double lyap_val;
      try {
        lyap_val = evaluate_lyap(lyap, query);
      } catch (...) {
        log_info("Query point", query, "was not in lyapunov domain");
        lyap_val = theta;
      }
      
      //if (lyap_val >= theta) {
      //  all_lower = false;
      //  break;
      //}
    }

    if (all_lower) {
      log_info("Region", i, "included in Lyapunov PI set");

      // TODO Can this be encapsulated in Plotter / color schemes?
      for (auto it = refined_pwa[i].first.vertices_begin(); it <
          refined_pwa[i].first.vertices_end(); it++) {
        RowVector2f plot_point = RowVector2f::Zero();
        Vector2d query = Vector2d::Zero();

        query[0] = CGAL::to_double(it->x());
        query[1] = CGAL::to_double(it->y());

        plot_point[0] = CGAL::to_double(it->x());
        plot_point[1] = CGAL::to_double(it->y());

        double lyap_val = evaluate_lyap(lyap, query);
        float t = lyap_val / theta;

        Vector3d rgb = tinycolormap::GetColor(t, tinycolormap::ColormapType::Viridis).ConvertToEigen();
        Vector4f point_value_color = Vector4f::Ones();
        point_value_color[0] = rgb[0];
        point_value_color[1] = rgb[1];
        point_value_color[2] = rgb[2];
        p.plot_points(plot_point, point_value_color);
      }

      num_lower_regions += 1;
    }
  }

  log_info(num_lower_regions, "polygonal regions included in Lyapunov function PI set");
  p.render();

  Vector2d zero = Vector2d::Zero();
  log_info("Value of Lyapunov function at 0 is", evaluate_lyap(lyap, zero));

  Vector2d ones = Vector2d::Ones();
  log_info("Value of Lyapunov function at (1, 1) is", evaluate_lyap(lyap, ones));

  Vector2d neg_ones = Vector2d::Ones();
  neg_ones[0] = -1;
  log_info("Value of Lyapunov function at (-1, 1) is", evaluate_lyap(lyap, neg_ones));

  // Plot simple scatter plot of Lyapunov function
}
