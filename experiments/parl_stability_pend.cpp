#include <chrono>
#include <thread>

#include <ompl/base/spaces/SO2StateSpace.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>

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
  Hyperparams params;

  auto env = std::make_shared<InvertedPendulumEnvironment>();

  Runner r(env,
      "/home/cannon/Documents/cannon/cannon/research/experiments/parl_configs/r10c10_short.yaml",
      false);
  r.run();

  auto controlled_system = r.get_agent()->get_controlled_system();
  log_info(controlled_system[0].A_);

  // TODO Delete, just for debugging
  VectorXd query = VectorXd::Zero(2);
  unsigned int zero_idx = r.get_agent()->get_nearest_ref_idx(query);
  auto zero_local_system = r.get_agent()->get_controlled_system()[zero_idx];
  auto zero_B = r.get_agent()->get_B_matrix_idx_(zero_idx);
  auto zero_k = r.get_agent()->get_k_vector_idx_(zero_idx);
  auto zero_c = zero_local_system.c_ - (zero_B * zero_k);

  auto analytic_k = -(zero_B.transpose() * zero_B).inverse() * zero_B.transpose() * zero_c;

  log_info("Affine term of region containing zero is", zero_local_system.c_);
  log_info("Analytic k is", analytic_k, ", learned k is", zero_k);
  log_info("Analytic controlled system has affine term", zero_B * analytic_k + zero_c, "\n");

  // Compute polytopal representation of Voronoi diagram of PARL refs
  auto diagram = compute_voronoi_diagram(r.get_agent());

  // Create transition map using CGAL polygon affine mapping
  auto parl_pwa_func = compute_parl_pwa_func(r.get_agent(), diagram);

  // TODO Testing
  parl_pwa_func = restrict_pwa(parl_pwa_func, 2.0);

  auto transition_map_pair = compute_transition_map(parl_pwa_func);

  log_info("Computed transition map with", transition_map_pair.first.size(), "elements");
  log_info(transition_map_pair.second.size(), "polygons mappped out of bounds");

  //Plotter p;
  //for (auto const& pair : parl_pwa_func) {
  //  p.plot_polygon(pair.first, generate_random_color());
  //}
  //for (auto const& pair : transition_map_pair.first) {
  //  float hue_1 = pair.first.first  * (360.0 / (float)parl_pwa_func.size()); 
  //  float hue_2 = pair.first.second * (360.0 / (float)parl_pwa_func.size());
  //  auto rgb = hsv_to_rgb((hue_1 + hue_2) / 2.0, 0.75, 0.9);
  //  Vector4f color = Vector4f::Ones();
  //  color.head(3) = rgb;
  //  p.plot_polygon(pair.second, color);
  //}


  //for (auto const& pair : transition_map_pair.second) {
  //  float hue = pair.first  * (360.0 / (float)parl_pwa_func.size()); 
  //  Vector4f color = Vector4f::Ones();
  //  auto rgb = hsv_to_rgb(hue, 0.75, 0.25);
  //  color.head(3) = rgb;
  //  for (auto const& poly : pair.second) {
  //    p.plot_polygon(poly, color);
  //  }
  //}
  //p.render();

  // Formulate and solve iterative LP giving Lyapunov function
  // (https://ieeexplore.ieee.org/document/6426761)
  //auto lyap = attempt_lp_solve(parl_pwa_func, transition_map_pair.first,
  //    transition_map_pair.second);
  
  // TODO Think about making a smaller PWA to feed to find_lyapunov; i.e., only
  // include regions close to 0.
  
  std::vector<LyapunovComponent> lyap;
  PWAFunc refined_pwa;
  double alpha_1, alpha_3, theta;
  std::tie(lyap, refined_pwa, alpha_1, alpha_3, theta) =
    find_lyapunov(parl_pwa_func, transition_map_pair.first,
        transition_map_pair.second, 50);

  check_lyap(lyap, refined_pwa, alpha_1, alpha_3, theta);

  Vector2d zero = Vector2d::Zero();
  log_info("Value of Lyapunov function at 0 is", evaluate_lyap(lyap, zero));

  Vector2d ones = Vector2d::Ones();
  log_info("Value of Lyapunov function at (1, 1) is", evaluate_lyap(lyap, ones));

  Vector2d neg_ones = Vector2d::Ones();
  neg_ones[0] = -1;
  log_info("Value of Lyapunov function at (-1, 1) is", evaluate_lyap(lyap, neg_ones));
}
