#include <chrono>
#include <thread>

#include <ompl/base/spaces/SO2StateSpace.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>

#include <cannon/research/parl/runner.hpp>
#include <cannon/research/parl/envs/inverted_pendulum.hpp>
#include <cannon/research/parl_stability/voronoi.hpp>
#include <cannon/research/parl_stability/transition_map.hpp>
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
  //auto transition_map = compute_transition_map(r.get_agent(), diagram);
  auto parl_pwa_func = compute_parl_pwa_func(r.get_agent(), diagram);
  auto transition_map = compute_transition_map(parl_pwa_func);

  log_info("Computed transition map with", transition_map.size(), "elements");

  Plotter p;
  for (auto const& pair : transition_map) {
    p.plot_polygon(pair.second, generate_random_color());
  }
  p.render();

  // TODO Formulate and solve iterative LP giving Lyapunov function
  // (https://ieeexplore.ieee.org/document/6426761)
}
