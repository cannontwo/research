#include <chrono>
#include <thread>

#include <ompl/base/spaces/SO2StateSpace.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>

#include <CGAL/Polygon_2_algorithms.h>

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

bool check_state_transition(const std::vector<std::pair<Polygon_2,
    AutonomousLinearParams>>& pwa_func, const TransitionMap& transition_map, const Vector2d& test_state) {

  for (unsigned int i = 0; i < pwa_func.size(); i++) {
    auto pair = pwa_func[i];
    if (is_inside(test_state, pair.first)) {
      auto next_state = (pair.second.A_ * test_state) + pair.second.c_;

      for (unsigned int j = 0; j < pwa_func.size(); j++) {
        auto pair2 = pwa_func[j];
        if (is_inside(next_state, pair2.first)) {
          if (transition_map.find(std::make_pair(i, j)) != transition_map.end()) {
            //log_info("Verified transition map for state", test_state);
            return true;
          } else {
            throw std::runtime_error("State transition bad because next state not in appropriate polygon");
          }
        }
      }

      // This means that the point gets mapped outside the bounded region
      // TODO We should be keeping track of these sets (\Omega_{ip}) as well
      return false;
    }
  }

  return false;
}

int main() {
  Hyperparams params;

  auto env = std::make_shared<InvertedPendulumEnvironment>();

  Runner r(env,
      "/home/cannon/Documents/cannon/cannon/research/experiments/parl_configs/r10c10_short.yaml",
      false);
  r.run();

  auto controlled_system = r.get_agent()->get_controlled_system();
  log_info(controlled_system[0].A_);

  // Compute polytopal representation of Voronoi diagram of PARL refs
  auto diagram = compute_voronoi_diagram(r.get_agent()->get_refs());

  // Create transition map using CGAL polygon affine mapping
  auto current_pwa = compute_parl_pwa_func(r.get_agent(), diagram);
  auto transition_map_pair = compute_transition_map(current_pwa);

  auto current_transition_map = transition_map_pair.first;
  auto current_out_map = transition_map_pair.second;

  for (unsigned int i = 0; i < 5; i++) {
    log_info("Computed transition map with", current_transition_map.size(), "elements");

    // Do actual test
    unsigned int num_verified = 0;
    const unsigned int GRID_SIZE = 200;
    for (unsigned int i = 0; i < GRID_SIZE; i++) {
      for (unsigned int j = 0; j < GRID_SIZE; j++) {
        Vector2d test_state = Vector2d::Zero();
        test_state[0] = -M_PI + ((2.0 * M_PI) / GRID_SIZE) * i;
        test_state[1] = -8.0 + (16.0 / GRID_SIZE) * j;

        bool verified = check_state_transition(current_pwa, current_transition_map, test_state);
        if (verified)
          num_verified += 1;
      }
    }

    log_info("On iteration", i, "transition map verified on", num_verified, "/", GRID_SIZE * GRID_SIZE, "grid states");

    log_info("Refining polygon map");
    std::tie(current_pwa, current_transition_map, current_out_map) =
      refine_pwa(current_pwa, current_transition_map, current_out_map);
  }
}

