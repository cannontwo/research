#include <chrono>
#include <thread>

#include <ompl/base/spaces/SO2StateSpace.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>

#include <CGAL/Polygon_2_algorithms.h>

#include <cannon/research/parl/runner.hpp>
#include <cannon/research/parl/envs/inverted_pendulum.hpp>
#include <cannon/research/parl_stability/voronoi.hpp>
#include <cannon/research/parl_stability/transition_map.hpp>
#include <cannon/plot/plotter.hpp>
#include <cannon/graphics/random_color.hpp>

using namespace cannon::research::parl;
using namespace cannon::plot;
using namespace cannon::graphics;

bool is_inside(const Vector2d& state, const Polygon_2& poly) {
  K::Point_2 query(state[0], state[1]);
  switch (CGAL::bounded_side_2(poly.begin(), poly.end(), query, K())) {
    case CGAL::ON_BOUNDED_SIDE:
      return true;
    case CGAL::ON_BOUNDARY:
      return true;
    case CGAL::ON_UNBOUNDED_SIDE:
      return false;
  }  
}

bool check_state_transition(const std::vector<std::pair<Polygon_2,
    AutonomousLinearParams>>& pwa_func, const std::map<std::pair<unsigned int,
    unsigned int>, Polygon_2>& transition_map, const Vector2d& test_state) {

  for (unsigned int i = 0; i < pwa_func.size(); i++) {
    auto pair = pwa_func[i];
    if (is_inside(test_state, pair.first)) {
      auto next_state = (pair.second.A_ * test_state) + pair.second.c_;

      for (unsigned int j = 0; j < pwa_func.size(); j++) {
        auto pair2 = pwa_func[j];
        if (is_inside(next_state, pair2.first)) {
          if (transition_map.find(std::make_pair(i, j)) != transition_map.end()) {
            log_info("Verified transition map for state", test_state);
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
  auto diagram = compute_voronoi_diagram(r.get_agent());

  // Create transition map using CGAL polygon affine mapping
  //auto transition_map = compute_transition_map(r.get_agent(), diagram);
  auto parl_pwa_func = compute_parl_pwa_func(r.get_agent(), diagram);
  auto transition_map = compute_transition_map(parl_pwa_func);

  log_info("Computed transition map with", transition_map.first.size(), "elements");

  Plotter p;
  for (auto const& pair : transition_map.first) {
    p.plot_polygon(pair.second, generate_random_color());
  }
  p.render();

  unsigned int num_verified = 0;
  const unsigned int GRID_SIZE = 200;
  for (unsigned int i = 0; i < GRID_SIZE; i++) {
    for (unsigned int j = 0; j < GRID_SIZE; j++) {
      Vector2d test_state = Vector2d::Zero();
      test_state[0] = -M_PI + ((2.0 * M_PI) / GRID_SIZE) * i;
      test_state[1] = -8.0 + (16.0 / GRID_SIZE) * j;

      bool verified = check_state_transition(parl_pwa_func, transition_map.first, test_state);
      if (verified)
        num_verified += 1;
    }
  }

  log_info("Transition map verified on", num_verified, "/", GRID_SIZE * GRID_SIZE, "grid states");
}
