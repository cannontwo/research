#include <chrono>
#include <thread>

#include <ompl/base/spaces/SO2StateSpace.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>

#include <cannon/research/parl/parl.hpp>
#include <cannon/research/parl/runner.hpp>
#include <cannon/research/parl/envs/inverted_pendulum.hpp>
#include <cannon/research/parl_stability/voronoi.hpp>
#include <cannon/research/parl_stability/transition_map.hpp>

#include <cannon/plot/plotter.hpp>

using namespace cannon::research::parl;
using namespace cannon::plot;

int main() {
  Hyperparams params;

  auto env = std::make_shared<InvertedPendulumEnvironment>();

  Runner r(env, "/home/cannon/Documents/cannon/cannon/research/experiments/parl_configs/r10c10_short.yaml", true);

  r.run();

  // Plot value function
  auto parl = r.get_agent();
  auto value_func = [&](const Vector2d& x) {
    return parl->predict_value(x);
  };

  // Plot control function
  auto control_func = [&](const Vector2d& x) {
    return parl->get_action(x, true)(0, 0);
  };

  {
  Plotter plotter;
  plotter.plot(value_func, 20, -M_PI, M_PI, -8.0, 8.0);
  plotter.render();
  }

  {
  Plotter plotter;
  plotter.plot(control_func, 20, -M_PI, M_PI, -8.0, 8.0);
  plotter.render();
  }

  auto controlled_system = r.get_agent()->get_controlled_system();
  auto diagram = compute_voronoi_diagram(r.get_agent()->get_dynam_refs());
  auto parl_pwa_func = compute_parl_pwa_func(r.get_agent(), diagram);
  save_pwa(parl_pwa_func, std::string("models/parl_lqr_pwa_") +
      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())
      + std::string(".h5"));

}
