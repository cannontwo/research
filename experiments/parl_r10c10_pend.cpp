#include <chrono>
#include <thread>

#include <ompl/base/spaces/SO2StateSpace.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>

#include <cannon/research/parl/parl.hpp>
#include <cannon/research/parl/runner.hpp>
#include <cannon/research/parl/envs/inverted_pendulum.hpp>

#include <cannon/plot/plotter.hpp>

using namespace cannon::research::parl;
using namespace cannon::plot;

int main() {
  Hyperparams params;

  auto env = std::make_shared<InvertedPendulumEnvironment>();

  Runner r(env, "/home/cannon/Documents/cannon/cannon/research/experiments/parl_configs/r10c10_short.yaml",false);

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
  plotter.plot(value_func, 50, -M_PI, M_PI);
  plotter.render();
  }

  {
  Plotter plotter;
  plotter.plot(control_func, 50, -M_PI, M_PI);
  plotter.render();
  }

}
