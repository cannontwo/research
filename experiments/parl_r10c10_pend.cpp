#include <chrono>
#include <thread>

#include <ompl/base/spaces/SO2StateSpace.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>

#include <cannon/research/parl/parl.hpp>
#include <cannon/research/parl/runner.hpp>
#include <cannon/research/parl/envs/inverted_pendulum.hpp>

using namespace cannon::research::parl;

int main() {
  Hyperparams params;

  auto env = std::make_shared<InvertedPendulumEnvironment>();

  Runner r(env, "/home/cannon/Documents/cannon/cannon/research/experiments/parl_configs/r10c10_short.yaml",false);
  r.run();
}
