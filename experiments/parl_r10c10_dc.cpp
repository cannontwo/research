#include <chrono>
#include <thread>

#include <ompl/base/spaces/SO2StateSpace.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>

#include <cannon/research/parl/parl.hpp>
#include <cannon/research/parl/runner.hpp>
#include <cannon/research/parl/envs/dynamic_car.hpp>

using namespace cannon::research::parl;

int main() {
  Hyperparams params;

  auto env = std::make_shared<DynamicCarEnvironment>();

  Runner r(env, "/home/cannon/Documents/cannon/cannon/research/experiments/parl_configs/r10c10_dc.yaml", true);
  r.run();
}
