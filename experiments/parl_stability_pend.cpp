#include <chrono>
#include <thread>

#include <ompl/base/spaces/SO2StateSpace.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>

#include <cannon/research/parl/runner.hpp>
#include <cannon/research/parl/envs/inverted_pendulum.hpp>
#include <cannon/research/parl_stability/voronoi.hpp>

using namespace cannon::research::parl;

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

  // TODO Create transition map using CGAL polygon affine mapping

  // TODO Formulate and solve iterative LP giving Lyapunov function
  // (https://ieeexplore.ieee.org/document/6426761)
}
