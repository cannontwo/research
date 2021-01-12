#ifdef CANNON_BUILD_RESEARCH

#include <catch2/catch.hpp>

#include <chrono>
#include <thread>

#include <ompl/base/spaces/SO2StateSpace.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>

#include <cannon/research/parl/runner.hpp>
#include <cannon/research/parl/envs/inverted_pendulum.hpp>

using namespace cannon::research::parl;

TEST_CASE("ParlRunner", "[research]") {
  Hyperparams params;

  auto env = std::make_shared<InvertedPendulumEnvironment>();
  auto p = std::make_shared<Parl>(env->get_state_space(), env->get_action_space(), env->sample_random_refs(5), params);

  Runner r(env, p, "experiments/parl_configs/r10c10_1.yaml");

  Runner r2(env, "experiments/parl_configs/r10c10_1.yaml");
  //r2.run();
}

#endif
