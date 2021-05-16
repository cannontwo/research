#ifdef CANNON_BUILD_RESEARCH

#include <catch2/catch.hpp>

#include <cannon/research/parl/parl.hpp>
#include <cannon/log/registry.hpp>

#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/base/spaces/SO2StateSpace.h>
#include <ompl/base/spaces/RealVectorBounds.h>
#include <ompl/control/spaces/RealVectorControlSpace.h>

using namespace cannon::research::parl;
using namespace cannon::log;

TEST_CASE("Parl", "[research]") {
  Hyperparams params;
  MatrixXd refs(2, 2);
  refs << 1.0, 3.0,
          2.0, 4.0;

  auto th_space = std::make_shared<ob::SO2StateSpace>();
  auto th_dot_space = std::make_shared<ob::RealVectorStateSpace>(1);
  ob::RealVectorBounds b(1);
  b.setLow(-8.0);
  b.setHigh(8.0);
  th_dot_space->setBounds(b);

  auto s = std::make_shared<ob::CompoundStateSpace>();
  s->addSubspace(th_space, 1.0);
  s->addSubspace(th_dot_space, 1.0);
  s->setup();

  auto a = std::make_shared<oc::RealVectorControlSpace>(s, 1);
  ob::RealVectorBounds b2(1);
  b2.setLow(-2.0);
  b2.setHigh(2.0);
  a->setBounds(b2);
  a->setup();

  Parl p(s, a, refs, params);
}

#endif
