#include <chrono>
#include <thread>

#include <ompl/base/spaces/SO2StateSpace.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>

#include <cannon/research/parl/parl.hpp>
#include <cannon/research/parl/runner.hpp>
#include <cannon/research/parl/envs/linear.hpp>

using namespace cannon::research::parl;

int main() {
  Hyperparams params;

  MatrixXd A(2, 2);
  A << 0, 1,
       15, 0;

  MatrixXd B(2, 1);
  B << 0, 
       3;
  
  VectorXd c(Vector2d::Zero());

  MatrixXd Q(2, 2);
  Q << 1, 0,
       0, 0.1;

  MatrixXd R(1, 1);
  R << 0.001;

  auto env = std::make_shared<LQREnvironment>(A, B, c, Q, R, VectorXd::Ones(2), VectorXd::Zero(2));

  Runner r(env, "/home/cannon/Documents/cannon/cannon/research/experiments/parl_configs/r1c1_linear.yaml", true);
  r.run();
}
