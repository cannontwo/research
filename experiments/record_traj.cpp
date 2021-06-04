#include <iostream>
#include <fstream>
#include <filesystem> // C++17

#include <Eigen/Dense>

#include <cannon/research/parl/envs/inverted_pendulum.hpp>
#include <cannon/math/multivariate_normal.hpp>

using namespace Eigen;

using namespace cannon::research::parl;
using namespace cannon::math;

int main() {
  std::string log_path = std::string("logs/");
  std::filesystem::create_directory(log_path);
  std::ofstream traj_file(log_path + "test_pend_traj.csv", std::ios::trunc);

  InvertedPendulumEnvironment env;

  VectorXd state = env.reset();

  traj_file << "th,thdot,torque,reward,done" << std::endl;

  int action_dim = env.get_action_space()->getDimension();
  MultivariateNormal d(MatrixXd::Identity(action_dim, action_dim));

  for (int i = 0; i < 200; i++) {
    VectorXd new_state;
    double reward;
    bool done;

    VectorXd action = d.sample();
    std::tie(new_state, reward, done) = env.step(action);

    traj_file << state[0] << "," <<
                 state[1] << "," <<
                 action[0] << "," <<
                 reward << "," <<
                 done << std::endl;


    state = new_state;
  }
}
